#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Persistent control state for immutable tabular structure generations."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from threading import RLock
from typing import Any

from rag.app.tabular_structure import (
    StructureGenerationConflict,
    StructureSnapshotChanged,
    StructureSnapshotMissing,
    load_tabular_structure_projection,
    page_tabular_structure_rows,
)


_GENERATION_STATUSES = {"shadow", "active", "retained", "failed"}
_PUBLIC_GENERATION_FIELDS = {
    "producer_generation_ref",
    "tenant_id",
    "kb_id",
    "document_id",
    "projection_version",
    "producer_schema_version",
    "row_count",
    "part_count",
    "status",
    "safe_error_code",
    "activated_at",
    "retained_at",
}


def _scope(tenant_id: str, dataset_id: str, document_id: str) -> tuple[str, str, str]:
    if not all(isinstance(value, str) and value for value in (tenant_id, dataset_id, document_id)):
        raise ValueError("tenant, dataset and document scope are required")
    return tenant_id, dataset_id, document_id


def _public_generation(record: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in record.items() if key in _PUBLIC_GENERATION_FIELDS}


def _validate_record(record: dict[str, Any]) -> None:
    required = {
        "producer_generation_ref",
        "tenant_id",
        "kb_id",
        "document_id",
        "projection_version",
        "producer_schema_version",
        "manifest_object_name",
        "manifest_sha256",
        "source_sha256",
        "row_count",
        "part_count",
        "status",
    }
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError("generation record is incomplete")
    if record["status"] not in _GENERATION_STATUSES:
        raise ValueError("generation status is invalid")


class InMemoryTabularStructureRepository:
    """Deterministic repository used by contract tests without a database."""

    def __init__(self):
        self._records: dict[str, dict[str, Any]] = {}
        self._authorization_scopes: set[tuple[str, str, str]] = set()
        self._lock = RLock()

    def add_authorization_scope(self, tenant_id: str, dataset_id: str, document_id: str) -> None:
        self._authorization_scopes.add(_scope(tenant_id, dataset_id, document_id))

    def is_authorized(self, tenant_id: str, dataset_id: str, document_id: str) -> bool:
        return _scope(tenant_id, dataset_id, document_id) in self._authorization_scopes

    def add_shadow(self, record: dict[str, Any]) -> dict[str, Any]:
        _validate_record(record)
        generation_ref = record["producer_generation_ref"]
        with self._lock:
            existing = self._records.get(generation_ref)
            if existing is not None:
                if existing != record:
                    raise StructureGenerationConflict("generation identity already exists with different metadata")
                return deepcopy(existing)
            self._records[generation_ref] = deepcopy(record)
            return deepcopy(record)

    def inject(self, record: dict[str, Any]) -> None:
        _validate_record(record)
        self._records[record["producer_generation_ref"]] = deepcopy(record)

    def get(self, producer_generation_ref: str) -> dict[str, Any] | None:
        record = self._records.get(producer_generation_ref)
        return deepcopy(record) if record else None

    def list_active(self, tenant_id: str, dataset_id: str, document_id: str) -> list[dict[str, Any]]:
        scope = _scope(tenant_id, dataset_id, document_id)
        return [
            deepcopy(record)
            for record in self._records.values()
            if (record["tenant_id"], record["kb_id"], record["document_id"]) == scope and record["status"] == "active"
        ]

    def activate(
        self,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        producer_generation_ref: str,
        expected_active_generation_ref: str | None,
    ) -> dict[str, Any]:
        scope = _scope(tenant_id, dataset_id, document_id)
        with self._lock:
            active = self.list_active(*scope)
            if len(active) > 1:
                raise StructureGenerationConflict("multiple active structure generations")
            active_ref = active[0]["producer_generation_ref"] if active else None
            if active_ref != expected_active_generation_ref:
                raise StructureSnapshotChanged("active generation changed")
            target = self._records.get(producer_generation_ref)
            if target is None or (target["tenant_id"], target["kb_id"], target["document_id"]) != scope:
                raise StructureSnapshotMissing("shadow structure generation is missing")
            if target["status"] != "shadow":
                raise StructureGenerationConflict("only a shadow generation can be activated")
            now = datetime.now(timezone.utc)
            if active_ref:
                self._records[active_ref]["status"] = "retained"
                self._records[active_ref]["retained_at"] = now
            target["status"] = "active"
            target["activated_at"] = now
            target["retained_at"] = None
            return deepcopy(target)

    def restore(
        self,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        retained_generation_ref: str,
        expected_active_generation_ref: str,
    ) -> dict[str, Any]:
        scope = _scope(tenant_id, dataset_id, document_id)
        with self._lock:
            active = self.list_active(*scope)
            if len(active) > 1:
                raise StructureGenerationConflict("multiple active structure generations")
            active_ref = active[0]["producer_generation_ref"] if active else None
            if active_ref != expected_active_generation_ref:
                raise StructureSnapshotChanged("active generation changed")
            target = self._records.get(retained_generation_ref)
            if target is None or (target["tenant_id"], target["kb_id"], target["document_id"]) != scope:
                raise StructureSnapshotMissing("retained structure generation is missing")
            if target["status"] != "retained":
                raise StructureGenerationConflict("only a retained generation can be restored")
            now = datetime.now(timezone.utc)
            if active_ref:
                self._records[active_ref]["status"] = "retained"
                self._records[active_ref]["retained_at"] = now
            target["status"] = "active"
            target["activated_at"] = now
            target["retained_at"] = None
            return deepcopy(target)


class PeeweeTabularStructureRepository:
    """RAGFlow database repository with document-scoped activation CAS."""

    @staticmethod
    def _models():
        from api.db.db_models import DB, Document, Knowledgebase, TabularStructureGeneration

        return DB, Document, Knowledgebase, TabularStructureGeneration

    def is_authorized(self, tenant_id: str, dataset_id: str, document_id: str) -> bool:
        _DB, Document, Knowledgebase, _Generation = self._models()
        return (
            Document.select(Document.id)
            .join(Knowledgebase, on=(Knowledgebase.id == Document.kb_id))
            .where(
                Document.id == document_id,
                Document.kb_id == dataset_id,
                Knowledgebase.tenant_id == tenant_id,
            )
            .exists()
        )

    def add_shadow(self, record: dict[str, Any]) -> dict[str, Any]:
        from peewee import IntegrityError

        _validate_record(record)
        DB, _Document, _Knowledgebase, Generation = self._models()
        existing = Generation.get_or_none(Generation.producer_generation_ref == record["producer_generation_ref"])
        if existing:
            existing_data = existing.to_dict()
            comparable = {key: existing_data.get(key) for key in record}
            if comparable != record:
                raise StructureGenerationConflict("generation identity already exists with different metadata")
            return existing_data
        try:
            with DB.atomic():
                Generation.create(**record)
        except IntegrityError:
            existing = Generation.get_by_id(record["producer_generation_ref"]).to_dict()
            comparable = {key: existing.get(key) for key in record}
            if comparable != record:
                raise StructureGenerationConflict("generation identity already exists with different metadata")
            return existing
        return Generation.get_by_id(record["producer_generation_ref"]).to_dict()

    def get(self, producer_generation_ref: str) -> dict[str, Any] | None:
        _DB, _Document, _Knowledgebase, Generation = self._models()
        record = Generation.get_or_none(Generation.producer_generation_ref == producer_generation_ref)
        return record.to_dict() if record else None

    def list_active(self, tenant_id: str, dataset_id: str, document_id: str) -> list[dict[str, Any]]:
        _DB, _Document, _Knowledgebase, Generation = self._models()
        return [
            row.to_dict()
            for row in Generation.select().where(
                Generation.tenant_id == tenant_id,
                Generation.kb_id == dataset_id,
                Generation.document_id == document_id,
                Generation.status == "active",
            )
        ]

    def activate(
        self,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        producer_generation_ref: str,
        expected_active_generation_ref: str | None,
    ) -> dict[str, Any]:
        DB, Document, _Knowledgebase, Generation = self._models()
        with DB.atomic():
            document = (
                Document.select()
                .where(Document.id == document_id, Document.kb_id == dataset_id)
                .for_update()
                .get()
            )
            if document.kb_id != dataset_id or not self.is_authorized(tenant_id, dataset_id, document_id):
                raise PermissionError("authorization scope rejected")
            active_rows = list(
                Generation.select().where(
                    Generation.tenant_id == tenant_id,
                    Generation.kb_id == dataset_id,
                    Generation.document_id == document_id,
                    Generation.status == "active",
                )
            )
            if len(active_rows) > 1:
                raise StructureGenerationConflict("multiple active structure generations")
            active_ref = active_rows[0].producer_generation_ref if active_rows else None
            if active_ref != expected_active_generation_ref:
                raise StructureSnapshotChanged("active generation changed")
            target = Generation.get_or_none(
                Generation.producer_generation_ref == producer_generation_ref,
                Generation.tenant_id == tenant_id,
                Generation.kb_id == dataset_id,
                Generation.document_id == document_id,
            )
            if target is None:
                raise StructureSnapshotMissing("shadow structure generation is missing")
            if target.status != "shadow":
                raise StructureGenerationConflict("only a shadow generation can be activated")
            now = datetime.now(timezone.utc)
            if active_rows:
                retained_count = (
                    Generation.update(status="retained", retained_at=now)
                    .where(Generation.producer_generation_ref == active_ref, Generation.status == "active")
                    .execute()
                )
                if retained_count != 1:
                    raise StructureSnapshotChanged("active generation compare-and-swap failed")
            activated_count = (
                Generation.update(status="active", activated_at=now, retained_at=None)
                .where(Generation.producer_generation_ref == producer_generation_ref, Generation.status == "shadow")
                .execute()
            )
            if activated_count != 1:
                raise StructureSnapshotChanged("shadow generation compare-and-swap failed")
            return Generation.get_by_id(producer_generation_ref).to_dict()

    def restore(
        self,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        retained_generation_ref: str,
        expected_active_generation_ref: str,
    ) -> dict[str, Any]:
        DB, Document, _Knowledgebase, Generation = self._models()
        with DB.atomic():
            document = (
                Document.select()
                .where(Document.id == document_id, Document.kb_id == dataset_id)
                .for_update()
                .get()
            )
            if document.kb_id != dataset_id or not self.is_authorized(tenant_id, dataset_id, document_id):
                raise PermissionError("authorization scope rejected")
            active_rows = list(
                Generation.select().where(
                    Generation.tenant_id == tenant_id,
                    Generation.kb_id == dataset_id,
                    Generation.document_id == document_id,
                    Generation.status == "active",
                )
            )
            if len(active_rows) > 1:
                raise StructureGenerationConflict("multiple active structure generations")
            active_ref = active_rows[0].producer_generation_ref if active_rows else None
            if active_ref != expected_active_generation_ref:
                raise StructureSnapshotChanged("active generation changed")
            target = Generation.get_or_none(
                Generation.producer_generation_ref == retained_generation_ref,
                Generation.tenant_id == tenant_id,
                Generation.kb_id == dataset_id,
                Generation.document_id == document_id,
            )
            if target is None:
                raise StructureSnapshotMissing("retained structure generation is missing")
            if target.status != "retained":
                raise StructureGenerationConflict("only a retained generation can be restored")
            now = datetime.now(timezone.utc)
            if active_ref:
                retained_count = (
                    Generation.update(status="retained", retained_at=now)
                    .where(Generation.producer_generation_ref == active_ref, Generation.status == "active")
                    .execute()
                )
                if retained_count != 1:
                    raise StructureSnapshotChanged("active generation compare-and-swap failed")
            activated_count = (
                Generation.update(status="active", activated_at=now, retained_at=None)
                .where(Generation.producer_generation_ref == retained_generation_ref, Generation.status == "retained")
                .execute()
            )
            if activated_count != 1:
                raise StructureSnapshotChanged("retained generation compare-and-swap failed")
            return Generation.get_by_id(retained_generation_ref).to_dict()


class TabularStructureService:
    @staticmethod
    def _repository(repository=None):
        return repository or PeeweeTabularStructureRepository()

    @classmethod
    def _authorize(cls, repository, tenant_id: str, dataset_id: str, document_id: str) -> None:
        if not repository.is_authorized(tenant_id, dataset_id, document_id):
            raise PermissionError("authorization scope rejected")

    @classmethod
    def register_shadow_generation(
        cls,
        storage,
        *,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        receipt: dict[str, Any],
        repository=None,
    ) -> dict[str, Any]:
        repository = cls._repository(repository)
        cls._authorize(repository, tenant_id, dataset_id, document_id)
        projection = load_tabular_structure_projection(
            storage,
            bucket=dataset_id,
            document_id=document_id,
            producer_generation_ref=receipt["producer_generation_ref"],
            manifest_object_name=receipt["manifest_object_name"],
            manifest_sha256=receipt["manifest_sha256"],
            expected_part_count=receipt["part_count"],
            tenant_id=tenant_id,
        )
        if receipt.get("row_count") != len(projection["rows"]):
            raise StructureSnapshotChanged("generation row count changed")
        record = {
            "producer_generation_ref": projection["producer_generation_ref"],
            "tenant_id": tenant_id,
            "kb_id": dataset_id,
            "document_id": document_id,
            "projection_version": projection["version"],
            "producer_schema_version": projection["producer_schema_version"],
            "manifest_object_name": receipt["manifest_object_name"],
            "manifest_sha256": receipt["manifest_sha256"],
            "source_sha256": projection["source_sha256"],
            "row_count": len(projection["rows"]),
            "part_count": receipt["part_count"],
            "status": "shadow",
            "safe_error_code": None,
            "activated_at": None,
            "retained_at": None,
        }
        return _public_generation(repository.add_shadow(record))

    @classmethod
    def get_active_generation(cls, *, tenant_id: str, dataset_id: str, document_id: str, repository=None) -> dict[str, Any]:
        repository = cls._repository(repository)
        cls._authorize(repository, tenant_id, dataset_id, document_id)
        active = repository.list_active(tenant_id, dataset_id, document_id)
        if not active:
            raise StructureSnapshotMissing("active structure generation is missing")
        if len(active) > 1:
            raise StructureGenerationConflict("multiple active structure generations")
        return _public_generation(active[0])

    @classmethod
    def activate_generation(
        cls,
        storage,
        *,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        producer_generation_ref: str,
        expected_active_generation_ref: str | None,
        repository=None,
    ) -> dict[str, Any]:
        repository = cls._repository(repository)
        cls._authorize(repository, tenant_id, dataset_id, document_id)
        target = repository.get(producer_generation_ref)
        if target is None or (
            target["tenant_id"],
            target["kb_id"],
            target["document_id"],
        ) != (tenant_id, dataset_id, document_id):
            raise StructureSnapshotMissing("shadow structure generation is missing")
        load_tabular_structure_projection(
            storage,
            bucket=dataset_id,
            document_id=document_id,
            producer_generation_ref=producer_generation_ref,
            manifest_object_name=target["manifest_object_name"],
            manifest_sha256=target["manifest_sha256"],
            expected_part_count=target["part_count"],
            tenant_id=tenant_id,
        )
        activated = repository.activate(
            tenant_id,
            dataset_id,
            document_id,
            producer_generation_ref,
            expected_active_generation_ref,
        )
        return _public_generation(activated)

    @classmethod
    def restore_retained_generation(
        cls,
        storage,
        *,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        retained_generation_ref: str,
        expected_active_generation_ref: str,
        repository=None,
    ) -> dict[str, Any]:
        repository = cls._repository(repository)
        cls._authorize(repository, tenant_id, dataset_id, document_id)
        target = repository.get(retained_generation_ref)
        if target is None or (
            target["tenant_id"],
            target["kb_id"],
            target["document_id"],
        ) != (tenant_id, dataset_id, document_id):
            raise StructureSnapshotMissing("retained structure generation is missing")
        load_tabular_structure_projection(
            storage,
            bucket=dataset_id,
            document_id=document_id,
            producer_generation_ref=retained_generation_ref,
            manifest_object_name=target["manifest_object_name"],
            manifest_sha256=target["manifest_sha256"],
            expected_part_count=target["part_count"],
            tenant_id=tenant_id,
        )
        restored = repository.restore(
            tenant_id,
            dataset_id,
            document_id,
            retained_generation_ref,
            expected_active_generation_ref,
        )
        return _public_generation(restored)

    @classmethod
    def _read_projection(
        cls,
        storage,
        *,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        producer_generation_ref: str,
        repository=None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        repository = cls._repository(repository)
        cls._authorize(repository, tenant_id, dataset_id, document_id)
        active = repository.list_active(tenant_id, dataset_id, document_id)
        if not active:
            raise StructureSnapshotMissing("active structure generation is missing")
        if len(active) > 1:
            raise StructureGenerationConflict("multiple active structure generations")
        record = active[0]
        if record["producer_generation_ref"] != producer_generation_ref:
            raise StructureSnapshotChanged("active generation changed")
        projection = load_tabular_structure_projection(
            storage,
            bucket=dataset_id,
            document_id=document_id,
            producer_generation_ref=producer_generation_ref,
            manifest_object_name=record["manifest_object_name"],
            manifest_sha256=record["manifest_sha256"],
            expected_part_count=record["part_count"],
            tenant_id=tenant_id,
        )
        return record, projection

    @classmethod
    def _read_generation_projection(
        cls,
        storage,
        *,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        producer_generation_ref: str,
        repository=None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        repository = cls._repository(repository)
        record = repository.get(producer_generation_ref)
        if record is None or (
            record["tenant_id"],
            record["kb_id"],
            record["document_id"],
        ) != (tenant_id, dataset_id, document_id):
            raise StructureSnapshotMissing("structure generation is missing")
        cls._authorize(repository, tenant_id, dataset_id, document_id)
        if record["status"] not in {"shadow", "active", "retained"}:
            raise StructureSnapshotMissing("structure generation is unavailable")
        projection = load_tabular_structure_projection(
            storage,
            bucket=dataset_id,
            document_id=document_id,
            producer_generation_ref=producer_generation_ref,
            manifest_object_name=record["manifest_object_name"],
            manifest_sha256=record["manifest_sha256"],
            expected_part_count=record["part_count"],
            tenant_id=tenant_id,
        )
        return record, projection

    @classmethod
    def read_generation_manifest(cls, storage, **kwargs) -> dict[str, Any]:
        record, projection = cls._read_generation_projection(storage, **kwargs)
        return {
            "producer_generation_ref": projection["producer_generation_ref"],
            "projection_version": projection["version"],
            "producer_schema_version": projection["producer_schema_version"],
            "structure_algorithm_version": projection["structure_algorithm_version"],
            "enumeration_rule_version": projection["enumeration_rule_version"],
            "row_count": record["row_count"],
            "tables": deepcopy(projection["tables"]),
        }

    @classmethod
    def read_generation(cls, storage, **kwargs) -> dict[str, Any]:
        record, projection = cls._read_generation_projection(storage, **kwargs)
        return {
            "status": record["status"],
            "producer_generation_ref": projection["producer_generation_ref"],
            "projection_version": projection["version"],
            "producer_schema_version": projection["producer_schema_version"],
            "structure_algorithm_version": projection["structure_algorithm_version"],
            "enumeration_rule_version": projection["enumeration_rule_version"],
            "row_count": record["row_count"],
        }

    @classmethod
    def read_generation_rows(
        cls,
        storage,
        *,
        table_ref: str,
        cursor: int = 0,
        page_size: int = 30,
        **kwargs,
    ) -> dict[str, Any]:
        _record, projection = cls._read_generation_projection(storage, **kwargs)
        return page_tabular_structure_rows(projection, table_ref=table_ref, cursor=cursor, page_size=page_size)

    @classmethod
    def read_active_manifest(cls, storage, **kwargs) -> dict[str, Any]:
        record, projection = cls._read_projection(storage, **kwargs)
        tables = deepcopy(projection["tables"])
        return {
            "producer_generation_ref": projection["producer_generation_ref"],
            "projection_version": projection["version"],
            "producer_schema_version": projection["producer_schema_version"],
            "structure_algorithm_version": projection["structure_algorithm_version"],
            "enumeration_rule_version": projection["enumeration_rule_version"],
            "row_count": record["row_count"],
            "tables": tables,
        }

    @classmethod
    def read_active_rows(
        cls,
        storage,
        *,
        table_ref: str,
        cursor: int = 0,
        page_size: int = 30,
        **kwargs,
    ) -> dict[str, Any]:
        _record, projection = cls._read_projection(storage, **kwargs)
        return page_tabular_structure_rows(projection, table_ref=table_ref, cursor=cursor, page_size=page_size)


__all__ = [
    "InMemoryTabularStructureRepository",
    "PeeweeTabularStructureRepository",
    "StructureGenerationConflict",
    "StructureSnapshotChanged",
    "StructureSnapshotMissing",
    "TabularStructureService",
]
