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
import hashlib
import hmac
import json
import re
from threading import RLock
import time
from typing import Any
import unicodedata

from rag.app.tabular_structure import (
    StructureGenerationConflict,
    StructureSnapshotChanged,
    StructureSnapshotMissing,
    load_tabular_structure_projection,
    page_tabular_structure_rows,
)


_GENERATION_STATUSES = {"shadow", "active", "retained", "failed"}
TABULAR_DISCOVERY_CONTRACT_VERSION = "discovery/v1"
TABULAR_DISCOVERY_NORMALIZATION_VERSION = "normalization/v1"
TABULAR_DISCOVERY_INDEX_SCHEMA_VERSION = "tabular-structure-index/v1"
TABULAR_DISCOVERY_RETRIEVAL_RULE = "bm25-ngram/v1"
_INDEX_TEXT_MAX_CHARS = 16_384
_UNSAFE_INDEX_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u202e]")
_PUBLIC_GENERATION_FIELDS = {
    "producer_generation_ref",
    "tenant_id",
    "kb_id",
    "document_id",
    "projection_version",
    "producer_schema_version",
    "source_sha256",
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


def _managed_generation_result(record: dict[str, Any], projection: dict[str, Any]) -> dict[str, Any]:
    """Return the complete version-bound response required by generation consumers."""
    return {
        "status": record["status"],
        "producer_generation_ref": projection["producer_generation_ref"],
        "source_sha256": projection["source_sha256"],
        "projection_version": projection["version"],
        "producer_schema_version": projection["producer_schema_version"],
        "structure_algorithm_version": projection["structure_algorithm_version"],
        "enumeration_rule_version": projection["enumeration_rule_version"],
        "row_count": record["row_count"],
    }


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


def normalize_tabular_discovery_query(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("discovery query must be text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _table_search_text(table: dict[str, Any]) -> tuple[str, str | None]:
    fields: list[str] = []
    label = table.get("table_label")
    if isinstance(label, str):
        fields.append(label)
    for context in table.get("table_context", []):
        if isinstance(context, dict):
            fields.extend(str(context.get(key, "")) for key in ("name", "value"))
    for column in table.get("ordered_columns", []):
        if not isinstance(column, dict):
            continue
        fields.extend(str(part) for part in column.get("header_path", []))
        fields.append(str(column.get("name", "")))
    text = normalize_tabular_discovery_query(" ".join(part for part in fields if part))
    if _UNSAFE_INDEX_CONTROL.search(text):
        return "", "unsafe_control_character"
    if len(text) > _INDEX_TEXT_MAX_CHARS:
        return "", "search_text_too_large"
    return text, None


def build_tabular_discovery_index_projection(
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    projection: dict[str, Any],
) -> list[dict[str, Any]]:
    generation_ref = projection["producer_generation_ref"]
    records = []
    for table in projection["tables"]:
        search_text, unsafe_reason = _table_search_text(table)
        identity_hash = hashlib.sha256(
            json.dumps(
                [tenant_id, dataset_id, document_id, generation_ref, table["table_ref"]],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "tenant_id": tenant_id,
                "kb_id": dataset_id,
                "document_id": document_id,
                "producer_generation_ref": generation_ref,
                "table_ref": table["table_ref"],
                "table_ordinal": table["table_ordinal"],
                "search_text": search_text,
                "identity_hash": identity_hash,
                "projection_status": "unsafe" if unsafe_reason else "safe",
                "unsafe_reason": unsafe_reason,
            }
        )
    return records


def _encode_discovery_cursor(
    secret: bytes,
    tenant_id: str,
    dataset_id: str,
    dataset_revision: int,
    query_digest: str,
    page_size: int,
    max_pages: int,
    page_ordinal: int,
    score_encoded: str,
    identity_hash: str,
) -> str:
    payload = json.dumps(
        [
            tenant_id,
            dataset_id,
            dataset_revision,
            query_digest,
            page_size,
            max_pages,
            page_ordinal,
            score_encoded,
            identity_hash,
        ],
        separators=(",", ":"),
    ).encode("ascii")
    signature = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return f"{payload.hex()}.{signature}"


def _decode_discovery_cursor(
    secret: bytes,
    cursor: str | None,
    tenant_id: str,
    dataset_id: str,
    dataset_revision: int,
    query_digest: str,
    page_size: int,
    max_pages: int,
) -> tuple[tuple[str, str] | None, int]:
    if cursor is None:
        return None, 1
    if not isinstance(cursor, str) or "." not in cursor:
        raise ValueError("discovery cursor is invalid")
    payload_hex, signature = cursor.split(".", 1)
    try:
        payload = bytes.fromhex(payload_hex)
        value = json.loads(payload)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("discovery cursor is invalid") from error
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("discovery cursor is invalid")
    if not isinstance(value, list) or len(value) != 9:
        raise ValueError("discovery cursor is invalid")
    if value[2] != dataset_revision:
        raise StructureSnapshotChanged("discovery index revision changed")
    if (
        value[0] != tenant_id
        or value[1] != dataset_id
        or value[3] != query_digest
        or value[4] != page_size
        or value[5] != max_pages
        or not isinstance(value[6], int)
        or not 2 <= value[6] <= max_pages
        or not isinstance(value[7], str)
        or not re.fullmatch(r"\d{16}", value[7])
        or not isinstance(value[8], str)
        or not re.fullmatch(r"[0-9a-f]{64}", value[8])
    ):
        raise ValueError("discovery cursor is invalid")
    return (value[7], value[8]), value[6]


def _encoded_lexical_score(query: str, search_text: str) -> str:
    query_terms = {term for term in query.split() if term}
    if not query_terms:
        return "0000000000000000"
    matched = sum(1 for term in query_terms if term in search_text)
    score = round((matched / len(query_terms)) * 1_000_000)
    return f"{score:016d}"


class InMemoryTabularStructureRepository:
    """Deterministic repository used by contract tests without a database."""

    def __init__(self):
        self._records: dict[str, dict[str, Any]] = {}
        self._authorization_scopes: set[tuple[str, str, str]] = set()
        self._dataset_revisions: dict[tuple[str, str], int] = {}
        self._backfill_status: dict[tuple[str, str], str] = {}
        self._table_index: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        self._backfill_cursors: dict[tuple[str, str], str | None] = {}
        self._backfill_commit_hook = None
        self._lock = RLock()

    @staticmethod
    def discovery_cursor_secret() -> bytes:
        return b"in-memory-tabular-discovery-contract-test"

    def complete_backfill(self, tenant_id: str, dataset_id: str) -> None:
        self._backfill_status[(tenant_id, dataset_id)] = "complete"
        self._backfill_cursors[(tenant_id, dataset_id)] = None

    def mark_backfill_pending(self, tenant_id: str, dataset_id: str) -> None:
        key = (tenant_id, dataset_id)
        self._backfill_status[key] = "pending"
        self._backfill_cursors[key] = None

    def backfill_state(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
        key = (tenant_id, dataset_id)
        return {
            "status": self._backfill_status.get(key, "pending"),
            "cursor": self._backfill_cursors.get(key),
        }

    def set_backfill_commit_hook(self, hook) -> None:
        self._backfill_commit_hook = hook

    def remove_active_generation(self, producer_generation_ref: str) -> None:
        self._records.pop(producer_generation_ref, None)

    def list_pending_backfill_datasets(self, limit: int) -> list[dict[str, Any]]:
        keys = sorted(
            key
            for key, status in self._backfill_status.items()
            if status == "pending"
        )
        return [
            {
                "tenant_id": tenant_id,
                "dataset_id": dataset_id,
                "cursor": self._backfill_cursors.get((tenant_id, dataset_id)),
            }
            for tenant_id, dataset_id in keys[:limit]
        ]

    def list_active_generations_for_backfill(
        self,
        tenant_id: str,
        dataset_id: str,
        after_document_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        records = sorted(
            (
                deepcopy(record)
                for record in self._records.values()
                if record["tenant_id"] == tenant_id
                and record["kb_id"] == dataset_id
                and record["status"] == "active"
                and (
                    after_document_id is None
                    or record["document_id"] > after_document_id
                )
            ),
            key=lambda record: record["document_id"],
        )
        return records[:limit]

    def commit_backfill_batch(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        expected_cursor: str | None,
        generations: list[dict[str, Any]],
        index_projections: list[list[dict[str, Any]]],
        next_cursor: str | None,
        complete: bool,
    ) -> None:
        with self._lock:
            state_key = (tenant_id, dataset_id)
            if self._backfill_cursors.get(state_key) != expected_cursor:
                raise StructureSnapshotChanged(
                    "discovery index backfill cursor changed"
                )
            hook, self._backfill_commit_hook = self._backfill_commit_hook, None
            if hook is not None:
                hook()
            for generation in generations:
                active = self.list_active(
                    tenant_id,
                    dataset_id,
                    generation["document_id"],
                )
                if (
                    len(active) != 1
                    or active[0]["producer_generation_ref"]
                    != generation["producer_generation_ref"]
                ):
                    raise StructureSnapshotChanged(
                        "active generation changed during discovery index backfill"
                    )
            revision = self.advance_dataset_revision(tenant_id, dataset_id)
            for generation, projection in zip(
                generations,
                index_projections,
                strict=True,
            ):
                document_id = generation["document_id"]
                for indexed in self._table_index.values():
                    if (
                        indexed["tenant_id"] == tenant_id
                        and indexed["kb_id"] == dataset_id
                        and indexed["document_id"] == document_id
                    ):
                        indexed["active"] = False
                for record in projection:
                    key = (
                        tenant_id,
                        dataset_id,
                        document_id,
                        generation["producer_generation_ref"],
                        record["table_ref"],
                    )
                    self._table_index[key] = {
                        **deepcopy(record),
                        "index_revision": revision,
                        "active": True,
                    }
            self._backfill_status[state_key] = "complete" if complete else "pending"
            self._backfill_cursors[state_key] = None if complete else next_cursor

    def advance_dataset_revision(self, tenant_id: str, dataset_id: str) -> int:
        key = (tenant_id, dataset_id)
        self._dataset_revisions[key] = self._dataset_revisions.get(key, 0) + 1
        return self._dataset_revisions[key]

    def seed_discovery_index(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        producer_generation_ref: str,
        table_ref: str,
        search_text: str,
        table_ordinal: int = 1,
        projection_status: str = "safe",
        unsafe_reason: str | None = None,
    ) -> None:
        revision = self.advance_dataset_revision(tenant_id, dataset_id)
        key = (tenant_id, dataset_id, document_id, producer_generation_ref, table_ref)
        self._table_index[key] = {
            "tenant_id": tenant_id,
            "kb_id": dataset_id,
            "document_id": document_id,
            "producer_generation_ref": producer_generation_ref,
            "table_ref": table_ref,
            "table_ordinal": table_ordinal,
            "search_text": search_text,
            "identity_hash": hashlib.sha256("\x00".join(key).encode()).hexdigest(),
            "index_revision": revision,
            "active": True,
            "projection_status": projection_status,
            "unsafe_reason": unsafe_reason,
        }

    def discover_active_tables(
        self,
        tenant_id: str,
        dataset_id: str,
        normalized_query: str,
        after: tuple[str, str] | None,
        limit: int,
    ) -> dict[str, Any]:
        key = (tenant_id, dataset_id)
        records = []
        unsafe = False
        for record in self._table_index.values():
            if (
                record["tenant_id"] != tenant_id
                or record["kb_id"] != dataset_id
                or not record["active"]
            ):
                continue
            if record["projection_status"] != "safe":
                unsafe = True
                continue
            score_encoded = _encoded_lexical_score(
                normalized_query,
                record["search_text"],
            )
            if score_encoded == "0000000000000000":
                continue
            records.append({**deepcopy(record), "score_encoded": score_encoded})
        records.sort(key=lambda record: (-int(record["score_encoded"]), record["identity_hash"]))
        if after is not None:
            last_score, last_identity = after
            records = [
                record
                for record in records
                if int(record["score_encoded"]) < int(last_score)
                or (
                    record["score_encoded"] == last_score
                    and record["identity_hash"] > last_identity
                )
            ]
        return {
            "index_revision": self._dataset_revisions.get(key, 0),
            "backfill_status": self._backfill_status.get(key, "pending"),
            "unsafe": unsafe,
            "records": records[:limit],
        }

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
        index_projection: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scope = _scope(tenant_id, dataset_id, document_id)
        with self._lock:
            active = self.list_active(*scope)
            if len(active) > 1:
                raise StructureGenerationConflict("multiple active structure generations")
            active_ref = active[0]["producer_generation_ref"] if active else None
            if active_ref != expected_active_generation_ref:
                raise StructureSnapshotChanged(
                    "active generation changed",
                    active_generation_ref=active_ref,
                )
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
            revision_key = (tenant_id, dataset_id)
            revision = self._dataset_revisions.get(revision_key, 0) + 1
            self._dataset_revisions[revision_key] = revision
            self._backfill_status.setdefault(revision_key, "complete")
            for record in self._table_index.values():
                if (
                    record["tenant_id"],
                    record["kb_id"],
                    record["document_id"],
                ) == scope:
                    record["active"] = False
            for record in index_projection:
                key = (
                    tenant_id,
                    dataset_id,
                    document_id,
                    producer_generation_ref,
                    record["table_ref"],
                )
                self._table_index[key] = {
                    **deepcopy(record),
                    "index_revision": revision,
                    "active": True,
                }
            return deepcopy(target)

    def restore(
        self,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        retained_generation_ref: str,
        expected_active_generation_ref: str,
        index_projection: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scope = _scope(tenant_id, dataset_id, document_id)
        with self._lock:
            active = self.list_active(*scope)
            if len(active) > 1:
                raise StructureGenerationConflict("multiple active structure generations")
            active_ref = active[0]["producer_generation_ref"] if active else None
            if active_ref != expected_active_generation_ref:
                raise StructureSnapshotChanged(
                    "active generation changed",
                    active_generation_ref=active_ref,
                )
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
            revision_key = (tenant_id, dataset_id)
            revision = self._dataset_revisions.get(revision_key, 0) + 1
            self._dataset_revisions[revision_key] = revision
            self._backfill_status.setdefault(revision_key, "complete")
            for record in self._table_index.values():
                if (
                    record["tenant_id"],
                    record["kb_id"],
                    record["document_id"],
                ) == scope:
                    record["active"] = False
            for record in index_projection:
                key = (
                    tenant_id,
                    dataset_id,
                    document_id,
                    retained_generation_ref,
                    record["table_ref"],
                )
                self._table_index[key] = {
                    **deepcopy(record),
                    "index_revision": revision,
                    "active": True,
                }
            return deepcopy(target)


class PeeweeTabularStructureRepository:
    """RAGFlow database repository with document-scoped activation CAS."""

    @staticmethod
    def _models():
        from api.db.db_models import (
            DB,
            Document,
            Knowledgebase,
            TabularStructureDatasetIndexState,
            TabularStructureGeneration,
            TabularStructureTableIndex,
        )

        return (
            DB,
            Document,
            Knowledgebase,
            TabularStructureGeneration,
            TabularStructureDatasetIndexState,
            TabularStructureTableIndex,
        )

    @staticmethod
    def discovery_cursor_secret() -> bytes:
        from common import settings

        value = settings.get_secret_key()
        if not isinstance(value, str) or len(value) < 32:
            raise RuntimeError("tabular discovery cursor secret is unavailable")
        return value.encode("utf-8")

    def is_authorized(self, tenant_id: str, dataset_id: str, document_id: str) -> bool:
        _DB, Document, Knowledgebase, _Generation, _IndexState, _TableIndex = self._models()
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
        DB, _Document, _Knowledgebase, Generation, _IndexState, _TableIndex = self._models()
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
        _DB, _Document, _Knowledgebase, Generation, _IndexState, _TableIndex = self._models()
        record = Generation.get_or_none(Generation.producer_generation_ref == producer_generation_ref)
        return record.to_dict() if record else None

    def list_active(self, tenant_id: str, dataset_id: str, document_id: str) -> list[dict[str, Any]]:
        _DB, _Document, _Knowledgebase, Generation, _IndexState, _TableIndex = self._models()
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
        index_projection: list[dict[str, Any]],
    ) -> dict[str, Any]:
        DB, Document, _Knowledgebase, Generation, TabularStructureDatasetIndexState, TabularStructureTableIndex = self._models()
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
                raise StructureSnapshotChanged(
                    "active generation changed",
                    active_generation_ref=active_ref,
                )
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
            index_revision = self._replace_active_index_projection(
                DB=DB,
                DatasetIndexState=TabularStructureDatasetIndexState,
                TableIndex=TabularStructureTableIndex,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
                producer_generation_ref=producer_generation_ref,
                index_projection=index_projection,
            )
            return Generation.get_by_id(producer_generation_ref).to_dict()

    def restore(
        self,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        retained_generation_ref: str,
        expected_active_generation_ref: str,
        index_projection: list[dict[str, Any]],
    ) -> dict[str, Any]:
        DB, Document, _Knowledgebase, Generation, TabularStructureDatasetIndexState, TabularStructureTableIndex = self._models()
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
                raise StructureSnapshotChanged(
                    "active generation changed",
                    active_generation_ref=active_ref,
                )
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
            self._replace_active_index_projection(
                DB=DB,
                DatasetIndexState=TabularStructureDatasetIndexState,
                TableIndex=TabularStructureTableIndex,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
                producer_generation_ref=retained_generation_ref,
                index_projection=index_projection,
            )
            return Generation.get_by_id(retained_generation_ref).to_dict()

    def discover_active_tables(
        self,
        tenant_id: str,
        dataset_id: str,
        normalized_query: str,
        after: tuple[str, str] | None,
        limit: int,
    ) -> dict[str, Any]:
        DB, Document, _Knowledgebase, Generation, DatasetIndexState, TableIndex = self._models()
        with DB.atomic():
            state = DatasetIndexState.get_or_none(
                DatasetIndexState.tenant_id == tenant_id,
                DatasetIndexState.kb_id == dataset_id,
            )
            if state is None:
                return {
                    "index_revision": 0,
                    "backfill_status": "pending",
                    "unsafe": False,
                    "records": [],
                }
            join_sql = (
                " FROM tabular_structure_table_index ti "
                "INNER JOIN document d ON d.id = ti.document_id AND d.kb_id = ti.kb_id "
                "INNER JOIN tabular_structure_generation g "
                "ON g.producer_generation_ref = ti.producer_generation_ref "
                "AND g.tenant_id = ti.tenant_id AND g.kb_id = ti.kb_id "
                "AND g.document_id = ti.document_id AND g.status = 'active' "
                "WHERE ti.tenant_id = %s AND ti.kb_id = %s AND ti.active = TRUE "
            )
            unsafe = DB.execute_sql(
                "SELECT 1" + join_sql + "AND ti.projection_status <> 'safe' LIMIT 1",
                (tenant_id, dataset_id),
            ).fetchone()
            score_sql = (
                "CAST(ROUND(GREATEST(MATCH(ti.search_text) AGAINST (%s IN NATURAL LANGUAGE MODE), 0) "
                "* 1000000) AS UNSIGNED)"
            )
            params: list[Any] = [normalized_query, tenant_id, dataset_id, normalized_query]
            keyset_sql = ""
            if after is not None:
                last_score, last_identity = after
                keyset_sql = (
                    f" AND ({score_sql} < %s OR ({score_sql} = %s AND ti.identity_hash > %s))"
                )
                params.extend(
                    [
                        normalized_query,
                        int(last_score),
                        normalized_query,
                        int(last_score),
                        last_identity,
                    ]
                )
            params.append(limit)
            rows = DB.execute_sql(
                "SELECT ti.document_id, ti.producer_generation_ref, ti.table_ref, "
                "ti.table_ordinal, ti.identity_hash, "
                f"{score_sql} AS score_encoded"
                + join_sql
                + "AND ti.projection_status = 'safe' "
                + "AND MATCH(ti.search_text) AGAINST (%s IN NATURAL LANGUAGE MODE) > 0"
                + keyset_sql
                + " ORDER BY score_encoded DESC, ti.identity_hash ASC LIMIT %s",
                tuple(params),
            ).fetchall()
            current_state = DatasetIndexState.get_or_none(
                DatasetIndexState.tenant_id == tenant_id,
                DatasetIndexState.kb_id == dataset_id,
            )
            if (
                current_state is None
                or int(current_state.index_revision) != int(state.index_revision)
            ):
                return {
                    "index_revision": (
                        int(current_state.index_revision)
                        if current_state is not None
                        else 0
                    ),
                    "backfill_status": (
                        current_state.backfill_status
                        if current_state is not None
                        else "pending"
                    ),
                    "unsafe": False,
                    "records": [],
                }
            return {
                "index_revision": int(state.index_revision),
                "backfill_status": state.backfill_status,
                "unsafe": unsafe is not None,
                "records": [
                    {
                        "document_id": row[0],
                        "producer_generation_ref": row[1],
                        "table_ref": row[2],
                        "table_ordinal": int(row[3]),
                        "identity_hash": row[4],
                        "score_encoded": f"{int(row[5]):016d}",
                    }
                    for row in rows
                ],
            }

    @classmethod
    def deactivate_document_index(cls, tenant_id: str, dataset_id: str, document_id: str) -> None:
        DB, _Document, _Knowledgebase, _Generation, DatasetIndexState, TableIndex = cls._models()
        state = (
            DatasetIndexState.select()
            .where(
                DatasetIndexState.tenant_id == tenant_id,
                DatasetIndexState.kb_id == dataset_id,
            )
            .for_update()
            .get_or_none()
        )
        if state is None:
            return
        revision = int(state.index_revision) + 1
        DatasetIndexState.update(index_revision=revision).where(
            DatasetIndexState.tenant_id == tenant_id,
            DatasetIndexState.kb_id == dataset_id,
            DatasetIndexState.index_revision == state.index_revision,
        ).execute()
        TableIndex.update(active=False, index_revision=revision).where(
            TableIndex.tenant_id == tenant_id,
            TableIndex.kb_id == dataset_id,
            TableIndex.document_id == document_id,
            TableIndex.active == True,  # noqa: E712
        ).execute()

    def list_pending_backfill_datasets(self, limit: int) -> list[dict[str, Any]]:
        _DB, _Document, _Knowledgebase, _Generation, DatasetIndexState, _TableIndex = self._models()
        return [
            {
                "tenant_id": row.tenant_id,
                "dataset_id": row.kb_id,
                "cursor": row.backfill_cursor,
            }
            for row in (
                DatasetIndexState.select()
                .where(DatasetIndexState.backfill_status == "pending")
                .order_by(DatasetIndexState.tenant_id, DatasetIndexState.kb_id)
                .limit(limit)
            )
        ]

    def list_active_generations_for_backfill(
        self,
        tenant_id: str,
        dataset_id: str,
        after_document_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        _DB, Document, _Knowledgebase, Generation, _DatasetIndexState, _TableIndex = self._models()
        query = (
            Generation.select(Generation)
            .join(
                Document,
                on=(
                    (Document.id == Generation.document_id)
                    & (Document.kb_id == Generation.kb_id)
                ),
            )
            .where(
                Generation.tenant_id == tenant_id,
                Generation.kb_id == dataset_id,
                Generation.status == "active",
            )
            .order_by(Generation.document_id)
            .limit(limit)
        )
        if after_document_id is not None:
            query = query.where(Generation.document_id > after_document_id)
        return [row.to_dict() for row in query]

    def commit_backfill_batch(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        expected_cursor: str | None,
        generations: list[dict[str, Any]],
        index_projections: list[list[dict[str, Any]]],
        next_cursor: str | None,
        complete: bool,
    ) -> None:
        DB, Document, _Knowledgebase, Generation, DatasetIndexState, TableIndex = self._models()
        with DB.atomic():
            state = (
                DatasetIndexState.select()
                .where(
                    DatasetIndexState.tenant_id == tenant_id,
                    DatasetIndexState.kb_id == dataset_id,
                )
                .for_update()
                .get()
            )
            if state.backfill_status != "pending" or state.backfill_cursor != expected_cursor:
                raise StructureSnapshotChanged(
                    "discovery index backfill cursor changed"
                )
            revision = int(state.index_revision) + 1
            for generation, projection in zip(
                generations,
                index_projections,
                strict=True,
            ):
                active = list(
                    Generation.select(Generation.producer_generation_ref)
                    .join(
                        Document,
                        on=(
                            (Document.id == Generation.document_id)
                            & (Document.kb_id == Generation.kb_id)
                        ),
                    )
                    .where(
                        Generation.tenant_id == tenant_id,
                        Generation.kb_id == dataset_id,
                        Generation.document_id == generation["document_id"],
                        Generation.status == "active",
                    )
                )
                if (
                    len(active) != 1
                    or active[0].producer_generation_ref
                    != generation["producer_generation_ref"]
                ):
                    raise StructureSnapshotChanged(
                        "active generation changed during discovery index backfill"
                    )
                TableIndex.update(active=False).where(
                    TableIndex.tenant_id == tenant_id,
                    TableIndex.kb_id == dataset_id,
                    TableIndex.document_id == generation["document_id"],
                    TableIndex.active == True,  # noqa: E712
                ).execute()
                for projected in projection:
                    values = {
                        **projected,
                        "index_revision": revision,
                        "active": True,
                    }
                    TableIndex.insert(**values).on_conflict(
                        preserve=[
                            TableIndex.table_ordinal,
                            TableIndex.search_text,
                            TableIndex.identity_hash,
                            TableIndex.index_revision,
                            TableIndex.active,
                            TableIndex.projection_status,
                            TableIndex.unsafe_reason,
                        ]
                    ).execute()
            updated = (
                DatasetIndexState.update(
                    index_revision=revision,
                    backfill_status="complete" if complete else "pending",
                    backfill_cursor=None if complete else next_cursor,
                )
                .where(
                    DatasetIndexState.tenant_id == tenant_id,
                    DatasetIndexState.kb_id == dataset_id,
                    DatasetIndexState.index_revision == state.index_revision,
                    DatasetIndexState.backfill_status == "pending",
                )
                .execute()
            )
            if updated != 1:
                raise StructureSnapshotChanged(
                    "discovery index backfill state changed"
                )

    @staticmethod
    def _replace_active_index_projection(
        *,
        DB,
        DatasetIndexState,
        TableIndex,
        tenant_id: str,
        dataset_id: str,
        document_id: str,
        producer_generation_ref: str,
        index_projection: list[dict[str, Any]],
    ) -> int:
        DatasetIndexState.insert(
            tenant_id=tenant_id,
            kb_id=dataset_id,
            index_revision=0,
            backfill_status="complete",
            backfill_cursor=None,
            index_schema_version=TABULAR_DISCOVERY_INDEX_SCHEMA_VERSION,
        ).on_conflict_ignore().execute()
        state = (
            DatasetIndexState.select()
            .where(
                DatasetIndexState.tenant_id == tenant_id,
                DatasetIndexState.kb_id == dataset_id,
            )
            .for_update()
            .get()
        )
        revision = int(state.index_revision) + 1
        DatasetIndexState.update(index_revision=revision).where(
            DatasetIndexState.tenant_id == tenant_id,
            DatasetIndexState.kb_id == dataset_id,
            DatasetIndexState.index_revision == state.index_revision,
        ).execute()
        TableIndex.update(active=False).where(
            TableIndex.tenant_id == tenant_id,
            TableIndex.kb_id == dataset_id,
            TableIndex.document_id == document_id,
            TableIndex.active == True,  # noqa: E712
        ).execute()
        for projected in index_projection:
            values = {
                **projected,
                "producer_generation_ref": producer_generation_ref,
                "index_revision": revision,
                "active": True,
            }
            TableIndex.insert(**values).on_conflict(
                preserve=[
                    TableIndex.table_ordinal,
                    TableIndex.search_text,
                    TableIndex.identity_hash,
                    TableIndex.index_revision,
                    TableIndex.active,
                    TableIndex.projection_status,
                    TableIndex.unsafe_reason,
                ]
            ).execute()
        return revision


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
    def backfill_active_generation_indexes(
        cls,
        storage,
        *,
        batch_size: int = 100,
        max_batches: int | None = None,
        repository=None,
    ) -> dict[str, int]:
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 1_000:
            raise ValueError("discovery backfill batch size is invalid")
        if max_batches is not None and (
            not isinstance(max_batches, int) or not 1 <= max_batches <= 10_000
        ):
            raise ValueError("discovery backfill max batches is invalid")
        repository = cls._repository(repository)
        result = {"batches": 0, "documents": 0, "datasets_completed": 0}
        while max_batches is None or result["batches"] < max_batches:
            pending = repository.list_pending_backfill_datasets(1)
            if not pending:
                break
            state = pending[0]
            tenant_id = state["tenant_id"]
            dataset_id = state["dataset_id"]
            cursor = state["cursor"]
            fetched = repository.list_active_generations_for_backfill(
                tenant_id,
                dataset_id,
                cursor,
                batch_size + 1,
            )
            generations = fetched[:batch_size]
            has_more = len(fetched) > batch_size
            index_projections = []
            for generation in generations:
                projection = load_tabular_structure_projection(
                    storage,
                    bucket=dataset_id,
                    document_id=generation["document_id"],
                    producer_generation_ref=generation[
                        "producer_generation_ref"
                    ],
                    manifest_object_name=generation["manifest_object_name"],
                    manifest_sha256=generation["manifest_sha256"],
                    expected_part_count=generation["part_count"],
                    tenant_id=tenant_id,
                )
                index_projections.append(
                    build_tabular_discovery_index_projection(
                        tenant_id=tenant_id,
                        dataset_id=dataset_id,
                        document_id=generation["document_id"],
                        projection=projection,
                    )
                )
            next_cursor = generations[-1]["document_id"] if has_more else None
            repository.commit_backfill_batch(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                expected_cursor=cursor,
                generations=generations,
                index_projections=index_projections,
                next_cursor=next_cursor,
                complete=not has_more,
            )
            result["batches"] += 1
            result["documents"] += len(generations)
            if not has_more:
                result["datasets_completed"] += 1
        return result

    @classmethod
    def discover_active_tables(
        cls,
        *,
        tenant_id: str,
        dataset_id: str,
        query: str,
        cursor: str | None,
        page_size: int,
        max_pages: int,
        max_evidence_bytes: int,
        max_evidence_tokens: int,
        deadline_ms: int,
        repository=None,
    ) -> dict[str, Any]:
        if not isinstance(page_size, int) or not 1 <= page_size <= 100:
            raise ValueError("discovery page size is invalid")
        if not isinstance(max_pages, int) or not 1 <= max_pages <= 100:
            raise ValueError("discovery max pages is invalid")
        if not isinstance(max_evidence_bytes, int) or max_evidence_bytes < 1:
            raise ValueError("discovery byte budget is invalid")
        if not isinstance(max_evidence_tokens, int) or max_evidence_tokens < 1:
            raise ValueError("discovery token budget is invalid")
        if not isinstance(deadline_ms, int) or deadline_ms < 1:
            raise ValueError("discovery deadline is invalid")
        started_at = time.monotonic()

        def deadline_exceeded() -> bool:
            return (time.monotonic() - started_at) * 1_000 >= deadline_ms

        repository = cls._repository(repository)
        normalized_query = normalize_tabular_discovery_query(query)
        query_digest = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()
        secret = repository.discovery_cursor_secret()
        initial = repository.discover_active_tables(
            tenant_id,
            dataset_id,
            normalized_query,
            None,
            0,
        )
        revision = initial["index_revision"]
        base = {
            "contract_version": TABULAR_DISCOVERY_CONTRACT_VERSION,
            "normalization_version": TABULAR_DISCOVERY_NORMALIZATION_VERSION,
            "query_digest": query_digest,
            "index_revision": revision,
            "seeds": [],
            "next_cursor": None,
            "has_more_in_window": False,
            "has_more_beyond_window": False,
            "incomplete": True,
            "incomplete_cause": None,
            "usage": {"candidates": 0, "evidence_bytes": 0, "evidence_tokens": 0},
        }
        if deadline_exceeded():
            return {**base, "incomplete_cause": "deadline"}
        after, page_ordinal = _decode_discovery_cursor(
            secret,
            cursor,
            tenant_id,
            dataset_id,
            revision,
            query_digest,
            page_size,
            max_pages,
        )
        indexed = repository.discover_active_tables(
            tenant_id,
            dataset_id,
            normalized_query,
            after,
            page_size + 1,
        )
        if deadline_exceeded():
            return {**base, "incomplete_cause": "deadline"}
        if indexed["index_revision"] != revision:
            return {
                **base,
                "index_revision": indexed["index_revision"],
                "incomplete_cause": "revision_drift",
            }
        if indexed["backfill_status"] != "complete":
            return {**base, "incomplete_cause": "backfill_pending"}
        if indexed["unsafe"]:
            return {**base, "incomplete_cause": "projection_unsafe"}
        selected = indexed["records"][:page_size]
        seeds = [
            {
                "document_id": record["document_id"],
                "producer_generation_ref": record["producer_generation_ref"],
                "table_ref": record["table_ref"],
                "table_ordinal": record["table_ordinal"],
                "retrieval_rule": TABULAR_DISCOVERY_RETRIEVAL_RULE,
                "score_encoded": score_encoded,
                "identity_hash": identity_hash,
            }
            for record in selected
            for score_encoded, identity_hash in [
                (record["score_encoded"], record["identity_hash"])
            ]
        ]
        evidence_bytes = len(json.dumps(seeds, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
        evidence_tokens = (evidence_bytes + 3) // 4
        if evidence_bytes > max_evidence_bytes or evidence_tokens > max_evidence_tokens:
            return {**base, "incomplete_cause": "budget"}
        has_more = len(indexed["records"]) > page_size
        has_more_in_window = has_more and page_ordinal < max_pages
        has_more_beyond_window = has_more and page_ordinal >= max_pages
        last = selected[-1] if selected else None
        return {
            **base,
            "seeds": seeds,
            "next_cursor": (
                _encode_discovery_cursor(
                    secret,
                    tenant_id,
                    dataset_id,
                    revision,
                    query_digest,
                    page_size,
                    max_pages,
                    page_ordinal + 1,
                    last["score_encoded"],
                    last["identity_hash"],
                )
                if has_more_in_window and last
                else None
            ),
            "has_more_in_window": has_more_in_window,
            "has_more_beyond_window": has_more_beyond_window,
            "incomplete": False,
            "usage": {
                "candidates": len(seeds),
                "evidence_bytes": evidence_bytes,
                "evidence_tokens": evidence_tokens,
            },
        }

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
        projection = load_tabular_structure_projection(
            storage,
            bucket=dataset_id,
            document_id=document_id,
            producer_generation_ref=producer_generation_ref,
            manifest_object_name=target["manifest_object_name"],
            manifest_sha256=target["manifest_sha256"],
            expected_part_count=target["part_count"],
            tenant_id=tenant_id,
        )
        index_projection = build_tabular_discovery_index_projection(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            projection=projection,
        )
        activated = repository.activate(
            tenant_id,
            dataset_id,
            document_id,
            producer_generation_ref,
            expected_active_generation_ref,
            index_projection,
        )
        return _managed_generation_result(activated, projection)

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
        projection = load_tabular_structure_projection(
            storage,
            bucket=dataset_id,
            document_id=document_id,
            producer_generation_ref=retained_generation_ref,
            manifest_object_name=target["manifest_object_name"],
            manifest_sha256=target["manifest_sha256"],
            expected_part_count=target["part_count"],
            tenant_id=tenant_id,
        )
        index_projection = build_tabular_discovery_index_projection(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            projection=projection,
        )
        restored = repository.restore(
            tenant_id,
            dataset_id,
            document_id,
            retained_generation_ref,
            expected_active_generation_ref,
            index_projection,
        )
        return _managed_generation_result(restored, projection)

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
            raise StructureSnapshotChanged(
                "active generation changed",
                active_generation_ref=record["producer_generation_ref"],
            )
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
        return _managed_generation_result(record, projection)

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
