"""Publish immutable tabular structure generations after document parsing."""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any, Callable


STRUCTURE_PRODUCER_ALGORITHM_VERSION = "region-producer/v2"


def is_complete_tabular_parse(current_task: dict[str, Any], tasks: list[dict[str, Any]] | None) -> bool:
    """Return true only when every row-range task for the document is done.

    The current task is considered complete because this hook runs after its
    ordinary chunk write and terminal progress update. Other tasks must already
    be terminal, which keeps concurrent row-range workers fail-closed.
    """

    if not isinstance(current_task, dict) or current_task.get("parser_id", "").lower() != "table":
        return False
    if current_task.get("progress", 0) < 1.0:
        return False
    if not isinstance(tasks, list) or not tasks:
        return False
    current_id = current_task.get("id")
    document_id = current_task.get("doc_id")
    for task in tasks:
        if not isinstance(task, dict) or task.get("doc_id", document_id) != document_id:
            return False
        if task.get("parser_id", "table").lower() != "table":
            return False
        if task.get("id") == current_id:
            continue
        if task.get("progress", 0) < 1.0:
            return False
    return True


def structure_generation_ref(document_id: str, binary: bytes) -> str:
    """Derive an idempotent generation identity from document and source bytes."""

    from rag.app.tabular_structure import PRODUCER_SCHEMA_VERSION, PROJECTION_VERSION

    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id is required")
    if not isinstance(binary, bytes) or not binary:
        raise ValueError("complete source bytes are required")
    source_sha256 = hashlib.sha256(binary).hexdigest()
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"fuxi:tabular-generation:{PRODUCER_SCHEMA_VERSION}:{PROJECTION_VERSION}:"
            f"{STRUCTURE_PRODUCER_ALGORITHM_VERSION}:{document_id}:{source_sha256}",
        )
    )


def _active_generation_ref(service, *, tenant_id: str, dataset_id: str, document_id: str) -> str | None:
    try:
        return service.get_active_generation(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
        )["producer_generation_ref"]
    except Exception as error:
        missing_type = getattr(service, "StructureSnapshotMissing", None)
        if missing_type is not None and isinstance(error, missing_type):
            return None
        if error.__class__.__name__ == "StructureSnapshotMissing":
            return None
        raise


def publish_tabular_structure_from_source(
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    filename: str,
    binary: bytes,
    storage=None,
    service=None,
    projection_builder: Callable[..., dict[str, Any]] | None = None,
    projection_store: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and atomically publish a structure projection from source bytes."""

    generation_ref = structure_generation_ref(document_id, binary)
    try:
        if service is None:
            from api.db.services.tabular_structure_service import TabularStructureService

            service = TabularStructureService
        existing = None
        try:
            existing = service.get_active_generation(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
            )
        except Exception as error:
            missing_type = getattr(service, "StructureSnapshotMissing", None)
            if not (
                (missing_type is not None and isinstance(error, missing_type))
                or error.__class__.__name__ == "StructureSnapshotMissing"
            ):
                raise
        if existing and existing["producer_generation_ref"] == generation_ref:
            result = {"status": "active", "producer_generation_ref": generation_ref}
            if isinstance(existing.get("row_count"), int):
                result["row_count"] = existing["row_count"]
            return result
        if storage is None:
            from common import settings

            storage = settings.STORAGE_IMPL
        if projection_builder is None:
            from rag.app.table import build_structure_projection

            projection_builder = build_structure_projection
        if projection_store is None:
            from rag.app.tabular_structure import store_tabular_structure_projection

            projection_store = store_tabular_structure_projection

        projection = projection_builder(
            filename,
            binary,
            producer_generation_ref=generation_ref,
        )
        receipt = projection_store(
            storage,
            bucket=dataset_id,
            document_id=document_id,
            projection=projection,
            tenant_id=tenant_id,
        )
        service.register_shadow_generation(
            storage,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            receipt=receipt,
        )
        expected_active = _active_generation_ref(
            service,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
        )
        try:
            service.activate_generation(
                storage,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
                producer_generation_ref=generation_ref,
                expected_active_generation_ref=expected_active,
            )
        except Exception:
            if _active_generation_ref(
                service,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
            ) != generation_ref:
                raise
        return {
            "status": "active",
            "producer_generation_ref": generation_ref,
            "row_count": len(projection["rows"]),
        }
    except Exception as error:
        try:
            if _active_generation_ref(
                service,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                document_id=document_id,
            ) == generation_ref:
                return {"status": "active", "producer_generation_ref": generation_ref}
        except Exception:
            pass
        logging.warning(
            "Tabular structure generation unavailable generation=%s reason=%s",
            generation_ref,
            error.__class__.__name__,
        )
        return {"status": "failed", "producer_generation_ref": generation_ref}


def publish_tabular_structure_generation(
    current_task: dict[str, Any],
    binary: bytes,
    *,
    tasks: list[dict[str, Any]] | None = None,
    storage=None,
    service=None,
    task_list_provider: Callable[[str], list[dict[str, Any]] | None] | None = None,
    projection_builder: Callable[..., dict[str, Any]] | None = None,
    projection_store: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build, persist, validate and activate one document-scoped generation.

    This function intentionally returns a safe status instead of raising. The
    ordinary relevance path has already succeeded and must remain available if
    the derived completeness read model cannot be published.
    """

    if tasks is None and task_list_provider is not None:
        tasks = task_list_provider(current_task["doc_id"])
    if not is_complete_tabular_parse(current_task, tasks):
        return {"status": "pending"}

    return publish_tabular_structure_from_source(
        tenant_id=current_task["tenant_id"],
        dataset_id=current_task["kb_id"],
        document_id=current_task["doc_id"],
        filename=current_task["name"],
        binary=binary,
        storage=storage,
        service=service,
        projection_builder=projection_builder,
        projection_store=projection_store,
    )


__all__ = [
    "is_complete_tabular_parse",
    "publish_tabular_structure_from_source",
    "publish_tabular_structure_generation",
    "structure_generation_ref",
]
