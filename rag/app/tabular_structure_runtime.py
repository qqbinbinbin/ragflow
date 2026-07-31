"""Publish immutable tabular structure generations after document parsing."""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any, Callable

from rag.app.tabular_structure import (
    ENUMERATION_RULE_VERSION,
    STRUCTURE_PRODUCER_ALGORITHM_VERSION,
)


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


def structure_generation_ref(
    document_id: str,
    binary: bytes,
    *,
    adr044_conversion_receipt: dict[str, str] | None = None,
) -> str:
    """Derive identity from source bytes and an explicitly supplied conversion receipt.

    The background worker and manual rebuild endpoint currently provide no
    governed ADR-044 receipt. In that case this function binds only the bytes it
    received and must not infer original-source or converter identity.
    """

    from rag.app.tabular_structure import (
        ENUMERATION_RULE_VERSION,
        PRODUCER_SCHEMA_VERSION,
        PROJECTION_VERSION,
        STRUCTURE_PRODUCER_ALGORITHM_VERSION,
        _validate_adr044_conversion_receipt,
    )

    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id is required")
    if not isinstance(binary, bytes) or not binary:
        raise ValueError("complete source bytes are required")
    source_sha256 = hashlib.sha256(binary).hexdigest()
    has_adr044_receipt = _validate_adr044_conversion_receipt(
        adr044_conversion_receipt,
        source_sha256,
    )
    source_identity = (
        f"{adr044_conversion_receipt['original_source_sha256']}:"
        f"{source_sha256}:{adr044_conversion_receipt['converter_version']}"
        if has_adr044_receipt
        else source_sha256
    )
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"fuxi:tabular-generation:{PRODUCER_SCHEMA_VERSION}:{PROJECTION_VERSION}:"
            f"{STRUCTURE_PRODUCER_ALGORITHM_VERSION}:{ENUMERATION_RULE_VERSION}:"
            f"{document_id}:{source_identity}",
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
    adr044_conversion_receipt: dict[str, str] | None = None,
    storage=None,
    service=None,
    projection_builder: Callable[..., dict[str, Any]] | None = None,
    projection_store: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and publish from source bytes plus an optional caller-owned receipt."""

    generation_ref = None
    try:
        generation_ref = (
            structure_generation_ref(
                document_id,
                binary,
                adr044_conversion_receipt=adr044_conversion_receipt,
            )
            if adr044_conversion_receipt is not None
            else structure_generation_ref(document_id, binary)
        )
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

        builder_kwargs = {"producer_generation_ref": generation_ref}
        if adr044_conversion_receipt is not None:
            builder_kwargs["adr044_conversion_receipt"] = adr044_conversion_receipt
        projection = projection_builder(filename, binary, **builder_kwargs)
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
        if generation_ref is not None:
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
            generation_ref or "unavailable",
            error.__class__.__name__,
        )
        result = {"status": "failed"}
        if generation_ref is not None:
            result["producer_generation_ref"] = generation_ref
        return result


def publish_tabular_structure_generation(
    current_task: dict[str, Any],
    binary: bytes,
    *,
    tasks: list[dict[str, Any]] | None = None,
    adr044_conversion_receipt: dict[str, str] | None = None,
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

    publish_kwargs = {
        "tenant_id": current_task["tenant_id"],
        "dataset_id": current_task["kb_id"],
        "document_id": current_task["doc_id"],
        "filename": current_task["name"],
        "binary": binary,
        "storage": storage,
        "service": service,
        "projection_builder": projection_builder,
        "projection_store": projection_store,
    }
    if adr044_conversion_receipt is not None:
        publish_kwargs["adr044_conversion_receipt"] = adr044_conversion_receipt
    return publish_tabular_structure_from_source(
        **publish_kwargs,
    )


__all__ = [
    "is_complete_tabular_parse",
    "publish_tabular_structure_from_source",
    "publish_tabular_structure_generation",
    "structure_generation_ref",
]
