"""Publish immutable tabular structure generations after document parsing."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any, Callable

from rag.app.tabular_structure import (
    ENUMERATION_RULE_VERSION,
    STRUCTURE_PRODUCER_ALGORITHM_VERSION,
    tabular_structure_projection_prefix,
)


TABULAR_STRUCTURE_GENERATION_TASK_TYPE = "tabular_generation"
TABULAR_GENERATION_CHECKPOINT_VERSION = "tabular-generation-checkpoint/v1"
TABULAR_GENERATION_MAX_ATTEMPTS = 5
TABULAR_GENERATION_NON_RETRYABLE_ERRORS = frozenset(
    {
        "source_changed",
        "legacy_generation_task_requires_source_snapshot",
        "invalid_generation_task",
        "task_cancelled",
    }
)
TABULAR_GENERATION_DELIVERY_UNKNOWN_MESSAGE = "Tabular structure generation delivery outcome unknown."


class TabularGenerationClaimUnavailable(RuntimeError):
    """The worker could not determine ownership of a generation message."""


class TabularGenerationRetryUnavailable(RuntimeError):
    """The worker could not durably transition a failed generation attempt."""


def _generation_progress(progress: float) -> float:
    """Keep completion reserved for the post-activation worker update."""
    return min(float(progress), 0.999999)


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
    parse_tasks = [
        task for task in tasks
        if isinstance(task, dict) and task.get("task_type", "") != TABULAR_STRUCTURE_GENERATION_TASK_TYPE
    ]
    if not parse_tasks:
        return False
    for task in parse_tasks:
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
    return _structure_generation_ref_from_source_sha256(
        document_id,
        hashlib.sha256(binary).hexdigest(),
        adr044_conversion_receipt=adr044_conversion_receipt,
    )


def _structure_generation_ref_from_source_sha256(
    document_id: str,
    source_sha256: str,
    *,
    adr044_conversion_receipt: dict[str, str] | None = None,
    force_nonce: str | None = None,
) -> str:
    """Derive identity from an already verified source snapshot."""
    from rag.app.tabular_structure import (
        ENUMERATION_RULE_VERSION,
        PRODUCER_SCHEMA_VERSION,
        PROJECTION_VERSION,
        STRUCTURE_PRODUCER_ALGORITHM_VERSION,
        _validate_adr044_conversion_receipt,
    )

    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id is required")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in source_sha256)
    ):
        raise ValueError("source SHA-256 is invalid")
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
    identity_suffix = f":force:{force_nonce}" if force_nonce else ""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"fuxi:tabular-generation:{PRODUCER_SCHEMA_VERSION}:{PROJECTION_VERSION}:"
            f"{STRUCTURE_PRODUCER_ALGORITHM_VERSION}:{ENUMERATION_RULE_VERSION}:"
            f"{document_id}:{source_identity}{identity_suffix}",
        )
    )


def structure_generation_ref_from_source_sha256(
    document_id: str,
    source_sha256: str,
    *,
    force_nonce: str | None = None,
) -> str:
    """Expose the source-bound identity for queue responses and tests."""
    return _structure_generation_ref_from_source_sha256(
        document_id, source_sha256, force_nonce=force_nonce
    )


def _force_task_digest(source_sha256: str, task_id: str) -> str:
    return f"force:{source_sha256}:{task_id}"


def _source_snapshot_from_task(task: dict[str, Any], actual_source_sha256: str) -> tuple[str, str | None]:
    expected = task.get("digest")
    if isinstance(expected, str) and expected.startswith("force:"):
        parts = expected.split(":")
        if len(parts) == 3 and len(parts[1]) == 64 and parts[2] == task.get("id"):
            return parts[1], parts[2]
        return "", None
    return expected or actual_source_sha256, None


def _generation_task_service():
    from api.db.services.task_service import TaskService

    return TaskService


def enqueue_tabular_structure_generation_if_complete(
    current_task: dict[str, Any],
    binary: bytes,
    *,
    tasks: list[dict[str, Any]] | None = None,
    task_list_provider: Callable[[str], list[dict[str, Any]] | None] | None = None,
    task_service=None,
    queue: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Queue one generation job after ordinary table parsing is complete."""
    if tasks is None and task_list_provider is not None:
        tasks = task_list_provider(current_task["doc_id"])
    if not is_complete_tabular_parse(current_task, tasks):
        return {"status": "pending"}
    return enqueue_tabular_structure_generation(
        tenant_id=current_task["tenant_id"],
        dataset_id=current_task["kb_id"],
        document_id=current_task["doc_id"],
        filename=current_task["name"],
        source_sha256=hashlib.sha256(binary).hexdigest(),
        task_service=task_service,
        queue=queue,
    )


def _storage_call(method, *args, tenant_id: str | None = None):
    try:
        accepts_tenant = "tenant_id" in __import__("inspect").signature(method).parameters
    except (TypeError, ValueError):
        accepts_tenant = False
    if tenant_id and accepts_tenant:
        return method(*args, tenant_id=tenant_id)
    return method(*args)


def _generation_checkpoint_name(document_id: str, generation_ref: str, sheet_ordinal: int) -> str:
    return (
        f"{tabular_structure_projection_prefix(document_id, generation_ref)}"
        f"checkpoint-sheet-{sheet_ordinal:06d}.json"
    )


def _put_generation_checkpoint(storage, bucket: str, object_name: str, payload: bytes, tenant_id: str) -> None:
    if _storage_call(storage.obj_exist, bucket, object_name, tenant_id=tenant_id):
        if _storage_call(storage.get, bucket, object_name, tenant_id=tenant_id) != payload:
            raise RuntimeError("immutable tabular generation checkpoint changed")
        return
    _storage_call(storage.put, bucket, object_name, payload, tenant_id=tenant_id)
    if _storage_call(storage.get, bucket, object_name, tenant_id=tenant_id) != payload:
        raise RuntimeError("tabular generation checkpoint verification failed")


def _read_generation_checkpoint(storage, bucket: str, object_name: str, tenant_id: str) -> dict[str, Any] | None:
    if not _storage_call(storage.obj_exist, bucket, object_name, tenant_id=tenant_id):
        return None
    raw = _storage_call(storage.get, bucket, object_name, tenant_id=tenant_id)
    try:
        value = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("tabular generation checkpoint is invalid") from error
    if not isinstance(value, dict):
        raise RuntimeError("tabular generation checkpoint is invalid")
    return value


def _default_sheet_count(filename: str, binary: bytes) -> int:
    return len(_load_workbook(binary).sheetnames)


def _load_workbook(binary: bytes):
    from io import BytesIO
    from rag.app.table import Excel

    return Excel()._load_excel_to_workbook(BytesIO(binary))


def _prepare_source_context(binary: bytes) -> dict[str, Any]:
    from rag.app.tabular_structure import (
        _formula_cached_result_kinds_by_sheet,
        _formula_coordinates_by_sheet,
        _formula_values_by_sheet,
    )

    return {
        "workbook": _load_workbook(binary),
        "formula_coordinates": _formula_coordinates_by_sheet(binary),
        "formula_cached_result_kinds": _formula_cached_result_kinds_by_sheet(binary),
        "formula_values": _formula_values_by_sheet(binary),
    }


def _merge_sheet_projections(
    projections: list[dict[str, Any]],
    *,
    generation_ref: str,
) -> dict[str, Any]:
    if not projections:
        raise RuntimeError("tabular generation has no completed Sheet checkpoints")
    first = projections[0]
    required = {
        "version",
        "producer_schema_version",
        "producer_generation_ref",
        "structure_algorithm_version",
        "enumeration_rule_version",
        "source_sha256",
    }
    if not required.issubset(first):
        raise RuntimeError("tabular generation checkpoint projection is incomplete")
    for projection in projections:
        if any(projection.get(key) != first.get(key) for key in required) or projection.get("producer_generation_ref") != generation_ref:
            raise RuntimeError("tabular generation checkpoints have mixed identity")
    return {
        **first,
        "tables": [table for projection in projections for table in projection.get("tables", [])],
        "rows": [row for projection in projections for row in projection.get("rows", [])],
    }


def enqueue_tabular_structure_generation(
    *,
    tenant_id: str,
    dataset_id: str,
    document_id: str,
    filename: str,
    source_sha256: str,
    force_generation: bool = False,
    task_service=None,
    queue: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Queue a document-scoped structure job without reading the workbook.

    The ordinary chunk pipeline owns source availability. This entrypoint only
    creates a durable worker task; the worker computes the source-bound
    generation identity after reading the immutable object.
    """
    if not all(isinstance(value, str) and value for value in (tenant_id, dataset_id, document_id, filename)):
        raise ValueError("generation task scope is required")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in source_sha256)
    ):
        raise ValueError("source SHA-256 is invalid")
    task_service = task_service or _generation_task_service()
    if not isinstance(force_generation, bool):
        raise ValueError("force_generation must be a boolean")
    task_id = uuid.uuid4().hex
    task = {
        "id": task_id,
        "doc_id": document_id,
        "tenant_id": tenant_id,
        "kb_id": dataset_id,
        "name": filename,
        "parser_id": "table",
        "task_type": TABULAR_STRUCTURE_GENERATION_TASK_TYPE,
        "progress": 0.0,
        "from_page": 0,
        "to_page": 0,
        "digest": _force_task_digest(source_sha256, task_id) if force_generation else source_sha256,
        "progress_msg": "Tabular structure generation queued.",
    }
    find_kwargs = {"source_sha256": source_sha256}
    if force_generation:
        find_kwargs["force_generation"] = True
    outcome = task_service.find_or_insert_generation_task(task, **find_kwargs)
    existing = outcome.get("task") if isinstance(outcome, dict) else None
    if existing is not None and not outcome.get("created", False):
        task = {**task, **existing}
        if task.get("progress") != 0:
            return {
                "status": "queued",
                "task_id": task["id"],
                "task_type": TABULAR_STRUCTURE_GENERATION_TASK_TYPE,
                "source_sha256": task.get("digest", source_sha256),
            }
    if isinstance(existing, dict):
        task = {**task, **existing}

    if queue is None:
        from common import settings
        from rag.utils.redis_conn import REDIS_CONN

        queue = lambda message: REDIS_CONN.queue_product_outcome(  # noqa: E731
            settings.get_svr_queue_name(0, "tabular"),
            message=message,
        )
    try:
        queued = queue(task)
    except Exception as error:
        _mark_generation_delivery_unknown(task_service, task["id"], str(error))
        logging.exception("Tabular structure generation queue outcome unknown task=%s", task["id"])
        raise TabularGenerationRetryUnavailable(task["id"]) from error
    if queued is not True and queued != "queued":
        _mark_generation_delivery_unknown(task_service, task["id"], "queue returned no delivery confirmation")
        raise TabularGenerationRetryUnavailable(task["id"])
    _mark_generation_delivery_queued(task_service, task["id"])
    return {
        "status": "queued",
        "task_id": task["id"],
        "task_type": TABULAR_STRUCTURE_GENERATION_TASK_TYPE,
        "source_sha256": source_sha256,
        **(
            {
                "producer_generation_ref": structure_generation_ref_from_source_sha256(
                    document_id, source_sha256, force_nonce=task["id"]
                )
            }
            if force_generation
            else {}
        ),
    }


def retry_tabular_structure_generation_task(
    task_id: str,
    *,
    task_service=None,
    queue: Callable[[dict[str, Any]], Any] | None = None,
    max_attempts: int = TABULAR_GENERATION_MAX_ATTEMPTS,
    failure_code: str | None = None,
) -> bool:
    """Requeue a recoverable generation failure within the total-attempt cap."""
    task_service = task_service or _generation_task_service()
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or not 1 <= max_attempts <= TABULAR_GENERATION_MAX_ATTEMPTS:
        raise ValueError("max_attempts must be a positive integer at most 5")
    retryable = failure_code not in TABULAR_GENERATION_NON_RETRYABLE_ERRORS
    try:
        outcome = task_service.fail_generation_attempt_and_reserve_retry(
            task_id,
            message=failure_code or "tabular_generation_failed",
            max_attempts=max_attempts,
            retryable=retryable,
        )
    except Exception as error:
        logging.exception("Unable to claim tabular generation retry task=%s", task_id)
        raise TabularGenerationRetryUnavailable(task_id) from error
    if not isinstance(outcome, dict):
        raise TabularGenerationRetryUnavailable(task_id)
    if outcome.get("status") != "retry":
        return False
    if queue is None:
        from common import settings
        from rag.utils.redis_conn import REDIS_CONN

        queue = lambda message: REDIS_CONN.queue_product_outcome(  # noqa: E731
            settings.get_svr_queue_name(0, "tabular"),
            message=message,
        )
    try:
        queued = queue({"id": task_id, "task_type": TABULAR_STRUCTURE_GENERATION_TASK_TYPE})
    except Exception as error:
        raise TabularGenerationRetryUnavailable(task_id) from error
    if queued is False or queued == "unknown":
        _mark_generation_delivery_unknown(task_service, task_id, "retry queue returned no delivery confirmation")
        raise TabularGenerationRetryUnavailable(task_id)
    if queued is not True and queued != "queued":
        raise TabularGenerationRetryUnavailable(task_id)
    _mark_generation_delivery_queued(task_service, task_id)
    return True


def _mark_generation_delivery_unknown(task_service, task_id: str, detail: str = "") -> None:
    message = TABULAR_GENERATION_DELIVERY_UNKNOWN_MESSAGE
    if detail:
        message = f"{message} {detail}"
    try:
        updated = task_service.mark_generation_delivery_unknown(task_id, message=message)
    except Exception as error:
        raise TabularGenerationRetryUnavailable(task_id) from error
    if not updated:
        raise TabularGenerationRetryUnavailable(task_id)


def _mark_generation_delivery_queued(task_service, task_id: str) -> None:
    marker = getattr(task_service, "mark_generation_delivery_queued", None)
    if marker is None:
        return
    try:
        if not marker(task_id):
            raise TabularGenerationRetryUnavailable(task_id)
    except TabularGenerationRetryUnavailable:
        raise
    except Exception as error:
        raise TabularGenerationRetryUnavailable(task_id) from error


def reconcile_tabular_structure_generation_tasks(
    *, task_service=None, queue: Callable[[dict[str, Any]], Any] | None = None, limit: int = 20
) -> int:
    """Re-deliver uncertain generation tasks from the worker's durable recovery tick."""
    task_service = task_service or _generation_task_service()
    tasks = task_service.list_generation_delivery_unknown_tasks(limit=limit)
    if queue is None:
        from common import settings
        from rag.utils.redis_conn import REDIS_CONN

        queue = lambda message: REDIS_CONN.queue_product_outcome(  # noqa: E731
            settings.get_svr_queue_name(0, "tabular"),
            message=message,
        )
    recovered = 0
    for task in tasks:
        task_id = task.get("id") if isinstance(task, dict) else None
        if not isinstance(task_id, str) or not task_service.claim_generation_delivery_reconciliation(task_id):
            continue
        try:
            queued = queue({"id": task_id, "task_type": TABULAR_STRUCTURE_GENERATION_TASK_TYPE})
        except Exception as error:
            task_service.mark_generation_delivery_unknown(task_id, message=str(error))
            continue
        if queued is True or queued == "queued":
            task_service.mark_generation_delivery_queued(task_id)
            recovered += 1
        else:
            task_service.mark_generation_delivery_unknown(
                task_id,
                message="reconciliation delivery remains unknown",
            )
    return recovered


def claim_tabular_structure_generation_attempt(task_id: str, *, task_service=None) -> bool:
    """Claim a queued generation message, preserving it on service failure."""
    task_service = task_service or _generation_task_service()
    try:
        claim = task_service.claim_generation_attempt(task_id)
    except Exception as error:
        raise TabularGenerationClaimUnavailable(task_id) from error
    return isinstance(claim, dict) and claim.get("status") == "claimed"


def run_tabular_structure_generation_job(
    task: dict[str, Any],
    binary: bytes,
    *,
    storage=None,
    service=None,
    projection_builder: Callable[..., Any] | None = None,
    projection_store: Callable[..., dict[str, Any]] | None = None,
    sheet_count_provider: Callable[..., int] | None = None,
    workbook_provider: Callable[[bytes], Any] | None = None,
    source_context_provider: Callable[[bytes], dict[str, Any]] | None = None,
    progress_callback: Callable[[float, str], Any] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Resume a source-bound Sheet job and publish a shadow for CAS activation."""
    if not isinstance(binary, bytes) or not binary:
        return {"status": "failed", "safe_error_code": "source_unavailable"}
    required = ("tenant_id", "kb_id", "doc_id", "name")
    if any(not isinstance(task.get(key), str) or not task[key] for key in required):
        return {"status": "failed", "safe_error_code": "invalid_generation_task"}
    tenant_id = task["tenant_id"]
    dataset_id = task["kb_id"]
    document_id = task["doc_id"]
    source_sha256 = hashlib.sha256(binary).hexdigest()
    expected_source_sha256, force_nonce = _source_snapshot_from_task(task, source_sha256)
    if expected_source_sha256 in {"", "source-pending"}:
        return {
            "status": "failed",
            "safe_error_code": "legacy_generation_task_requires_source_snapshot",
        }
    if expected_source_sha256 and expected_source_sha256 != source_sha256:
        return {
            "status": "failed",
            "safe_error_code": "source_changed",
        }
    generation_ref = structure_generation_ref_from_source_sha256(
        document_id, source_sha256, force_nonce=force_nonce
    )

    if service is None:
        from api.db.services.tabular_structure_service import TabularStructureService

        service = TabularStructureService
    if storage is None:
        from common import settings

        storage = settings.STORAGE_IMPL
    if projection_builder is None:
        from rag.app.table import build_structure_projection

        projection_builder = build_structure_projection
    if projection_store is None:
        from rag.app.tabular_structure import store_tabular_structure_projection

        projection_store = store_tabular_structure_projection
    if sheet_count_provider is None:
        sheet_count_provider = _default_sheet_count

    def cancelled() -> bool:
        return bool(cancel_check is not None and cancel_check())

    def cancelled_result() -> dict[str, Any]:
        return {
            "status": "cancelled",
            "producer_generation_ref": generation_ref,
            "safe_error_code": "task_cancelled",
        }

    try:
        if cancelled():
            return cancelled_result()
        source_context = (
            source_context_provider(binary)
            if source_context_provider is not None
            else _prepare_source_context(binary)
        )
        workbook = source_context.get("workbook")
        if workbook is None and workbook_provider is not None:
            workbook = workbook_provider(binary)
            source_context["workbook"] = workbook
        if workbook is None:
            workbook = _load_workbook(binary)
            source_context["workbook"] = workbook

        active = _active_generation_ref(
            service,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
        )
        if force_nonce is None and active == generation_ref:
            return {"status": "active", "producer_generation_ref": generation_ref}
        actual_sheet_count = len(getattr(workbook, "sheetnames", ()))
        sheet_count = int(sheet_count_provider(task["name"], binary))
        if sheet_count < 1 or sheet_count != actual_sheet_count:
            raise RuntimeError("tabular workbook has no worksheets")
        projections = []
        for sheet_ordinal in range(1, sheet_count + 1):
            if cancelled():
                return cancelled_result()
            checkpoint_name = _generation_checkpoint_name(document_id, generation_ref, sheet_ordinal)
            checkpoint = _read_generation_checkpoint(storage, dataset_id, checkpoint_name, tenant_id)
            if checkpoint is None:
                if progress_callback is not None:
                    progress_callback(
                        _generation_progress((sheet_ordinal - 1) / sheet_count),
                        f"Building worksheet {sheet_ordinal}/{sheet_count}.",
                    )
                built = projection_builder(
                    task["name"],
                    binary,
                    producer_generation_ref=generation_ref,
                    sheet_ordinals={sheet_ordinal},
                    workbook=workbook,
                    source_context=source_context,
                )
                if isinstance(built, tuple) and len(built) == 2:
                    projection, audit = built
                else:
                    projection, audit = built, {}
                if cancelled():
                    return cancelled_result()
                checkpoint = {
                    "version": TABULAR_GENERATION_CHECKPOINT_VERSION,
                    "producer_generation_ref": generation_ref,
                    "source_sha256": source_sha256,
                    "sheet_ordinal": sheet_ordinal,
                    "projection": projection,
                    "audit": audit,
                }
                payload = json.dumps(checkpoint, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                _put_generation_checkpoint(storage, dataset_id, checkpoint_name, payload, tenant_id)
                if cancelled():
                    return cancelled_result()
            if (
                checkpoint.get("version") != TABULAR_GENERATION_CHECKPOINT_VERSION
                or checkpoint.get("producer_generation_ref") != generation_ref
                or checkpoint.get("source_sha256") != source_sha256
                or checkpoint.get("sheet_ordinal") != sheet_ordinal
                or not isinstance(checkpoint.get("projection"), dict)
            ):
                raise RuntimeError("tabular generation checkpoint identity is invalid")
            sheet_projection = checkpoint["projection"]
            if not isinstance(sheet_projection, dict):
                raise RuntimeError("tabular generation checkpoint projection is invalid")
            projections.append(sheet_projection)
            if progress_callback is not None:
                progress_callback(
                    _generation_progress(sheet_ordinal / sheet_count),
                    f"Worksheet {sheet_ordinal}/{sheet_count} complete.",
                )

        if cancelled():
            return cancelled_result()
        projection = _merge_sheet_projections(projections, generation_ref=generation_ref)
        if cancelled():
            return cancelled_result()
        service.persist_shadow_generation(
            storage,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            projection=projection,
            projection_store=projection_store,
        )
        if cancelled():
            return cancelled_result()
        return {
            "status": "shadow",
            "producer_generation_ref": generation_ref,
            "row_count": len(projection.get("rows", [])),
        }
    except Exception as error:
        logging.warning(
            "Tabular structure generation job unavailable generation=%s reason=%s",
            generation_ref,
            error.__class__.__name__,
        )
        return {
            "status": "failed",
            "producer_generation_ref": generation_ref,
            "safe_error_code": error.__class__.__name__.lower(),
        }

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


def build_tabular_structure_shadow_from_source(
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
    """Build and register an immutable generation without changing active state."""

    generation_ref = structure_generation_ref(
        document_id,
        binary,
        adr044_conversion_receipt=adr044_conversion_receipt,
    )
    if service is None:
        from api.db.services.tabular_structure_service import TabularStructureService

        service = TabularStructureService
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
    service.persist_shadow_generation(
        storage,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        projection=projection,
        projection_store=projection_store,
    )
    return {
        "status": "shadow",
        "producer_generation_ref": generation_ref,
        "row_count": len(projection["rows"]),
    }


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
        shadow = build_tabular_structure_shadow_from_source(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            document_id=document_id,
            filename=filename,
            binary=binary,
            adr044_conversion_receipt=adr044_conversion_receipt,
            storage=storage,
            service=service,
            projection_builder=projection_builder,
            projection_store=projection_store,
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
            "row_count": shadow["row_count"],
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
    "build_tabular_structure_shadow_from_source",
    "enqueue_tabular_structure_generation",
    "enqueue_tabular_structure_generation_if_complete",
    "is_complete_tabular_parse",
    "run_tabular_structure_generation_job",
    "publish_tabular_structure_from_source",
    "publish_tabular_structure_generation",
    "reconcile_tabular_structure_generation_tasks",
    "structure_generation_ref",
    "structure_generation_ref_from_source_sha256",
]
