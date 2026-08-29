import hashlib
import ast
import uuid
import json
import sys
from pathlib import Path

import pytest

import rag.app.tabular_structure as tabular_structure
import rag.app.tabular_structure_runtime as runtime
from rag.app.tabular_structure_runtime import (
    TABULAR_STRUCTURE_GENERATION_TASK_TYPE,
    TABULAR_GENERATION_MAX_ATTEMPTS,
    TabularGenerationClaimUnavailable,
    TabularGenerationRetryUnavailable,
    claim_tabular_structure_generation_attempt,
    enqueue_tabular_structure_generation,
    run_tabular_structure_generation_job,
    is_complete_tabular_parse,
    publish_tabular_structure_from_source,
    publish_tabular_structure_generation,
    retry_tabular_structure_generation_task,
    reconcile_tabular_structure_generation_tasks,
    structure_generation_ref,
    structure_generation_ref_from_source_sha256,
)


def test_generation_task_insert_returns_task_fields_after_peewee_insert_count():
    from api.db.services.task_service import TaskService

    implementation = TaskService.insert_generation_task.__func__
    while hasattr(implementation, "__wrapped__"):
        implementation = implementation.__wrapped__

    inserted = {}

    class Service(TaskService):
        insert = staticmethod(lambda **fields: inserted.update(fields) or 1)

    task = {
        "id": "generation-task-1",
        "doc_id": "document-1",
        "from_page": 0,
        "to_page": 0,
        "task_type": "tabular_generation",
        "priority": 0,
        "begin_at": None,
        "progress": 0.0,
        "progress_msg": "queued",
        "digest": "a" * 64,
    }

    result = implementation(Service, task)

    assert result == inserted
    assert inserted["id"] == "generation-task-1"
    assert inserted["task_type"] == "tabular_generation"


def test_generation_enqueue_returns_without_reading_or_building_source(monkeypatch):
    calls = []

    class Service:
        @staticmethod
        def find_or_insert_generation_task(task, *, source_sha256):
            calls.append(("find_or_insert", task, source_sha256))
            return {"created": True, "task": task}

    monkeypatch.setattr(
        "rag.app.tabular_structure_runtime._generation_task_service",
        lambda: Service,
    )

    result = enqueue_tabular_structure_generation(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        document_id="document-1",
        filename="workbook.xlsx",
        source_sha256="a" * 64,
        task_service=Service,
        queue=lambda task: calls.append(("queue", task)) or True,
    )

    assert result["status"] == "queued"
    assert result["task_type"] == TABULAR_STRUCTURE_GENERATION_TASK_TYPE
    assert [entry[0] for entry in calls] == ["find_or_insert", "queue"]


def test_generation_enqueue_uses_outcome_aware_default_queue(monkeypatch):
    calls = []

    class Service:
        @staticmethod
        def find_or_insert_generation_task(task, *, source_sha256):
            return {"created": True, "task": task}

        @staticmethod
        def mark_generation_delivery_queued(task_id):
            calls.append(("queued", task_id))
            return True

    class Redis:
        @staticmethod
        def queue_product_outcome(queue, *, message):
            calls.append(("queue", queue, message))
            return "queued"

        def queue_product(self, *_args, **_kwargs):
            raise AssertionError("tabular generation must use the outcome-aware queue")

    monkeypatch.setitem(sys.modules, "rag.utils.redis_conn", type("RedisModule", (), {"REDIS_CONN": Redis()})())
    monkeypatch.setattr("common.settings.get_svr_queue_name", lambda *_args, **_kwargs: "common")

    result = enqueue_tabular_structure_generation(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        document_id="document-1",
        filename="workbook.xlsx",
        source_sha256="a" * 64,
        task_service=Service,
    )

    assert result["status"] == "queued"
    assert [entry[0] for entry in calls] == ["queue", "queued"]


def test_generation_enqueue_returns_source_bound_generation_reference(monkeypatch):
    source = b"workbook"
    source_sha256 = hashlib.sha256(source).hexdigest()

    class Service:
        @staticmethod
        def find_or_insert_generation_task(task, *, source_sha256):
            return {"created": True, "task": {**task, "digest": source_sha256}}

    result = enqueue_tabular_structure_generation(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        document_id="document-1",
        filename="workbook.xlsx",
        source_sha256=source_sha256,
        task_service=Service,
        queue=lambda _task: True,
    )

    assert result["status"] == "queued"
    assert result["source_sha256"] == source_sha256
    assert "producer_generation_ref" not in result


def test_force_generation_enqueue_creates_a_distinct_task_identity():
    source_sha256 = hashlib.sha256(b"workbook").hexdigest()
    queued = []

    class Service:
        @staticmethod
        def find_or_insert_generation_task(task, *, source_sha256, force_generation=False):
            queued.append(task)
            return {"created": True, "task": task}

    result = enqueue_tabular_structure_generation(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        document_id="document-1",
        filename="workbook.xlsx",
        source_sha256=source_sha256,
        force_generation=True,
        task_service=Service,
        queue=lambda task: True,
    )

    assert result["status"] == "queued"
    assert queued[0]["digest"].startswith("force:")
    assert result["producer_generation_ref"] != structure_generation_ref(
        "document-1", b"workbook"
    )


def test_force_generation_ref_is_stable_for_the_same_task():
    source = b"workbook"
    source_sha256 = hashlib.sha256(source).hexdigest()
    first = runtime.structure_generation_ref_from_source_sha256(
        "document-1", source_sha256, force_nonce="task-1"
    )
    second = runtime.structure_generation_ref_from_source_sha256(
        "document-1", source_sha256, force_nonce="task-1"
    )

    assert first == second
    assert first != structure_generation_ref("document-1", source)


def test_generation_enqueue_requires_a_source_snapshot():
    with pytest.raises(ValueError, match="source SHA-256 is invalid"):
        enqueue_tabular_structure_generation(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            filename="workbook.xlsx",
            source_sha256=None,
            task_service=object(),
            queue=lambda _task: True,
        )


def test_generation_enqueue_reuses_the_existing_atomic_task_without_requeueing():
    calls = []

    class Service:
        @staticmethod
        def find_or_insert_generation_task(task, *, source_sha256):
            calls.append(("find_or_insert", task, source_sha256))
            return {"created": False, "task": {"id": "existing-task"}}

    result = enqueue_tabular_structure_generation(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        document_id="document-1",
        filename="workbook.xlsx",
        source_sha256="a" * 64,
        task_service=Service,
        queue=lambda task: calls.append(("queue", task)) or True,
    )

    assert result["task_id"] == "existing-task"
    assert [entry[0] for entry in calls] == ["find_or_insert", "queue"]


def test_generation_enqueue_keeps_unknown_delivery_recoverable():
    calls = []

    class Service:
        @staticmethod
        def find_or_insert_generation_task(task, *, source_sha256):
            calls.append(("find_or_insert", task, source_sha256))
            return {"created": True, "task": task}

        @staticmethod
        def mark_generation_delivery_unknown(task_id, *, message):
            calls.append(("unknown", task_id, message))

    def queue(_task):
        calls.append(("queue",))
        raise TimeoutError("delivery outcome unknown")

    with pytest.raises(TabularGenerationRetryUnavailable):
        enqueue_tabular_structure_generation(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            filename="workbook.xlsx",
            source_sha256="a" * 64,
            task_service=Service,
            queue=queue,
        )
    assert [entry[0] for entry in calls] == ["find_or_insert", "queue", "unknown"]


def test_generation_enqueue_keeps_a_false_delivery_outcome_recoverable():
    calls = []

    class Service:
        @staticmethod
        def find_or_insert_generation_task(task, *, source_sha256):
            return {"created": True, "task": {**task, "digest": source_sha256}}

        @staticmethod
        def mark_generation_delivery_unknown(task_id, *, message):
            calls.append((task_id, message))

    with pytest.raises(TabularGenerationRetryUnavailable):
        enqueue_tabular_structure_generation(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            document_id="document-1",
            filename="workbook.xlsx",
            source_sha256="a" * 64,
            task_service=Service,
            queue=lambda _task: False,
        )

    assert len(calls) == 1
    assert calls[0][1] == "Tabular structure generation delivery outcome unknown. queue returned no delivery confirmation"


def test_generation_delivery_reconciliation_requeues_unknown_tasks():
    calls = []

    class Service:
        @staticmethod
        def list_generation_delivery_unknown_tasks(*, limit):
            calls.append(("list", limit))
            return [{"id": "generation-task", "task_type": TABULAR_STRUCTURE_GENERATION_TASK_TYPE}]

        @staticmethod
        def claim_generation_delivery_reconciliation(task_id):
            calls.append(("claim", task_id))
            return True

        @staticmethod
        def mark_generation_delivery_queued(task_id):
            calls.append(("queued", task_id))

    assert reconcile_tabular_structure_generation_tasks(
        task_service=Service,
        queue=lambda task: calls.append(("queue", task)) or True,
    ) == 1
    assert [entry[0] for entry in calls] == ["list", "claim", "queue", "queued"]


def test_generation_delivery_reconciliation_uses_stale_queued_window():
    from api.db.services.task_service import TaskService

    implementation = TaskService.list_generation_delivery_unknown_tasks.__func__
    while hasattr(implementation, "__wrapped__"):
        implementation = implementation.__wrapped__
    source = Path(TaskService.__module__.replace(".", "/") + ".py")
    source_text = source.read_text(encoding="utf-8")
    source_tree = ast.parse(source_text)
    method_source = ast.get_source_segment(
        source_text,
        next(
            node
            for node in ast.walk(source_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "list_generation_delivery_unknown_tasks"
        ),
    )
    assert method_source is not None
    assert "TABULAR_GENERATION_DELIVERY_RECONCILIATION_STALE_SECONDS" in method_source
    assert "cls.model.update_time <= stale_before" in method_source
    assert "cls.model.progress == 0" in method_source
    assert implementation.__name__ == "list_generation_delivery_unknown_tasks"


def test_generation_delivery_reconciliation_claim_accepts_stale_unmarked_tasks():
    from api.db.services.task_service import TaskService

    implementation = TaskService.claim_generation_delivery_reconciliation.__func__
    while hasattr(implementation, "__wrapped__"):
        implementation = implementation.__wrapped__
    source = Path(TaskService.__module__.replace(".", "/") + ".py")
    source_text = source.read_text(encoding="utf-8")
    source_tree = ast.parse(source_text)
    method = next(
        node
        for node in ast.walk(source_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "claim_generation_delivery_reconciliation"
    )
    method_source = ast.get_source_segment(source_text, method)
    assert method_source is not None
    assert "claimed_is_stale" in method_source
    assert "current_timestamp() - task.update_time" in method_source
    assert "cls.model.update_time <= stale_before" in method_source
    assert implementation.__name__ == "claim_generation_delivery_reconciliation"


def test_generation_task_read_preserves_joined_execution_context():
    from api.db.services.task_service import TaskService

    implementation = TaskService.get_generation_task.__func__
    while hasattr(implementation, "__wrapped__"):
        implementation = implementation.__wrapped__
    source = Path(TaskService.__module__.replace(".", "/") + ".py")
    source_text = source.read_text(encoding="utf-8")
    source_tree = ast.parse(source_text)
    method = next(
        node
        for node in ast.walk(source_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_generation_task"
    )
    method_source = ast.get_source_segment(source_text, method)
    assert method_source is not None
    assert ".dicts()" in method_source
    assert "return task.to_dict()" not in method_source
    assert 'Knowledgebase.tenant_id.alias("tenant_id")' in method_source
    assert implementation.__name__ == "get_generation_task"


def test_generation_job_rejects_legacy_source_pending_tasks():
    result = run_tabular_structure_generation_job(
        {
            "tenant_id": "tenant-1",
            "kb_id": "dataset-1",
            "doc_id": "document-1",
            "name": "workbook.xlsx",
            "digest": "source-pending",
        },
        b"workbook",
        service=object(),
    )

    assert result == {
        "status": "failed",
        "safe_error_code": "legacy_generation_task_requires_source_snapshot",
    }


def test_generation_job_skips_completed_sheet_checkpoints_and_does_not_activate_early():
    calls = []
    generation_ref = structure_generation_ref("document-1", b"workbook")
    source_sha256 = hashlib.sha256(b"workbook").hexdigest()

    class Workbook:
        sheetnames = ["Sheet 1", "Sheet 2"]

    checkpoints = {
        1: {
            "version": runtime.TABULAR_GENERATION_CHECKPOINT_VERSION,
            "producer_generation_ref": generation_ref,
            "source_sha256": source_sha256,
            "sheet_ordinal": 1,
            "projection": {
                "version": "tabular-structure-projection/v6",
                "producer_schema_version": "table-producer/v6",
                "producer_generation_ref": generation_ref,
                "structure_algorithm_version": "region-producer/v22",
                "enumeration_rule_version": "enumeration-rules/v9",
                "source_sha256": source_sha256,
                "tables": [],
                "rows": [],
            },
            "audit": {},
        },
    }

    class Storage:
        def __init__(self):
            self.objects = {}

        def obj_exist(self, _bucket, name, tenant_id=None):
            return name.endswith("sheet-000001.json") or name in self.objects

        def get(self, _bucket, name, tenant_id=None):
            if name.endswith("sheet-000001.json"):
                return json.dumps(checkpoints[1]).encode()
            return self.objects[name]

        def put(self, _bucket, name, payload, tenant_id=None):
            self.objects[name] = payload
            calls.append("put")

    class Service:
        class StructureSnapshotMissing(LookupError):
            pass

        @staticmethod
        def get_active_generation(**_kwargs):
            raise Service.StructureSnapshotMissing()

        @staticmethod
        def persist_shadow_generation(*_args, **_kwargs):
            calls.append("shadow")

        @staticmethod
        def activate_generation(*_args, **_kwargs):
            calls.append("activate")

    def build(*_args, **kwargs):
        calls.append(kwargs["sheet_ordinals"])
        if kwargs["sheet_ordinals"] == {2}:
            return {
                "version": "tabular-structure-projection/v6",
                "producer_schema_version": "table-producer/v6",
                "producer_generation_ref": generation_ref,
                "structure_algorithm_version": "region-producer/v22",
                "enumeration_rule_version": "enumeration-rules/v9",
                "source_sha256": source_sha256,
                "tables": [],
                "rows": [],
            }, {}
        raise AssertionError("completed Sheet must not be rebuilt")

    result = run_tabular_structure_generation_job(
        {
            "tenant_id": "tenant-1",
            "kb_id": "dataset-1",
            "doc_id": "document-1",
            "name": "workbook.xlsx",
        },
        b"workbook",
        storage=Storage(),
        service=Service,
        projection_builder=build,
        sheet_count_provider=lambda *_args: 2,
        source_context_provider=lambda _binary: {"workbook": Workbook()},
    )

    assert result["status"] == "shadow"
    assert {2} in calls
    assert calls[-1] == "shadow"


def test_generation_job_keeps_a_sheet_that_exceeds_the_legacy_budget():
    """A slow Sheet remains resumable; the legacy budget is observational only."""
    calls = []
    generation_ref = structure_generation_ref("document-1", b"workbook")
    source_sha256 = hashlib.sha256(b"workbook").hexdigest()

    class Workbook:
        sheetnames = ["Sheet 1"]

    class Storage:
        def __init__(self):
            self.objects = {}

        def obj_exist(self, _bucket, name, tenant_id=None):
            return name in self.objects

        def get(self, _bucket, name, tenant_id=None):
            return self.objects[name]

        def put(self, _bucket, name, payload, tenant_id=None):
            self.objects[name] = payload
            calls.append("checkpoint")

    class Service:
        class StructureSnapshotMissing(LookupError):
            pass

        @staticmethod
        def get_active_generation(**_kwargs):
            raise Service.StructureSnapshotMissing()

        @staticmethod
        def persist_shadow_generation(*_args, **_kwargs):
            calls.append("shadow")

    def projection():
        return {
            "version": "tabular-structure-projection/v6",
            "producer_schema_version": "table-producer/v6",
            "producer_generation_ref": generation_ref,
            "structure_algorithm_version": "region-producer/v22",
            "enumeration_rule_version": "enumeration-rules/v9",
            "source_sha256": source_sha256,
            "tables": [],
            "rows": [],
        }, {}

    result = run_tabular_structure_generation_job(
        {
            "tenant_id": "tenant-1",
            "kb_id": "dataset-1",
            "doc_id": "document-1",
            "name": "workbook.xlsx",
        },
        b"workbook",
        storage=Storage(),
        service=Service,
        projection_builder=lambda *_args, **_kwargs: projection(),
        sheet_count_provider=lambda *_args: 1,
        source_context_provider=lambda _binary: {"workbook": Workbook()},
    )

    assert result["status"] == "shadow"
    assert calls == ["checkpoint", "shadow"]
    assert "activate" not in calls


def test_generation_ref_from_source_snapshot_matches_worker_bytes_ref():
    source = b"workbook"
    digest = hashlib.sha256(source).hexdigest()
    assert structure_generation_ref_from_source_sha256("document-1", digest) == structure_generation_ref(
        "document-1", source
    )


def test_generation_progress_never_reports_complete_before_activation():
    progress = []

    class Workbook:
        sheetnames = ["Sheet 1"]

    class Storage:
        def __init__(self):
            self.objects = {}

        def obj_exist(self, _bucket, name, **_kwargs):
            return name in self.objects

        def put(self, _bucket, name, payload, **_kwargs):
            self.objects[name] = payload

        def get(self, _bucket, name, **_kwargs):
            return self.objects[name]

    class Service:
        class StructureSnapshotMissing(LookupError):
            pass

        @staticmethod
        def get_active_generation(**_kwargs):
            raise Service.StructureSnapshotMissing()

        @staticmethod
        def persist_shadow_generation(*_args, **_kwargs):
            pass

        @staticmethod
        def activate_generation(*_args, **_kwargs):
            pass

    projection = {
        "version": "tabular-structure-projection/v6",
        "producer_schema_version": "table-producer/v6",
        "producer_generation_ref": structure_generation_ref("document-1", b"workbook"),
        "structure_algorithm_version": "region-producer/v22",
        "enumeration_rule_version": "enumeration-rules/v9",
        "source_sha256": hashlib.sha256(b"workbook").hexdigest(),
        "tables": [],
        "rows": [],
    }

    result = run_tabular_structure_generation_job(
        {
            "tenant_id": "tenant-1",
            "kb_id": "dataset-1",
            "doc_id": "document-1",
            "name": "workbook.xlsx",
        },
        b"workbook",
        storage=Storage(),
        service=Service,
        projection_builder=lambda *_args, **_kwargs: (projection, {}),
        sheet_count_provider=lambda *_args: 1,
        source_context_provider=lambda _binary: {"workbook": Workbook()},
        progress_callback=lambda value, _message: progress.append(value),
    )

    assert result["status"] == "shadow"
    assert progress
    assert max(progress) < 1.0


def test_generation_job_failure_keeps_completed_checkpoints_and_never_activates():
    calls = []

    class Workbook:
        sheetnames = ["Sheet 1", "Sheet 2"]

    class Storage:
        def __init__(self):
            self.objects = {}

        def obj_exist(self, _bucket, name, **_kwargs):
            return name in self.objects

        def put(self, _bucket, name, payload, tenant_id=None):
            self.objects[name] = payload
            calls.append(("checkpoint", name, payload))

        def get(self, _bucket, name, tenant_id=None):
            return self.objects[name]

    class Service:
        class StructureSnapshotMissing(LookupError):
            pass

        @staticmethod
        def get_active_generation(**_kwargs):
            raise Service.StructureSnapshotMissing()

        @staticmethod
        def persist_shadow_generation(*_args, **_kwargs):
            calls.append(("shadow",))

        @staticmethod
        def activate_generation(*_args, **_kwargs):
            calls.append(("activate",))

    def build(*_args, **kwargs):
        if kwargs["sheet_ordinals"] == {2}:
            raise TimeoutError("sheet budget exceeded")
        return {
            "version": "tabular-structure-projection/v6",
            "producer_schema_version": "table-producer/v6",
            "producer_generation_ref": structure_generation_ref("document-1", b"workbook"),
            "structure_algorithm_version": "region-producer/v22",
            "enumeration_rule_version": "enumeration-rules/v9",
            "source_sha256": hashlib.sha256(b"workbook").hexdigest(),
            "tables": [],
            "rows": [],
        }, {}

    result = run_tabular_structure_generation_job(
        {
            "tenant_id": "tenant-1",
            "kb_id": "dataset-1",
            "doc_id": "document-1",
            "name": "workbook.xlsx",
        },
        b"workbook",
        storage=Storage(),
        service=Service,
        projection_builder=build,
        sheet_count_provider=lambda *_args: 2,
        source_context_provider=lambda _binary: {"workbook": Workbook()},
    )

    assert result["status"] == "failed"
    assert any(entry[0] == "checkpoint" for entry in calls)
    assert not any(entry[0] in {"shadow", "activate"} for entry in calls)


def test_generation_job_cancellation_keeps_completed_checkpoints_and_never_activates():
    calls = []

    class Workbook:
        sheetnames = ["Sheet 1", "Sheet 2"]

    class Storage:
        def __init__(self):
            self.objects = {}

        def obj_exist(self, _bucket, name, **_kwargs):
            return name in self.objects

        def put(self, _bucket, name, payload, tenant_id=None):
            self.objects[name] = payload
            calls.append(("checkpoint", name))

        def get(self, _bucket, name, tenant_id=None):
            return self.objects[name]

    class Service:
        class StructureSnapshotMissing(LookupError):
            pass

        @staticmethod
        def get_active_generation(**_kwargs):
            raise Service.StructureSnapshotMissing()

        @staticmethod
        def persist_shadow_generation(*_args, **_kwargs):
            calls.append(("shadow",))

        @staticmethod
        def activate_generation(*_args, **_kwargs):
            calls.append(("activate",))

    def build(*_args, **kwargs):
        calls.append(("build", kwargs["sheet_ordinals"]))
        generation = structure_generation_ref("document-1", b"workbook")
        return {
            "version": "tabular-structure-projection/v6",
            "producer_schema_version": "table-producer/v6",
            "producer_generation_ref": generation,
            "structure_algorithm_version": "region-producer/v22",
            "enumeration_rule_version": "enumeration-rules/v9",
            "source_sha256": hashlib.sha256(b"workbook").hexdigest(),
            "tables": [],
            "rows": [],
        }, {}

    checks = 0

    def cancel_check():
        nonlocal checks
        checks += 1
        return checks >= 4

    result = run_tabular_structure_generation_job(
        {
            "tenant_id": "tenant-1",
            "kb_id": "dataset-1",
            "doc_id": "document-1",
            "name": "workbook.xlsx",
        },
        b"workbook",
        storage=Storage(),
        service=Service,
        projection_builder=build,
        sheet_count_provider=lambda *_args: 2,
        source_context_provider=lambda _binary: {"workbook": Workbook()},
        cancel_check=cancel_check,
    )

    assert result == {
        "status": "cancelled",
        "producer_generation_ref": structure_generation_ref("document-1", b"workbook"),
        "safe_error_code": "task_cancelled",
    }
    assert [entry[0] for entry in calls] == ["build", "checkpoint"]
    assert not any(entry[0] in {"shadow", "activate"} for entry in calls)


def test_generation_job_retry_reuses_completed_checkpoints_after_sheet_failure():
    calls = []
    generation = structure_generation_ref("document-1", b"workbook")
    source_sha256 = hashlib.sha256(b"workbook").hexdigest()

    class Workbook:
        sheetnames = ["Sheet 1", "Sheet 2"]

    class Storage:
        def __init__(self):
            self.objects = {}

        def obj_exist(self, _bucket, name, **_kwargs):
            return name in self.objects

        def put(self, _bucket, name, payload, tenant_id=None):
            self.objects[name] = payload

        def get(self, _bucket, name, tenant_id=None):
            return self.objects[name]

    class Service:
        class StructureSnapshotMissing(LookupError):
            pass

        @staticmethod
        def get_active_generation(**_kwargs):
            raise Service.StructureSnapshotMissing()

        @staticmethod
        def persist_shadow_generation(*_args, **_kwargs):
            calls.append("shadow")

        @staticmethod
        def activate_generation(*_args, **_kwargs):
            calls.append("activate")

    def projection():
        return {
            "version": "tabular-structure-projection/v6",
            "producer_schema_version": "table-producer/v6",
            "producer_generation_ref": generation,
            "structure_algorithm_version": "region-producer/v22",
            "enumeration_rule_version": "enumeration-rules/v9",
            "source_sha256": source_sha256,
            "tables": [],
            "rows": [],
        }, {}

    storage = Storage()
    failed_once = True

    def build(_name, _binary, **kwargs):
        nonlocal failed_once
        ordinal = next(iter(kwargs["sheet_ordinals"]))
        calls.append(("build", ordinal))
        if ordinal == 2 and failed_once:
            failed_once = False
            raise TimeoutError("sheet budget exceeded")
        return projection()

    task = {
        "tenant_id": "tenant-1",
        "kb_id": "dataset-1",
        "doc_id": "document-1",
        "name": "workbook.xlsx",
    }
    first = run_tabular_structure_generation_job(
        task,
        b"workbook",
        storage=storage,
        service=Service,
        projection_builder=build,
        sheet_count_provider=lambda *_args: 2,
        source_context_provider=lambda _binary: {"workbook": Workbook()},
    )
    second = run_tabular_structure_generation_job(
        task,
        b"workbook",
        storage=storage,
        service=Service,
        projection_builder=build,
        sheet_count_provider=lambda *_args: 2,
        source_context_provider=lambda _binary: {"workbook": Workbook()},
    )

    assert first["status"] == "failed"
    assert second["status"] == "shadow"
    assert [entry for entry in calls if isinstance(entry, tuple)] == [
        ("build", 1),
        ("build", 2),
        ("build", 2),
    ]
    assert calls[-1] == "shadow"


def test_generation_job_rejects_a_sheet_count_that_could_skip_real_sheets():
    calls = []

    class Workbook:
        sheetnames = ["Sheet 1", "Sheet 2"]

    class Service:
        class StructureSnapshotMissing(LookupError):
            pass

        @staticmethod
        def get_active_generation(**_kwargs):
            raise Service.StructureSnapshotMissing()

        @staticmethod
        def persist_shadow_generation(*_args, **_kwargs):
            calls.append("shadow")

        @staticmethod
        def activate_generation(*_args, **_kwargs):
            calls.append("activate")

    result = run_tabular_structure_generation_job(
        {
            "tenant_id": "tenant-1",
            "kb_id": "dataset-1",
            "doc_id": "document-1",
            "name": "workbook.xlsx",
        },
        b"workbook",
        storage=type("Storage", (), {})(),
        service=Service,
        projection_builder=lambda *_args, **_kwargs: {},
        sheet_count_provider=lambda *_args: 1,
        source_context_provider=lambda _binary: {"workbook": Workbook()},
    )

    assert result["status"] == "failed"
    assert calls == []


def test_generation_job_rejects_a_changed_source_without_retrying_or_activating():
    calls = []

    class Service:
        @staticmethod
        def get_active_generation(**_kwargs):
            calls.append("active")
            return None

        @staticmethod
        def persist_shadow_generation(*_args, **_kwargs):
            calls.append("shadow")

        @staticmethod
        def activate_generation(*_args, **_kwargs):
            calls.append("activate")

    result = run_tabular_structure_generation_job(
        {
            "tenant_id": "tenant-1",
            "kb_id": "dataset-1",
            "doc_id": "document-1",
            "name": "workbook.xlsx",
            "digest": hashlib.sha256(b"original").hexdigest(),
        },
        b"changed",
        storage=type("Storage", (), {})(),
        service=Service,
    )

    assert result == {"status": "failed", "safe_error_code": "source_changed"}
    assert calls == []


def test_generation_retry_allows_at_most_five_total_attempts():
    calls = []
    transitions = []
    queued = []

    class Service:
        @staticmethod
        def fail_generation_attempt_and_reserve_retry(task_id, *, message, max_attempts, retryable):
            transitions.append((task_id, message, max_attempts, retryable))
            retry_count = len(transitions)
            return (
                {"status": "retry", "retry_count": retry_count}
                if retry_count < max_attempts
                else {"status": "exhausted", "retry_count": retry_count - 1}
            )

    for _ in range(TABULAR_GENERATION_MAX_ATTEMPTS):
        retry_tabular_structure_generation_task(
            "generation-task",
            task_service=Service,
            queue=lambda task: (queued.append(task), True)[1],
        )

    assert len(transitions) == TABULAR_GENERATION_MAX_ATTEMPTS
    assert len(queued) == TABULAR_GENERATION_MAX_ATTEMPTS - 1


def test_generation_retry_preserves_delivery_when_retry_queue_returns_false():
    calls = []

    class Service:
        @staticmethod
        def fail_generation_attempt_and_reserve_retry(task_id, *, message, max_attempts, retryable):
            calls.append(("transition", task_id, message, max_attempts, retryable))
            return {"status": "retry", "retry_count": 1}

        @staticmethod
        def mark_generation_delivery_unknown(task_id, *, message):
            calls.append(("unknown", task_id, message))

    with pytest.raises(TabularGenerationRetryUnavailable):
        retry_tabular_structure_generation_task(
            "generation-task",
            task_service=Service,
            queue=lambda _task: False,
        )
    assert calls[0][0] == "transition"
    assert calls[1][0] == "unknown"


def test_generation_retry_preserves_delivery_when_retry_queue_raises():
    calls = []

    class Service:
        @staticmethod
        def fail_generation_attempt_and_reserve_retry(task_id, *, message, max_attempts, retryable):
            calls.append(("transition", task_id, message, max_attempts, retryable))
            return {"status": "retry", "retry_count": 1}

        @staticmethod
        def fail_generation_retry_delivery(task_id, *, message):
            calls.append(("fail", task_id, message))
            return True

    def queue(_task):
        raise ConnectionError("queue unavailable")

    with pytest.raises(TabularGenerationRetryUnavailable):
        retry_tabular_structure_generation_task(
            "generation-task",
            task_service=Service,
            queue=queue,
        )
    assert calls == [
        ("transition", "generation-task", "tabular_generation_failed", TABULAR_GENERATION_MAX_ATTEMPTS, True),
    ]


def test_generation_retry_does_not_claim_deterministic_source_change():
    calls = []

    class Service:
        @staticmethod
        def fail_generation_attempt_and_reserve_retry(task_id, *, message, max_attempts, retryable):
            calls.append(("transition", task_id, message, max_attempts, retryable))
            return {"status": "failed", "retry_count": 0}

    assert retry_tabular_structure_generation_task(
        "generation-task",
        task_service=Service,
        queue=lambda _task: calls.append("queue"),
        failure_code="source_changed",
    ) is False
    assert calls == [
        ("transition", "generation-task", "source_changed", TABULAR_GENERATION_MAX_ATTEMPTS, False),
    ]


def test_generation_retry_treats_claim_service_failure_as_terminal_for_this_worker_attempt():
    calls = []

    class Service:
        @staticmethod
        def fail_generation_attempt_and_reserve_retry(*_args, **_kwargs):
            calls.append("transition")
            raise ConnectionError("database unavailable")

    with pytest.raises(TabularGenerationRetryUnavailable):
        retry_tabular_structure_generation_task(
            "generation-task",
            task_service=Service,
            queue=lambda _task: calls.append("queue"),
        )
    assert calls == [
        "transition",
    ]


def test_generation_retry_preserves_delivery_when_marking_failure_is_unavailable():
    calls = []

    class Service:
        @staticmethod
        def fail_generation_attempt_and_reserve_retry(_task_id, *, message, max_attempts, retryable):
            calls.append(message)
            raise ConnectionError("database unavailable")

    with pytest.raises(TabularGenerationRetryUnavailable):
        retry_tabular_structure_generation_task(
            "generation-task",
            task_service=Service,
            queue=lambda _task: calls.append("queue"),
            failure_code="worker_error",
        )
    assert calls == ["worker_error"]


def test_generation_retry_finishes_the_attempt_and_claims_requeue_in_one_transition():
    calls = []

    class Service:
        @staticmethod
        def fail_generation_attempt_and_reserve_retry(task_id, *, message, max_attempts, retryable):
            calls.append(("transition", task_id, message, max_attempts, retryable))
            return {"status": "retry", "retry_count": 1}

    assert retry_tabular_structure_generation_task(
        "generation-task",
        task_service=Service,
        queue=lambda _task: (calls.append("queue"), True)[1],
        failure_code="worker_error",
    ) is True
    assert calls == [
        ("transition", "generation-task", "worker_error", TABULAR_GENERATION_MAX_ATTEMPTS, True),
        "queue",
    ]


def test_generation_retry_does_not_requeue_when_transition_is_unavailable():
    calls = []

    class Service:
        @staticmethod
        def fail_generation_attempt_and_reserve_retry(*_args, **_kwargs):
            calls.append("transition")
            return None

    with pytest.raises(TabularGenerationRetryUnavailable):
        retry_tabular_structure_generation_task(
            "generation-task",
            task_service=Service,
            queue=lambda _task: calls.append("queue"),
            failure_code="worker_error",
        )
    assert calls == ["transition"]


def test_generation_retry_does_not_ack_when_retry_delivery_outcome_is_unknown():
    class Service:
        @staticmethod
        def fail_generation_attempt_and_reserve_retry(*_args, **_kwargs):
            return {"status": "retry", "retry_count": 1}

        @staticmethod
        def fail_generation_retry_delivery(*_args, **_kwargs):
            raise ConnectionError("database unavailable")

    with pytest.raises(TabularGenerationRetryUnavailable):
        retry_tabular_structure_generation_task(
            "generation-task",
            task_service=Service,
            queue=lambda _task: False,
        )


def test_generation_retry_rejects_non_boolean_queue_outcome():
    class Service:
        @staticmethod
        def fail_generation_attempt_and_reserve_retry(*_args, **_kwargs):
            return {"status": "retry", "retry_count": 1}

    with pytest.raises(TabularGenerationRetryUnavailable):
        retry_tabular_structure_generation_task(
            "generation-task",
            task_service=Service,
            queue=lambda _task: None,
        )


def test_generation_retry_rejects_attempt_cap_above_five():
    class Service:
        @staticmethod
        def fail_generation_attempt_and_reserve_retry(*_args, **_kwargs):
            raise AssertionError("service must not receive an invalid cap")

    with pytest.raises(ValueError, match="at most 5"):
        retry_tabular_structure_generation_task(
            "generation-task",
            task_service=Service,
            max_attempts=6,
        )


def test_generation_attempt_claim_returns_false_for_duplicate_message():
    class Service:
        @staticmethod
        def claim_generation_attempt(_task_id):
            return {"status": "running"}

    assert claim_tabular_structure_generation_attempt(
        "generation-task", task_service=Service
    ) is False


def test_generation_attempt_claim_preserves_message_when_service_is_unavailable():
    class Service:
        @staticmethod
        def claim_generation_attempt(_task_id):
            raise ConnectionError("database unavailable")

    with pytest.raises(TabularGenerationClaimUnavailable):
        claim_tabular_structure_generation_attempt("generation-task", task_service=Service)


def test_generation_retry_claim_requires_failed_state_and_returns_to_queued_state():
    source = Path(runtime.__file__).parents[2] / "api" / "db" / "services" / "task_service.py"
    service_source = source.read_text(encoding="utf-8")
    method = service_source[service_source.index("def fail_generation_attempt_and_reserve_retry"):]
    assert "with DB.atomic()" in method
    assert "retryable" in method
    assert "& (cls.model.progress > 0)" in method
    assert "& (cls.model.progress < 1)" in method
    assert "progress=0.0" in method


def _task(task_id, progress, *, parser_id="table"):
    return {
        "id": task_id,
        "doc_id": "document-1",
        "parser_id": parser_id,
        "progress": progress,
    }


def test_structure_generation_waits_for_every_row_range_task():
    current = _task("task-2", 1.0)
    tasks = [_task("task-1", 1.0), current, _task("task-3", 0.5)]

    assert is_complete_tabular_parse({**current, "progress": 0.5}, [_task("task-1", 1.0), {**current, "progress": 0.5}]) is False
    assert is_complete_tabular_parse(current, tasks) is False
    assert is_complete_tabular_parse(current, [_task("task-1", 1.0), current, _task("task-3", 1.0)]) is True


def test_non_table_tasks_never_publish_a_structure_generation():
    current = _task("task-1", 1.0, parser_id="naive")

    assert is_complete_tabular_parse(current, [current]) is False


def test_generation_ref_is_idempotent_for_same_document_and_source():
    first = structure_generation_ref("document-1", b"same workbook")
    second = structure_generation_ref("document-1", b"same workbook")
    changed = structure_generation_ref("document-1", b"changed workbook")

    uuid.UUID(first)
    assert first == second
    assert first != changed


def test_generation_ref_includes_the_producer_schema_version(monkeypatch):
    original = structure_generation_ref("document-1", b"workbook")
    monkeypatch.setattr(tabular_structure, "PRODUCER_SCHEMA_VERSION", "table-producer/test-next")
    changed = structure_generation_ref("document-1", b"workbook")

    assert original != changed


def test_generation_ref_includes_the_projection_version(monkeypatch):
    original = structure_generation_ref("document-1", b"workbook")
    monkeypatch.setattr(
        tabular_structure,
        "PROJECTION_VERSION",
        "tabular-structure-projection/test-next",
    )
    changed = structure_generation_ref("document-1", b"workbook")

    assert original != changed


def test_generation_ref_includes_the_multi_region_algorithm_version(monkeypatch):
    original = structure_generation_ref("document-1", b"workbook")
    monkeypatch.setattr(tabular_structure, "STRUCTURE_PRODUCER_ALGORITHM_VERSION", "region-producer/test-next")
    changed = structure_generation_ref("document-1", b"workbook")

    assert original != changed


def test_generation_ref_includes_the_enumeration_rule_version(monkeypatch):
    original = structure_generation_ref("document-1", b"workbook")
    monkeypatch.setattr(
        tabular_structure,
        "ENUMERATION_RULE_VERSION",
        "enumeration-rules/test-next",
        raising=False,
    )
    changed = structure_generation_ref("document-1", b"workbook")

    assert original != changed


def test_adr044_generation_ref_binds_original_and_converter_identity():
    converted = b"converted workbook"
    receipt = {
        "original_source_sha256": "1" * 64,
        "converted_source_sha256": hashlib.sha256(converted).hexdigest(),
        "converter_version": "converter-build/one",
    }

    original = structure_generation_ref(
        "document-1",
        converted,
        adr044_conversion_receipt=receipt,
    )
    changed_source = structure_generation_ref(
        "document-1",
        converted,
        adr044_conversion_receipt={**receipt, "original_source_sha256": "2" * 64},
    )
    changed_converter = structure_generation_ref(
        "document-1",
        converted,
        adr044_conversion_receipt={**receipt, "converter_version": "converter-build/two"},
    )

    assert len({original, changed_source, changed_converter}) == 3


def test_runtime_reexports_the_structure_producer_versions():
    assert runtime.STRUCTURE_PRODUCER_ALGORITHM_VERSION == (
        tabular_structure.STRUCTURE_PRODUCER_ALGORITHM_VERSION
    )
    assert runtime.ENUMERATION_RULE_VERSION == tabular_structure.ENUMERATION_RULE_VERSION


def test_structure_publication_failure_is_safe_for_ordinary_parse(monkeypatch, caplog):
    current = {
        **_task("task-1", 1.0),
        "name": "anonymous.xlsx",
        "tenant_id": "tenant-1",
        "kb_id": "dataset-1",
    }

    def fail(*_args, **_kwargs):
        raise ValueError("synthetic structure failure")

    monkeypatch.setattr(runtime, "structure_generation_ref", lambda *_args: str(uuid.uuid4()))
    monkeypatch.setattr(runtime, "_active_generation_ref", lambda *_args, **_kwargs: None)
    result = publish_tabular_structure_generation(
        current,
        b"workbook",
        tasks=[current],
        projection_builder=fail,
    )

    assert result["status"] == "failed"
    assert "synthetic structure failure" not in caplog.text


def test_structure_publication_is_shadow_first_and_idempotent():
    current = {
        **_task("task-1", 1.0),
        "name": "anonymous.xlsx",
        "tenant_id": "tenant-1",
        "kb_id": "dataset-1",
    }
    calls = []
    generation_ref = structure_generation_ref("document-1", b"workbook")

    def build(name, binary, *, producer_generation_ref):
        calls.append(("build", name, binary, producer_generation_ref))
        return {"rows": [{"row_ref": "row-1"}]}

    def store(storage, **kwargs):
        calls.append(("store", storage, kwargs["projection"]))
        return {
            "producer_generation_ref": generation_ref,
            "manifest_object_name": "manifest.json",
            "manifest_sha256": "a" * 64,
            "part_count": 1,
            "row_count": 1,
        }

    class Service:
        active = None

        class StructureSnapshotMissing(LookupError):
            pass

        @staticmethod
        def persist_shadow_generation(storage, **kwargs):
            receipt = kwargs["projection_store"](
                storage,
                bucket=kwargs["dataset_id"],
                document_id=kwargs["document_id"],
                projection=kwargs["projection"],
                tenant_id=kwargs["tenant_id"],
            )
            calls.append(("persist", storage, receipt))
            return {"status": "shadow"}

        @classmethod
        def get_active_generation(cls, **kwargs):
            if Service.active is None:
                raise Service.StructureSnapshotMissing("active structure generation is missing")
            return {"producer_generation_ref": Service.active}

        @classmethod
        def activate_generation(cls, storage, **kwargs):
            calls.append(("activate", storage, kwargs["producer_generation_ref"]))
            Service.active = kwargs["producer_generation_ref"]
            return {"status": "active"}

    first = publish_tabular_structure_generation(
        current,
        b"workbook",
        tasks=[current],
        storage="storage",
        service=Service,
        projection_builder=build,
        projection_store=store,
    )
    second = publish_tabular_structure_generation(
        current,
        b"workbook",
        tasks=[current],
        storage="storage",
        service=Service,
        projection_builder=build,
        projection_store=store,
    )

    assert first["status"] == second["status"] == "active"
    assert [item[0] for item in calls] == [
        "build", "store", "persist", "activate",
    ]


def test_structure_runtime_uses_one_guarded_store_and_registration_entrypoint():
    runtime_path = Path(runtime.__file__)
    source = runtime_path.read_text(encoding="utf-8")
    module = __import__("ast").parse(source)
    builder = next(
        node
        for node in module.body
        if isinstance(node, __import__("ast").FunctionDef)
        and node.name == "build_tabular_structure_shadow_from_source"
    )
    builder_source = __import__("ast").get_source_segment(source, builder)

    assert builder_source is not None
    assert "service.persist_shadow_generation(" in builder_source
    assert "service.register_shadow_generation(" not in builder_source


def test_explicit_receipt_publication_binds_identity_and_reaches_the_producer():
    converted = b"converted workbook"
    receipt = {
        "original_source_sha256": "1" * 64,
        "converted_source_sha256": hashlib.sha256(converted).hexdigest(),
        "converter_version": "converter-build/one",
    }
    current = {
        **_task("task-1", 1.0),
        "name": "anonymous.xlsx",
        "tenant_id": "tenant-1",
        "kb_id": "dataset-1",
    }
    expected_generation_ref = structure_generation_ref(
        "document-1",
        converted,
        adr044_conversion_receipt=receipt,
    )
    builder_calls = []

    class Service:
        class StructureSnapshotMissing(LookupError):
            pass

        @staticmethod
        def get_active_generation(**_kwargs):
            raise Service.StructureSnapshotMissing()

        @staticmethod
        def persist_shadow_generation(storage, **kwargs):
            return kwargs["projection_store"](
                storage,
                bucket=kwargs["dataset_id"],
                document_id=kwargs["document_id"],
                projection=kwargs["projection"],
                tenant_id=kwargs["tenant_id"],
            )

        @staticmethod
        def activate_generation(*_args, **_kwargs):
            return None

    def build(_name, _binary, **kwargs):
        builder_calls.append(kwargs)
        return {"rows": [], "producer_generation_ref": kwargs["producer_generation_ref"]}

    result = publish_tabular_structure_generation(
        current,
        converted,
        tasks=[current],
        adr044_conversion_receipt=receipt,
        storage="storage",
        service=Service,
        projection_builder=build,
        projection_store=lambda *_args, **_kwargs: {
            "producer_generation_ref": expected_generation_ref,
            "manifest_object_name": "manifest.json",
            "manifest_sha256": "a" * 64,
            "part_count": 1,
            "row_count": 0,
        },
    )

    assert result == {
        "status": "active",
        "producer_generation_ref": expected_generation_ref,
        "row_count": 0,
    }
    assert builder_calls == [
        {
            "producer_generation_ref": expected_generation_ref,
            "adr044_conversion_receipt": receipt,
        }
    ]


def test_invalid_explicit_adr044_receipt_fails_without_calling_the_producer():
    converted = b"converted workbook"
    current = {
        **_task("task-1", 1.0),
        "name": "anonymous.xlsx",
        "tenant_id": "tenant-1",
        "kb_id": "dataset-1",
    }
    builder_calls = []

    result = publish_tabular_structure_generation(
        current,
        converted,
        tasks=[current],
        adr044_conversion_receipt={
            "original_source_sha256": "1" * 64,
            "converted_source_sha256": "2" * 64,
            "converter_version": "converter-build/one",
        },
        projection_builder=lambda *_args, **_kwargs: builder_calls.append(True),
    )

    assert result == {"status": "failed"}
    assert builder_calls == []


def test_publication_without_a_receipt_does_not_infer_adr044_identity():
    current = {
        **_task("task-1", 1.0),
        "name": "anonymous.xlsx",
        "tenant_id": "tenant-1",
        "kb_id": "dataset-1",
    }
    binary = b"workbook bytes without a governed conversion receipt"
    generation_ref = structure_generation_ref("document-1", binary)
    builder_calls = []

    class Service:
        class StructureSnapshotMissing(LookupError):
            pass

        @staticmethod
        def get_active_generation(**_kwargs):
            raise Service.StructureSnapshotMissing()

        @staticmethod
        def persist_shadow_generation(storage, **kwargs):
            return kwargs["projection_store"](
                storage,
                bucket=kwargs["dataset_id"],
                document_id=kwargs["document_id"],
                projection=kwargs["projection"],
                tenant_id=kwargs["tenant_id"],
            )

        @staticmethod
        def activate_generation(*_args, **_kwargs):
            return None

    def build(_name, _binary, **kwargs):
        builder_calls.append(kwargs)
        return {"rows": [], "producer_generation_ref": kwargs["producer_generation_ref"]}

    result = publish_tabular_structure_generation(
        current,
        binary,
        tasks=[current],
        storage="storage",
        service=Service,
        projection_builder=build,
        projection_store=lambda *_args, **_kwargs: {
            "producer_generation_ref": generation_ref,
            "manifest_object_name": "manifest.json",
            "manifest_sha256": "a" * 64,
            "part_count": 1,
            "row_count": 0,
        },
    )

    assert result == {
        "status": "active",
        "producer_generation_ref": generation_ref,
        "row_count": 0,
    }
    assert builder_calls == [{"producer_generation_ref": generation_ref}]


def test_concurrent_shadow_registration_accepts_an_already_active_generation():
    current = {
        **_task("task-1", 1.0),
        "name": "anonymous.xlsx",
        "tenant_id": "tenant-1",
        "kb_id": "dataset-1",
    }
    generation_ref = structure_generation_ref("document-1", b"workbook")

    class Service:
        active_reads = 0

        class StructureSnapshotMissing(LookupError):
            pass

        @classmethod
        def get_active_generation(cls, **_kwargs):
            cls.active_reads += 1
            if cls.active_reads == 1:
                raise cls.StructureSnapshotMissing("not active yet")
            return {"producer_generation_ref": generation_ref}

        @staticmethod
        def persist_shadow_generation(*_args, **_kwargs):
            raise RuntimeError("concurrent generation state changed")

    result = publish_tabular_structure_generation(
        current,
        b"workbook",
        tasks=[current],
        storage="storage",
        service=Service,
        projection_builder=lambda *_args, **_kwargs: {"rows": []},
        projection_store=lambda *_args, **_kwargs: {
            "producer_generation_ref": generation_ref,
            "manifest_object_name": "manifest.json",
            "manifest_sha256": "a" * 64,
            "part_count": 0,
            "row_count": 0,
        },
    )

    assert result == {"status": "active", "producer_generation_ref": generation_ref}


def test_legacy_and_refactored_executors_call_the_same_runtime_hook():
    repo_root = Path(__file__).resolve().parents[2]
    legacy = (repo_root / "rag" / "svr" / "task_executor.py").read_text(encoding="utf-8")
    refactored = (repo_root / "rag" / "svr" / "task_executor_refactor" / "task_handler.py").read_text(encoding="utf-8")

    for source in (legacy, refactored):
        assert "enqueue_tabular_structure_generation_if_complete" in source
        assert "run_tabular_structure_generation_job" in source
        assert "TaskService.get_tasks" in source
        assert '"progress": 1.0' in source
        assert "cancel_check=" in source


def test_task_executor_never_downgrades_to_the_legacy_mutation_path():
    repo_root = Path(__file__).resolve().parents[2]
    executor_path = repo_root / "rag" / "svr" / "task_executor.py"
    source = executor_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    handle_task = next(
        node
        for node in module.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_task"
    )
    method_source = ast.get_source_segment(source, handle_task)

    assert method_source is not None
    assert 'run_mode != "0"' in method_source
    assert "unsupported TE_RUN_MODE" in method_source
    assert "do_handle_task(task)" not in method_source
    assert method_source.count("TaskManager.run_refactored_task(") == 1


def test_ordinary_table_completion_reuses_the_already_loaded_source_for_generation_queue():
    repo_root = Path(runtime.__file__).parents[2]
    source = (repo_root / "rag" / "svr" / "task_executor.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    handle_task = next(
        node for node in module.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "do_handle_task"
    )
    method_source = ast.get_source_segment(source, handle_task)
    assert method_source is not None
    hook = method_source[method_source.index("enqueue_tabular_structure_generation_if_complete"):]
    assert "await get_storage_binary" not in hook
    assert "                    binary," in hook


def test_refactored_table_completion_reuses_the_already_loaded_source_for_generation_queue():
    repo_root = Path(runtime.__file__).parents[2]
    source = (repo_root / "rag" / "svr" / "task_executor_refactor" / "task_handler.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    handler = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TaskHandler"
    )
    method = next(
        node for node in handler.body
        if isinstance(node, ast.AsyncFunctionDef)
        and "enqueue_tabular_structure_generation_if_complete" in (ast.get_source_segment(source, node) or "")
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    hook = method_source[method_source.index("enqueue_tabular_structure_generation_if_complete"):]
    assert "self._get_storage_binary" not in hook
    assert "                    binary," in hook


def test_generation_task_is_not_recorded_as_a_parse_operation():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "rag" / "svr" / "task_executor.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    handle_task = next(
        node
        for node in module.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_task"
    )
    method_source = ast.get_source_segment(source, handle_task)

    assert method_source is not None
    assert 'task_type != "tabular_generation"' in method_source
    assert method_source.count("redis_msg.ack()") == 1
    assert method_source.rfind("redis_msg.ack()") > method_source.rfind("finally:")


def test_structure_only_build_publishes_from_source_without_parse_tasks():
    calls = []
    generation_ref = structure_generation_ref("document-1", b"workbook")

    class Service:
        class StructureSnapshotMissing(LookupError):
            pass

        active = False

        @classmethod
        def get_active_generation(cls, **_kwargs):
            if cls.active:
                return {"producer_generation_ref": generation_ref}
            raise cls.StructureSnapshotMissing()

        @classmethod
        def persist_shadow_generation(cls, storage, **kwargs):
            calls.append(("persist", storage, kwargs))
            return kwargs["projection_store"](
                storage,
                bucket=kwargs["dataset_id"],
                document_id=kwargs["document_id"],
                projection=kwargs["projection"],
                tenant_id=kwargs["tenant_id"],
            )

        @classmethod
        def activate_generation(cls, storage, **kwargs):
            calls.append(("activate", storage, kwargs))
            cls.active = True

    result = publish_tabular_structure_from_source(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        document_id="document-1",
        filename="anonymous.xlsx",
        binary=b"workbook",
        storage="projection-storage",
        service=Service,
        projection_builder=lambda *_args, **kwargs: {
            "rows": [{"row_ref_kwd": "row-1"}],
            "producer_generation_ref": kwargs["producer_generation_ref"],
        },
        projection_store=lambda *_args, **_kwargs: {
            "producer_generation_ref": generation_ref,
            "manifest_object_name": "manifest.json",
            "manifest_sha256": "a" * 64,
            "part_count": 1,
            "row_count": 1,
        },
    )

    assert result == {
        "status": "active",
        "producer_generation_ref": generation_ref,
        "row_count": 1,
    }
    assert [call[0] for call in calls] == ["persist", "activate"]
    assert calls[-1][2]["expected_active_generation_ref"] is None


def test_explicit_receipt_structure_build_passes_the_receipt_to_the_producer():
    converted = b"converted workbook"
    receipt = {
        "original_source_sha256": "1" * 64,
        "converted_source_sha256": hashlib.sha256(converted).hexdigest(),
        "converter_version": "converter-build/one",
    }
    generation_ref = structure_generation_ref(
        "document-1",
        converted,
        adr044_conversion_receipt=receipt,
    )
    builder_calls = []

    class Service:
        class StructureSnapshotMissing(LookupError):
            pass

        @staticmethod
        def get_active_generation(**_kwargs):
            raise Service.StructureSnapshotMissing()

        @staticmethod
        def persist_shadow_generation(storage, **kwargs):
            return kwargs["projection_store"](
                storage,
                bucket=kwargs["dataset_id"],
                document_id=kwargs["document_id"],
                projection=kwargs["projection"],
                tenant_id=kwargs["tenant_id"],
            )

        @staticmethod
        def activate_generation(*_args, **_kwargs):
            return None

    def build(_name, _binary, **kwargs):
        builder_calls.append(kwargs)
        return {"rows": [], "producer_generation_ref": kwargs["producer_generation_ref"]}

    result = publish_tabular_structure_from_source(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        document_id="document-1",
        filename="anonymous.xlsx",
        binary=converted,
        adr044_conversion_receipt=receipt,
        storage="storage",
        service=Service,
        projection_builder=build,
        projection_store=lambda *_args, **_kwargs: {
            "producer_generation_ref": generation_ref,
            "manifest_object_name": "manifest.json",
            "manifest_sha256": "a" * 64,
            "part_count": 1,
            "row_count": 0,
        },
    )

    assert result == {
        "status": "active",
        "producer_generation_ref": generation_ref,
        "row_count": 0,
    }
    assert builder_calls == [
        {
            "producer_generation_ref": generation_ref,
            "adr044_conversion_receipt": receipt,
        }
    ]
