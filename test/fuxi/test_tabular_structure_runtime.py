import uuid
from pathlib import Path

import rag.app.tabular_structure as tabular_structure
import rag.app.tabular_structure_runtime as runtime
from rag.app.tabular_structure_runtime import (
    is_complete_tabular_parse,
    publish_tabular_structure_from_source,
    publish_tabular_structure_generation,
    structure_generation_ref,
)


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
    monkeypatch.setattr(tabular_structure, "PROJECTION_VERSION", "tabular-structure-projection/v2")
    changed = structure_generation_ref("document-1", b"workbook")

    assert original != changed


def test_generation_ref_includes_the_multi_region_algorithm_version(monkeypatch):
    original = structure_generation_ref("document-1", b"workbook")
    monkeypatch.setattr(tabular_structure, "STRUCTURE_PRODUCER_ALGORITHM_VERSION", "region-producer/test-next")
    changed = structure_generation_ref("document-1", b"workbook")

    assert original != changed


def test_runtime_reexports_the_structure_producer_algorithm_version():
    assert runtime.STRUCTURE_PRODUCER_ALGORITHM_VERSION == (
        tabular_structure.STRUCTURE_PRODUCER_ALGORITHM_VERSION
    )


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
        def register_shadow_generation(storage, **kwargs):
            calls.append(("shadow", storage, kwargs["receipt"]))
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
        "build", "store", "shadow", "activate",
    ]


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
        def register_shadow_generation(*_args, **_kwargs):
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
        assert "publish_tabular_structure_generation" in source
        assert "TaskService.get_tasks" in source
        assert '"progress": 1.0' in source


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
        def register_shadow_generation(cls, storage, **kwargs):
            calls.append(("register", storage, kwargs))

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
    assert [call[0] for call in calls] == ["register", "activate"]
    assert calls[-1][2]["expected_active_generation_ref"] is None
