import ast
import asyncio
import copy
import hashlib
import importlib.util
import inspect
import json
import sys
import time
import unicodedata
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig, ObjectApiResponse

from rag.app import tabular_structure
from test.fuxi.test_table_semantic_rows import _load_table_module
from test.fuxi.test_tabular_structure_projection import _workbook_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_MODULE_PATH = REPO_ROOT / "api" / "db" / "services" / "tabular_structure_service.py"
CHUNK_API_PATH = REPO_ROOT / "api" / "apps" / "restful_apis" / "chunk_api.py"


def _load_authorized_document_source_name(document_service):
    module = ast.parse(CHUNK_API_PATH.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_authorized_document_source_name"
    )
    namespace = {
        "DocumentService": document_service,
        "unicodedata": unicodedata,
    }
    exec(
        compile(
            ast.Module(body=[helper], type_ignores=[]),
            str(CHUNK_API_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["_authorized_document_source_name"]


class _Storage:
    def __init__(self):
        self.objects = {}
        self.get_calls = []
        self.rm_calls = []

    def put(self, bucket, name, binary, tenant_id=None):
        self.objects[(bucket, name)] = bytes(binary)

    def get(self, bucket, name, tenant_id=None):
        self.get_calls.append((bucket, name, tenant_id))
        return self.objects.get((bucket, name))

    def obj_exist(self, bucket, name, tenant_id=None):
        return (bucket, name) in self.objects

    def rm(self, bucket, name, tenant_id=None):
        self.rm_calls.append((bucket, name, tenant_id))
        self.objects.pop((bucket, name), None)

    def rm_strict(self, bucket, name, tenant_id=None):
        return self.rm(bucket, name, tenant_id)


@pytest.fixture
def table_parser(monkeypatch):
    return _load_table_module(monkeypatch).Excel()


@pytest.fixture
def service_module():
    assert SERVICE_MODULE_PATH.is_file(), "Task 3 tabular structure service is missing"
    module_name = "fuxi_task3_tabular_structure_service"
    spec = importlib.util.spec_from_file_location(module_name, SERVICE_MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def generation_repository(service_module):
    repository = service_module.InMemoryTabularStructureRepository()
    repository.add_authorization_scope("tenant-owner", "dataset-1", "document-1")
    return repository


def _stored_generation(
    table_parser,
    *,
    generation_ref=None,
    rows_per_part=2,
    document_id="document-1",
):
    projection = tabular_structure.build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(include_note=False),
        producer_generation_ref=generation_ref or str(uuid.uuid4()),
        parser=table_parser,
    )
    storage = _Storage()
    receipt = tabular_structure.store_tabular_structure_projection(
        storage,
        bucket="dataset-1",
        document_id=document_id,
        projection=projection,
        rows_per_part=rows_per_part,
        tenant_id="tenant-owner",
    )
    return storage, projection, receipt


def _stored_generation_with_contract(
    table_parser,
    *,
    document_id: str,
    structure_algorithm_version: str,
    enumeration_rule_version: str,
):
    storage, projection, _receipt = _stored_generation(
        table_parser,
        document_id=document_id,
    )
    projection = copy.deepcopy(projection)
    projection["structure_algorithm_version"] = structure_algorithm_version
    projection["enumeration_rule_version"] = enumeration_rule_version
    projection["producer_generation_ref"] = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "fuxi:tabular-generation:"
            f"{projection['producer_schema_version']}:"
            f"{projection['version']}:"
            f"{structure_algorithm_version}:"
            f"{enumeration_rule_version}:"
            f"{document_id}:{projection['source_sha256']}",
        )
    )

    table_refs = {}
    for table in projection["tables"]:
        old_ref = table["table_ref"]
        membership_sha256 = old_ref.split("_", 3)[2]
        identity = tabular_structure._versioned_digest(
            "tabular-table/v2",
            projection["producer_schema_version"],
            projection["version"],
            structure_algorithm_version,
            enumeration_rule_version,
            projection["source_sha256"],
            table["sheet_ordinal"],
            table["table_ordinal"],
            membership_sha256,
        )
        new_ref = f"tbl_v2_{membership_sha256}_{identity}"
        table["table_ref"] = new_ref
        table_refs[old_ref] = new_ref
    for row in projection["rows"]:
        row["producer_generation_ref_kwd"] = projection[
            "producer_generation_ref"
        ]
        row["table_ref_kwd"] = table_refs[row["table_ref_kwd"]]
        row["row_ref_kwd"] = (
            f"{row['table_ref_kwd']}:{row['row_ordinal_int']}"
        )
        row["id"] = "tsr_v1_" + tabular_structure._versioned_digest(
            "tabular-row-record/v1",
            projection["producer_generation_ref"],
            row["row_ref_kwd"],
        )

    part = {
        "version": tabular_structure.PROJECTION_PART_VERSION,
        "producer_generation_ref": projection["producer_generation_ref"],
        "part_number": 1,
        "row_offset": 0,
        "row_count": len(projection["rows"]),
        "rows": projection["rows"],
    }
    part_payload = tabular_structure._canonical_json(part)
    part_sha256 = hashlib.sha256(part_payload).hexdigest()
    document_ref = tabular_structure._versioned_digest(
        "tabular-structure-document/v1",
        document_id,
    )
    prefix = (
        f"_fuxi/tabular-structure/v1/{document_ref}/"
        f"{projection['producer_generation_ref']}"
    )
    part_object_name = f"{prefix}/part-000001-{part_sha256}.json"
    manifest = {
        "version": projection["version"],
        "producer_schema_version": projection["producer_schema_version"],
        "producer_generation_ref": projection["producer_generation_ref"],
        "structure_algorithm_version": structure_algorithm_version,
        "enumeration_rule_version": enumeration_rule_version,
        "source_sha256": projection["source_sha256"],
        "row_count": len(projection["rows"]),
        "tables": projection["tables"],
        "parts": [
            {
                "part_number": 1,
                "object_name": part_object_name,
                "row_offset": 0,
                "row_count": len(projection["rows"]),
                "sha256": part_sha256,
            }
        ],
    }
    manifest_payload = tabular_structure._canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    manifest_object_name = f"{prefix}/manifest-{manifest_sha256}.json"
    storage = _Storage()
    storage.put("dataset-1", part_object_name, part_payload)
    storage.put("dataset-1", manifest_object_name, manifest_payload)
    receipt = {
        "producer_generation_ref": projection["producer_generation_ref"],
        "manifest_object_name": manifest_object_name,
        "manifest_sha256": manifest_sha256,
        "part_count": 1,
        "row_count": len(projection["rows"]),
    }
    return storage, projection, receipt


def _stored_generation_with_historical_delete_manifest(table_parser):
    storage, projection, receipt = _stored_generation(table_parser)
    old_manifest_name = receipt["manifest_object_name"]
    manifest = json.loads(storage.objects[("dataset-1", old_manifest_name)])
    del manifest["structure_algorithm_version"]
    del manifest["enumeration_rule_version"]
    manifest_payload = tabular_structure._canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    manifest_object_name = (
        old_manifest_name.rsplit("manifest-", 1)[0]
        + f"manifest-{manifest_sha256}.json"
    )
    del storage.objects[("dataset-1", old_manifest_name)]
    storage.put("dataset-1", manifest_object_name, manifest_payload)
    receipt.update(
        manifest_object_name=manifest_object_name,
        manifest_sha256=manifest_sha256,
    )
    return storage, projection, receipt


def _active_generation_record(projection, receipt, *, document_id: str):
    return {
        "producer_generation_ref": projection["producer_generation_ref"],
        "tenant_id": "tenant-owner",
        "kb_id": "dataset-1",
        "document_id": document_id,
        "projection_version": projection["version"],
        "producer_schema_version": projection["producer_schema_version"],
        "manifest_object_name": receipt["manifest_object_name"],
        "manifest_sha256": receipt["manifest_sha256"],
        "source_sha256": projection["source_sha256"],
        "row_count": len(projection["rows"]),
        "part_count": receipt["part_count"],
        "status": "active",
        "safe_error_code": None,
        "activated_at": None,
        "retained_at": None,
    }


def test_generation_model_has_document_scoped_control_fields():
    model_path = REPO_ROOT / "api" / "db" / "db_models.py"
    module = ast.parse(model_path.read_text(encoding="utf-8"))
    model = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "TabularStructureGeneration")
    assignments = {
        target.id: value
        for statement in model.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
        for value in [statement.value]
    }

    assert set(assignments) >= {
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
        "safe_error_code",
        "activated_at",
        "retained_at",
    }
    generation_call = assignments["producer_generation_ref"]
    assert isinstance(generation_call, ast.Call)
    assert any(keyword.arg == "primary_key" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in generation_call.keywords)
    manifest_call = assignments["manifest_object_name"]
    assert isinstance(manifest_call, ast.Call)
    assert not any(keyword.arg == "index" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in manifest_call.keywords)
    service_source = SERVICE_MODULE_PATH.read_text(encoding="utf-8")
    assert '"deleting"' in service_source


def test_structure_discovery_models_separate_dataset_state_from_table_recall_index():
    model_path = REPO_ROOT / "api" / "db" / "db_models.py"
    module = ast.parse(model_path.read_text(encoding="utf-8"))
    models = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.ClassDef)
    }

    state = models["TabularStructureDatasetIndexState"]
    table_index = models["TabularStructureTableIndex"]
    state_fields = {
        target.id
        for statement in state.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    table_fields = {
        target.id
        for statement in table_index.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }

    assert {
        "tenant_id",
        "kb_id",
        "index_revision",
        "backfill_status",
        "backfill_cursor",
        "index_schema_version",
    } <= state_fields
    assert {
        "tenant_id",
        "kb_id",
        "document_id",
        "producer_generation_ref",
        "table_ref",
        "table_ordinal",
        "search_text",
        "identity_hash",
        "index_revision",
        "active",
        "projection_status",
        "unsafe_reason",
    } <= table_fields


def test_peewee_activation_switches_generation_index_and_dataset_revision_in_one_transaction():
    module = ast.parse(SERVICE_MODULE_PATH.read_text(encoding="utf-8"))
    repository = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "PeeweeTabularStructureRepository")
    activate = next(node for node in repository.body if isinstance(node, ast.FunctionDef) and node.name == "activate")
    source = ast.get_source_segment(SERVICE_MODULE_PATH.read_text(encoding="utf-8"), activate)

    assert source is not None
    assert "with DB.atomic()" in source
    assert ".for_update()" in source
    assert "index_projection" in source
    assert "index_revision" in source
    assert "TabularStructureDatasetIndexState" in source
    assert "TabularStructureTableIndex" in source


def test_generation_activation_and_restore_reject_document_deletion_gate():
    source = SERVICE_MODULE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    repository = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PeeweeTabularStructureRepository"
    )
    for method_name in ("activate", "restore"):
        method = next(
            node
            for node in repository.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        method_source = ast.get_source_segment(source, method)

        assert method_source is not None
        assert ".for_update()" in method_source
        assert "TaskStatus.CANCEL.value" in method_source
        assert method_source.index("TaskStatus.CANCEL.value") < (
            method_source.index("Generation.update(")
        )


def test_both_generation_workers_use_bounded_retry_and_do_not_fail_ordinary_documents():
    for relative_path in (
        REPO_ROOT / "rag" / "svr" / "task_executor.py",
        REPO_ROOT / "rag" / "svr" / "task_executor_refactor" / "task_handler.py",
    ):
        source = relative_path.read_text(encoding="utf-8")
        assert "retry_tabular_structure_generation_task" in source
        assert "failure_code" in source
        assert "source_changed" in source
        assert 'task_type != "tabular_generation"' in source or "task_type == \"tabular_generation\"" in source


def test_both_generation_workers_claim_before_reading_source():
    for relative_path in (
        REPO_ROOT / "rag" / "svr" / "task_executor.py",
        REPO_ROOT / "rag" / "svr" / "task_executor_refactor" / "task_handler.py",
    ):
        source = relative_path.read_text(encoding="utf-8")
        claim_position = source.index("claim_tabular_structure_generation_attempt")
        source_read_position = source.index("get_storage_address", claim_position)
        assert claim_position < source_read_position


def test_generation_claim_unavailability_does_not_ack_the_redis_message():
    source = (REPO_ROOT / "rag" / "svr" / "task_executor.py").read_text(encoding="utf-8")
    assert "TabularGenerationClaimUnavailable" in source
    assert "ack_message = False" in source
    assert "if ack_message:" in source
    assert "set_progress(task_id, prog=-1" in source
    assert "if not generation_delivery_unavailable:" in source
    assert "TabularGenerationRetryUnavailable" in source
    assert "except TabularGenerationRetryUnavailable:" in source


def test_both_generation_workers_mark_complete_only_after_shadow_publication():
    runtime_source = (REPO_ROOT / "rag" / "app" / "tabular_structure_runtime.py").read_text(encoding="utf-8")
    assert "def _generation_progress" in runtime_source
    assert "min(float(progress), 0.999999)" in runtime_source
    for relative_path in (
        REPO_ROOT / "rag" / "svr" / "task_executor.py",
        REPO_ROOT / "rag" / "svr" / "task_executor_refactor" / "task_handler.py",
    ):
        source = relative_path.read_text(encoding="utf-8")
        assert "_generation_progress" in source
        assert 'result.get("status") in {"shadow", "active"}' in source
        assert "Tabular structure generation shadow ready." in source


def test_retry_claim_must_confirm_the_failed_to_queued_update():
    source = (REPO_ROOT / "api" / "db" / "services" / "task_service.py").read_text(encoding="utf-8")
    method = source[source.index("def fail_generation_attempt_and_reserve_retry"):]
    assert "updated = cls.model.update(" in method
    assert "if updated != 1:" in method
    assert "with DB.atomic()" in method
    assert "retryable" in method
    assert "if task is None:" in method
    assert "& (cls.model.progress == 0)" in method


def test_generation_retry_contract_has_hard_five_attempt_cap_and_stale_recovery():
    task_service_source = (REPO_ROOT / "api" / "db" / "services" / "task_service.py").read_text(encoding="utf-8")
    runtime_source = (REPO_ROOT / "rag" / "app" / "tabular_structure_runtime.py").read_text(encoding="utf-8")
    assert "TABULAR_GENERATION_MAX_ATTEMPTS = 5" in task_service_source
    assert "not 1 <= max_attempts <= TABULAR_GENERATION_MAX_ATTEMPTS" in task_service_source
    assert "not 1 <= max_attempts <= TABULAR_GENERATION_MAX_ATTEMPTS" in runtime_source
    assert "Recovering stale tabular structure generation attempt." in task_service_source


def test_peewee_repository_exposes_the_runtime_model_bundle(service_module, monkeypatch):
    class AnonymousModel:
        pass

    model_module = type(sys)("api.db.db_models")
    for name in (
        "DB",
        "Document",
        "Knowledgebase",
        "TabularStructureGeneration",
        "TabularStructureDatasetIndexState",
        "TabularStructureTableIndex",
    ):
        model = type(name, (AnonymousModel,), {})
        setattr(model_module, name, model)
    monkeypatch.setitem(sys.modules, "api.db.db_models", model_module)

    models = service_module.PeeweeTabularStructureRepository._models()

    assert isinstance(models, tuple)
    assert len(models) == 6
    assert [model.__name__ for model in models[1:]] == [
        "Document",
        "Knowledgebase",
        "TabularStructureGeneration",
        "TabularStructureDatasetIndexState",
        "TabularStructureTableIndex",
    ]


def test_document_generation_purge_removes_every_status_and_exact_scoped_projection(
    service_module,
    table_parser,
    monkeypatch,
    request,
):
    from peewee import (
        BooleanField,
        CharField,
        IntegerField,
        Model,
        ModelSelect,
        SqliteDatabase,
    )

    db_instance = SqliteDatabase(":memory:")
    request.addfinalizer(db_instance.close)

    class BaseModel(Model):
        class Meta:
            database = db_instance

    class Document(BaseModel):
        id = CharField(primary_key=True)
        kb_id = CharField()

    class Knowledgebase(BaseModel):
        id = CharField(primary_key=True)
        tenant_id = CharField()

    class Generation(BaseModel):
        producer_generation_ref = CharField(primary_key=True)
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        manifest_object_name = CharField()
        manifest_sha256 = CharField()
        part_count = IntegerField()
        status = CharField()

    class DatasetIndexState(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()
        index_revision = IntegerField()

    class TableIndex(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        producer_generation_ref = CharField()
        table_ref = CharField()
        index_revision = IntegerField()
        active = BooleanField(default=True)

    db_instance.create_tables(
        [
            Document,
            Knowledgebase,
            Generation,
            DatasetIndexState,
            TableIndex,
        ]
    )
    Knowledgebase.create(id="dataset-1", tenant_id="tenant-owner")
    Document.create(id="document-target", kb_id="dataset-1")
    Document.create(id="document-other", kb_id="dataset-1")
    monkeypatch.setattr(
        service_module.PeeweeTabularStructureRepository,
        "_models",
        staticmethod(
            lambda: (
                db_instance,
                Document,
                Knowledgebase,
                Generation,
                DatasetIndexState,
                TableIndex,
            )
        ),
    )
    monkeypatch.setattr(ModelSelect, "for_update", lambda query: query)

    target_storage = _Storage()
    target_generations = []
    for status in ("shadow", "active", "retained", "failed", "deleting"):
        storage, projection, receipt = _stored_generation(
            table_parser,
            document_id="document-target",
        )
        target_storage.objects.update(storage.objects)
        target_generations.append((status, projection, receipt))
        Generation.create(
            producer_generation_ref=projection["producer_generation_ref"],
            tenant_id="tenant-owner",
            kb_id="dataset-1",
            document_id="document-target",
            manifest_object_name=receipt["manifest_object_name"],
            manifest_sha256=receipt["manifest_sha256"],
            part_count=receipt["part_count"],
            status=status,
        )
        TableIndex.create(
            tenant_id="tenant-owner",
            kb_id="dataset-1",
            document_id="document-target",
            producer_generation_ref=projection["producer_generation_ref"],
            table_ref=f"table-{status}",
            index_revision=7,
        )

    other_storage, other_projection, other_receipt = _stored_generation(
        table_parser,
        document_id="document-other",
    )
    target_storage.objects.update(other_storage.objects)
    Generation.create(
        producer_generation_ref=other_projection["producer_generation_ref"],
        tenant_id="tenant-owner",
        kb_id="dataset-1",
        document_id="document-other",
        manifest_object_name=other_receipt["manifest_object_name"],
        manifest_sha256=other_receipt["manifest_sha256"],
        part_count=other_receipt["part_count"],
        status="active",
    )
    TableIndex.create(
        tenant_id="tenant-owner",
        kb_id="dataset-1",
        document_id="document-other",
        producer_generation_ref=other_projection["producer_generation_ref"],
        table_ref="table-other",
        index_revision=7,
    )
    DatasetIndexState.create(
        tenant_id="tenant-owner",
        kb_id="dataset-1",
        index_revision=7,
    )

    result = service_module.PeeweeTabularStructureRepository.purge_document_generations(
        target_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-target",
    )

    assert result == {
        "generation_count": 5,
        "table_index_count": 5,
        "object_count": 15,
        "index_revision": 8,
    }
    assert not Generation.select().where(
        Generation.document_id == "document-target"
    ).exists()
    assert not TableIndex.select().where(
        TableIndex.document_id == "document-target"
    ).exists()
    assert Generation.select().where(
        Generation.document_id == "document-other"
    ).count() == 1
    assert TableIndex.select().where(
        TableIndex.document_id == "document-other"
    ).count() == 1
    assert DatasetIndexState.get().index_revision == 8
    for _status, _projection, receipt in target_generations:
        manifest_delete_index = target_storage.rm_calls.index(
            ("dataset-1", receipt["manifest_object_name"], "tenant-owner")
        )
        projection_prefix = receipt["manifest_object_name"].rsplit("/", 1)[0] + "/"
        assert all(
            delete_index < manifest_delete_index
            for delete_index, (_bucket, object_name, _tenant_id) in enumerate(
                target_storage.rm_calls
            )
            if object_name.startswith(projection_prefix)
            and object_name != receipt["manifest_object_name"]
        )
        assert not any(
            bucket == "dataset-1"
            and (
                name == receipt["manifest_object_name"]
                or name.startswith(projection_prefix)
            )
            for bucket, name in target_storage.objects
        )
    assert all(
        key in target_storage.objects for key in other_storage.objects
    )


def test_document_generation_purge_is_noop_for_non_tabular_document(
    service_module,
    monkeypatch,
    request,
):
    from peewee import (
        BooleanField,
        CharField,
        IntegerField,
        Model,
        ModelSelect,
        SqliteDatabase,
    )

    db_instance = SqliteDatabase(":memory:")
    request.addfinalizer(db_instance.close)

    class BaseModel(Model):
        class Meta:
            database = db_instance

    class Document(BaseModel):
        id = CharField(primary_key=True)
        kb_id = CharField()

    class Knowledgebase(BaseModel):
        id = CharField(primary_key=True)
        tenant_id = CharField()

    class Generation(BaseModel):
        producer_generation_ref = CharField(primary_key=True)
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        status = CharField()

    class DatasetIndexState(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()
        index_revision = IntegerField()

    class TableIndex(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        producer_generation_ref = CharField()
        table_ref = CharField()
        index_revision = IntegerField()
        active = BooleanField(default=True)

    db_instance.create_tables(
        [
            Document,
            Knowledgebase,
            Generation,
            DatasetIndexState,
            TableIndex,
        ]
    )
    Knowledgebase.create(id="dataset-1", tenant_id="tenant-owner")
    Document.create(id="document-pdf", kb_id="dataset-1")
    DatasetIndexState.create(
        tenant_id="tenant-owner",
        kb_id="dataset-1",
        index_revision=11,
    )
    monkeypatch.setattr(
        service_module.PeeweeTabularStructureRepository,
        "_models",
        staticmethod(
            lambda: (
                db_instance,
                Document,
                Knowledgebase,
                Generation,
                DatasetIndexState,
                TableIndex,
            )
        ),
    )
    monkeypatch.setattr(ModelSelect, "for_update", lambda query: query)
    storage = MagicMock()

    result = service_module.PeeweeTabularStructureRepository.purge_document_generations(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-pdf",
    )

    assert result == {
        "generation_count": 0,
        "table_index_count": 0,
        "object_count": 0,
        "index_revision": 11,
    }
    assert DatasetIndexState.get().index_revision == 11
    assert storage.method_calls == []


def test_writing_generation_purge_uses_exact_prefix_without_manifest(
    service_module,
    monkeypatch,
    request,
):
    from peewee import (
        BooleanField,
        CharField,
        IntegerField,
        Model,
        ModelSelect,
        SqliteDatabase,
    )

    db_instance = SqliteDatabase(":memory:")
    request.addfinalizer(db_instance.close)

    class BaseModel(Model):
        class Meta:
            database = db_instance

    class Document(BaseModel):
        id = CharField(primary_key=True)
        kb_id = CharField()

    class Knowledgebase(BaseModel):
        id = CharField(primary_key=True)
        tenant_id = CharField()

    class Generation(BaseModel):
        producer_generation_ref = CharField(primary_key=True)
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        manifest_object_name = CharField()
        manifest_sha256 = CharField()
        part_count = IntegerField()
        status = CharField()

    class DatasetIndexState(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()
        index_revision = IntegerField()

    class TableIndex(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        producer_generation_ref = CharField()
        table_ref = CharField()
        index_revision = IntegerField()
        active = BooleanField(default=True)

    db_instance.create_tables(
        [
            Document,
            Knowledgebase,
            Generation,
            DatasetIndexState,
            TableIndex,
        ]
    )
    Knowledgebase.create(id="dataset-1", tenant_id="tenant-owner")
    Document.create(id="document-1", kb_id="dataset-1")
    generation_ref = "36b8f84d-df4e-4d49-b662-bcde71a8764f"
    document_ref = hashlib.sha256(
        b"tabular-structure-document/v1\x00document-1"
    ).hexdigest()
    prefix = (
        f"_fuxi/tabular-structure/v1/{document_ref}/{generation_ref}/"
    )
    Generation.create(
        producer_generation_ref=generation_ref,
        tenant_id="tenant-owner",
        kb_id="dataset-1",
        document_id="document-1",
        manifest_object_name=prefix,
        manifest_sha256="0" * 64,
        part_count=0,
        status="writing",
    )
    DatasetIndexState.create(
        tenant_id="tenant-owner",
        kb_id="dataset-1",
        index_revision=3,
    )
    monkeypatch.setattr(
        service_module.PeeweeTabularStructureRepository,
        "_models",
        staticmethod(
            lambda: (
                db_instance,
                Document,
                Knowledgebase,
                Generation,
                DatasetIndexState,
                TableIndex,
            )
        ),
    )
    monkeypatch.setattr(ModelSelect, "for_update", lambda query: query)
    calls = []

    class Storage:
        def rm_prefix_strict(self, bucket, object_prefix, tenant_id=None):
            calls.append((bucket, object_prefix, tenant_id))
            return 2

    result = service_module.PeeweeTabularStructureRepository.purge_document_generations(
        Storage(),
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
    )

    assert calls == [("dataset-1", prefix, "tenant-owner")]
    assert result == {
        "generation_count": 1,
        "table_index_count": 0,
        "object_count": 2,
        "index_revision": 4,
    }
    assert not Generation.select().exists()


def test_generation_purge_retries_interrupted_write_without_manifest_delete(
    service_module,
    monkeypatch,
    request,
):
    from peewee import (
        BooleanField,
        CharField,
        IntegerField,
        Model,
        ModelSelect,
        SqliteDatabase,
    )

    db_instance = SqliteDatabase(":memory:")
    request.addfinalizer(db_instance.close)

    class BaseModel(Model):
        class Meta:
            database = db_instance

    class Document(BaseModel):
        id = CharField(primary_key=True)
        kb_id = CharField()

    class Knowledgebase(BaseModel):
        id = CharField(primary_key=True)
        tenant_id = CharField()

    class Generation(BaseModel):
        producer_generation_ref = CharField(primary_key=True)
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        manifest_object_name = CharField()
        manifest_sha256 = CharField()
        part_count = IntegerField()
        status = CharField()

    class DatasetIndexState(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()
        index_revision = IntegerField()

    class TableIndex(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        producer_generation_ref = CharField()
        table_ref = CharField()
        index_revision = IntegerField()
        active = BooleanField(default=True)

    db_instance.create_tables(
        [
            Document,
            Knowledgebase,
            Generation,
            DatasetIndexState,
            TableIndex,
        ]
    )
    Knowledgebase.create(id="dataset-1", tenant_id="tenant-owner")
    Document.create(id="document-1", kb_id="dataset-1")
    generation_ref = "36b8f84d-df4e-4d49-b662-bcde71a8764f"
    prefix = tabular_structure.tabular_structure_projection_prefix(
        "document-1",
        generation_ref,
    )
    Generation.create(
        producer_generation_ref=generation_ref,
        tenant_id="tenant-owner",
        kb_id="dataset-1",
        document_id="document-1",
        manifest_object_name=prefix,
        manifest_sha256="0" * 64,
        part_count=0,
        status="parts_deleted",
    )
    DatasetIndexState.create(
        tenant_id="tenant-owner",
        kb_id="dataset-1",
        index_revision=3,
    )
    monkeypatch.setattr(
        service_module.PeeweeTabularStructureRepository,
        "_models",
        staticmethod(
            lambda: (
                db_instance,
                Document,
                Knowledgebase,
                Generation,
                DatasetIndexState,
                TableIndex,
            )
        ),
    )
    monkeypatch.setattr(ModelSelect, "for_update", lambda query: query)

    result = service_module.PeeweeTabularStructureRepository.purge_document_generations(
        SimpleNamespace(),
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
    )

    assert result == {
        "generation_count": 1,
        "table_index_count": 0,
        "object_count": 0,
        "index_revision": 4,
    }
    assert not Generation.select().exists()


def test_generation_write_intent_is_durable_before_object_store_mutation(
    service_module,
    table_parser,
    monkeypatch,
    request,
):
    from peewee import CharField, Model, ModelSelect, SqliteDatabase

    db_instance = SqliteDatabase(":memory:")
    request.addfinalizer(db_instance.close)

    class BaseModel(Model):
        class Meta:
            database = db_instance

    class Document(BaseModel):
        id = CharField(primary_key=True)
        kb_id = CharField()
        run = CharField()

    db_instance.create_tables([Document])
    Document.create(id="document-1", kb_id="dataset-1", run="1")
    monkeypatch.setattr(ModelSelect, "for_update", lambda query: query)
    fake_models = type(sys)("api.db.db_models")
    fake_models.DB = db_instance
    fake_models.Document = Document
    monkeypatch.setitem(sys.modules, "api.db.db_models", fake_models)
    constants_module = sys.modules["common.constants"]
    monkeypatch.setattr(
        constants_module,
        "TaskStatus",
        SimpleNamespace(CANCEL=SimpleNamespace(value="2")),
        raising=False,
    )
    projection = tabular_structure.build_tabular_structure_projection(
        "anonymous.xlsx",
        _workbook_bytes(include_note=False),
        producer_generation_ref="36b8f84d-df4e-4d49-b662-bcde71a8764f",
        parser=table_parser,
    )
    events = []

    class Repository:
        record = None

        @staticmethod
        def is_authorized(*_args):
            return True

        @classmethod
        def begin_write(cls, record):
            events.append("intent")
            cls.record = copy.deepcopy(record)
            return copy.deepcopy(record)

    def fail_store(*_args, **_kwargs):
        events.append("store")
        raise OSError("object store unavailable")

    with pytest.raises(OSError, match="object store unavailable"):
        service_module.TabularStructureService.persist_shadow_generation(
            SimpleNamespace(),
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
            projection=projection,
            projection_store=fail_store,
            repository=Repository,
        )

    assert events == ["intent", "store"]
    assert Repository.record is not None
    assert Repository.record["status"] == "writing"


def test_peewee_generation_write_intent_lifecycle(
    service_module,
    monkeypatch,
    request,
):
    from peewee import (
        BooleanField,
        CharField,
        DateTimeField,
        IntegerField,
        Model,
        ModelSelect,
        SqliteDatabase,
    )

    db_instance = SqliteDatabase(":memory:")
    request.addfinalizer(db_instance.close)

    class BaseModel(Model):
        class Meta:
            database = db_instance

    class Document(BaseModel):
        id = CharField(primary_key=True)
        kb_id = CharField()
        run = CharField()

    class Knowledgebase(BaseModel):
        id = CharField(primary_key=True)
        tenant_id = CharField()

    class Generation(BaseModel):
        producer_generation_ref = CharField(primary_key=True)
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        projection_version = CharField()
        producer_schema_version = CharField()
        manifest_object_name = CharField()
        manifest_sha256 = CharField()
        source_sha256 = CharField()
        row_count = IntegerField()
        part_count = IntegerField()
        status = CharField()
        safe_error_code = CharField(null=True)
        activated_at = DateTimeField(null=True)
        retained_at = DateTimeField(null=True)

        def to_dict(self):
            return dict(self.__data__)

    class DatasetIndexState(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()

    class TableIndex(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        active = BooleanField(default=True)

    db_instance.create_tables(
        [
            Document,
            Knowledgebase,
            Generation,
            DatasetIndexState,
            TableIndex,
        ]
    )
    Knowledgebase.create(id="dataset-1", tenant_id="tenant-owner")
    Document.create(id="document-1", kb_id="dataset-1", run="1")
    monkeypatch.setattr(
        service_module.PeeweeTabularStructureRepository,
        "_models",
        staticmethod(
            lambda: (
                db_instance,
                Document,
                Knowledgebase,
                Generation,
                DatasetIndexState,
                TableIndex,
            )
        ),
    )
    monkeypatch.setattr(ModelSelect, "for_update", lambda query: query)
    repository = service_module.PeeweeTabularStructureRepository()
    generation_ref = "36b8f84d-df4e-4d49-b662-bcde71a8764f"
    writing = {
        "producer_generation_ref": generation_ref,
        "tenant_id": "tenant-owner",
        "kb_id": "dataset-1",
        "document_id": "document-1",
        "projection_version": "projection/v1",
        "producer_schema_version": "producer/v1",
        "manifest_object_name": tabular_structure.tabular_structure_projection_prefix(
            "document-1",
            generation_ref,
        ),
        "manifest_sha256": "0" * 64,
        "source_sha256": "1" * 64,
        "row_count": 7,
        "part_count": 0,
        "status": "writing",
        "safe_error_code": None,
        "activated_at": None,
        "retained_at": None,
    }

    assert repository.begin_write(writing) == writing
    assert repository.begin_write(writing) == writing
    with pytest.raises(
        service_module.StructureGenerationConflict,
        match="generation identity already exists with different metadata",
    ):
        repository.begin_write({**writing, "source_sha256": "2" * 64})

    Document.update(run="2").where(Document.id == "document-1").execute()
    with pytest.raises(
        service_module.StructureSnapshotChanged,
        match="document is canceled for deletion",
    ):
        repository.begin_write(writing)
    Document.update(run="1").where(Document.id == "document-1").execute()

    shadow = {
        **writing,
        "manifest_object_name": (
            f"{writing['manifest_object_name']}manifest-{'3' * 64}.json"
        ),
        "manifest_sha256": "3" * 64,
        "part_count": 2,
        "status": "shadow",
    }
    assert repository.complete_write(shadow) == shadow
    assert repository.complete_write(shadow) == shadow
    assert repository.begin_write(writing) == shadow


def test_document_service_deletion_invokes_generation_purge_before_document_row_delete():
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    module = ast.parse(document_service_path.read_text(encoding="utf-8"))
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DocumentService"
    )
    deletion = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "delete_document_and_update_kb_counts"
    )
    source = ast.get_source_segment(
        document_service_path.read_text(encoding="utf-8"), deletion
    )

    assert source is not None
    assert "TaskStatus.CANCEL.value" in source
    assert "StatusEnum.INVALID.value" in source
    assert "purge_document_generations" in source
    assert "settings.STORAGE_IMPL" in source
    assert "cancel_all_task_of" in source
    assert "settings.docStoreConn.delete_strict(" in source
    assert "cls.delete_chunk_images(doc, tenant_id)" in source
    assert '{"doc_id": doc.id}' in source
    cancel_gate = source.index("run=TaskStatus.CANCEL.value")
    delete_intent = source.index("status=StatusEnum.INVALID.value")
    cancel_tasks = source.index("cancel_all_task_of(doc.id)")
    delete_images = source.index("cls.delete_chunk_images(doc, tenant_id)")
    strict_delete = source.index("settings.docStoreConn.delete_strict(")
    purge_generations = source.index(
        "PeeweeTabularStructureRepository.purge_document_generations("
    )
    delete_tasks = source.index("TaskService.filter_delete")
    delete_document = source.index("cls.model.delete()")
    assert cancel_gate < delete_intent < cancel_tasks < delete_images < strict_delete
    assert strict_delete < purge_generations < delete_tasks < delete_document
    assert "deactivate_document_index" not in source
    assert "Failed to delete chunk images" not in source


def test_document_delete_skips_chunk_store_only_after_strict_missing_index_probe():
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    source = document_service_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DocumentService"
    )
    remove_document = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef) and node.name == "remove_document"
    )
    delete_document = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "delete_document_and_update_kb_counts"
    )
    remove_source = ast.get_source_segment(source, remove_document)
    delete_source = ast.get_source_segment(source, delete_document)

    assert remove_source is not None
    assert delete_source is not None
    assert "chunk_index_exists=chunk_index_exists" not in remove_source
    assert "index_exist_strict" not in remove_source
    assert "index_exist_strict" in delete_source
    assert "if chunk_index_exists:" in delete_source
    delete_intent = delete_source.index("status=StatusEnum.INVALID.value")
    strict_probe = delete_source.index("index_exist_strict")
    guarded_start = delete_source.index("if chunk_index_exists:")
    assert delete_intent < strict_probe < guarded_start
    assert guarded_start < delete_source.index("cls.delete_chunk_images")
    assert guarded_start < delete_source.index("settings.docStoreConn.delete_strict")
    assert delete_source.index("settings.docStoreConn.delete_strict") < (
        delete_source.index("purge_document_generations")
    )


def test_document_deletion_owns_cleanup_only_after_durable_delete_intent(
    monkeypatch,
    request,
):
    from peewee import CharField, Model, SqliteDatabase

    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    db_instance = SqliteDatabase(":memory:")
    request.addfinalizer(db_instance.close)

    class BaseModel(Model):
        class Meta:
            database = db_instance

    class Document(BaseModel):
        id = CharField(primary_key=True)
        run = CharField()
        status = CharField()

    db_instance.create_tables([Document])
    Document.create(id="document-live", run="1", status="1")
    Document.create(id="document-stopped", run="2", status="1")
    Document.create(id="document-deleting", run="2", status="0")

    namespace = {
        "DB": db_instance,
        "TaskStatus": SimpleNamespace(CANCEL=SimpleNamespace(value="2")),
        "StatusEnum": SimpleNamespace(INVALID=SimpleNamespace(value="0")),
    }
    deletion_owns_cleanup = _load_class_method(
        document_service_path,
        "DocumentService",
        "document_deletion_owns_cleanup",
        namespace,
    )
    service = SimpleNamespace(model=Document)

    assert not deletion_owns_cleanup(service, "document-live")
    assert not deletion_owns_cleanup(service, "document-stopped")
    assert deletion_owns_cleanup(service, "document-deleting")
    assert deletion_owns_cleanup(service, "document-missing")


def test_task_cancellation_checks_the_persistent_document_gate():
    task_service_path = REPO_ROOT / "api" / "db" / "services" / "task_service.py"
    module = ast.parse(task_service_path.read_text(encoding="utf-8"))
    cancellation = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "has_canceled"
    )
    source = ast.get_source_segment(
        task_service_path.read_text(encoding="utf-8"), cancellation
    )

    assert source is not None
    assert "REDIS_CONN.get" in source
    assert "TaskService.do_cancel(task_id)" in source


def test_task_handler_cancellation_never_deletes_the_whole_document_scope():
    task_handler_path = (
        REPO_ROOT
        / "rag"
        / "svr"
        / "task_executor_refactor"
        / "task_handler.py"
    )
    source = task_handler_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    handler = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TaskHandler"
    )
    handle_task = next(
        node
        for node in handler.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_task"
    )
    handle_task_source = ast.get_source_segment(source, handle_task)

    assert handle_task_source is not None
    assert "rollback_task_writes" in handle_task_source
    assert "document_deletion_owns_cleanup" in handle_task_source
    assert handle_task_source.index("document_deletion_owns_cleanup") < (
        handle_task_source.index("rollback_task_writes")
    )
    assert "docStoreConn.delete" not in handle_task_source
    assert '{"doc_id": task_doc_id}' not in handle_task_source


@pytest.mark.asyncio
async def test_canceled_task_rollback_removes_images_before_exact_chunk_ids():
    chunk_service_path = (
        REPO_ROOT
        / "rag"
        / "svr"
        / "task_executor_refactor"
        / "chunk_service.py"
    )
    calls = []

    class Storage:
        def rm_strict(self, bucket, name, tenant_id=None):
            calls.append(("image", bucket, name, tenant_id))

        def obj_exist_strict(self, bucket, name, tenant_id=None):
            calls.append(("exists", bucket, name, tenant_id))
            return False

    class DocStore:
        def delete_strict(self, condition, index_name, dataset_id):
            calls.append(("chunks", condition, index_name, dataset_id))

    async def thread_pool_exec(operation, *args):
        return operation(*args)

    namespace = {
        "settings": SimpleNamespace(
            STORAGE_IMPL=Storage(),
            docStoreConn=DocStore(),
        ),
        "search": SimpleNamespace(index_name=lambda tenant_id: f"index-{tenant_id}"),
        "thread_pool_exec": thread_pool_exec,
    }
    rollback_task_writes = _load_class_method(
        chunk_service_path,
        "ChunkService",
        "rollback_task_writes",
        namespace,
    )
    service = SimpleNamespace(
        _task_context=SimpleNamespace(
            doc_id="document-1",
            tenant_id="tenant-1",
        ),
        _persisted_chunk_ids={"chunk-2", "chunk-1"},
        _persisted_image_ids={"chunk-2"},
    )

    await rollback_task_writes(
        service,
        "tenant-1",
        "dataset-1",
    )

    assert calls == [
        ("image", "dataset-1", "chunk-2", "tenant-1"),
        ("exists", "dataset-1", "chunk-2", "tenant-1"),
        (
            "chunks",
            {"id": ["chunk-1", "chunk-2"]},
            "index-tenant-1",
            "dataset-1",
        ),
    ]
    assert service._persisted_chunk_ids == set()
    assert service._persisted_image_ids == set()


def test_document_store_writes_share_the_document_deletion_gate():
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    source = document_service_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DocumentService"
    )
    guarded_write = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute_document_store_write"
    )
    guarded_source = ast.get_source_segment(source, guarded_write)

    assert guarded_source is not None
    assert ".for_update()" in guarded_source
    assert "TaskStatus.CANCEL.value" in guarded_source
    assert guarded_source.index(".for_update()") < guarded_source.index(
        "write_operation("
    )

    for relative_path in (
        "rag/svr/task_executor_refactor/chunk_service.py",
        "rag/svr/task_executor_refactor/chunk_post_processor.py",
        "rag/svr/task_executor_refactor/raptor_service.py",
        "rag/advanced_rag/knowlege_compile/structure.py",
        "rag/advanced_rag/knowlege_compile/dataset_nav.py",
    ):
        writer_source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "DocumentService.execute_document_store_write" in writer_source


def test_document_metadata_writes_share_the_document_deletion_gate():
    metadata_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "doc_metadata_service.py"
    )
    source = metadata_service_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DocMetadataService"
    )
    update_metadata = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "update_document_metadata"
    )
    method_source = ast.get_source_segment(source, update_metadata)

    assert method_source is not None
    assert "with DB.atomic():" in method_source
    assert ".for_update()" in method_source
    assert "TaskStatus.CANCEL.value" in method_source
    assert method_source.index(".for_update()") < method_source.index(
        "settings.docStoreConn"
    )


def test_document_store_write_gate_enforces_live_document_scope(monkeypatch, request):
    from peewee import CharField, Model, ModelSelect, SqliteDatabase

    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    module = ast.parse(document_service_path.read_text(encoding="utf-8"))
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DocumentService"
    )
    method = copy.deepcopy(
        next(
            node
            for node in service.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "execute_document_store_write"
        )
    )
    method.decorator_list = []

    db_instance = SqliteDatabase(":memory:")
    request.addfinalizer(db_instance.close)

    class BaseModel(Model):
        class Meta:
            database = db_instance

    class Document(BaseModel):
        id = CharField(primary_key=True)
        kb_id = CharField()
        run = CharField()

    db_instance.create_tables([Document])
    Document.create(id="document-live", kb_id="dataset-1", run="1")
    Document.create(id="document-cancel", kb_id="dataset-1", run="2")
    monkeypatch.setattr(ModelSelect, "for_update", lambda query: query)

    namespace = {
        "DB": db_instance,
        "TaskStatus": SimpleNamespace(CANCEL=SimpleNamespace(value="2")),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(document_service_path),
            "exec",
        ),
        namespace,
    )
    guarded_write = namespace["execute_document_store_write"]

    writes = []

    def write_operation(value):
        writes.append(value)
        return "written"

    assert (
        guarded_write(
            SimpleNamespace(model=Document),
            "document-live",
            "dataset-1",
            write_operation,
            "payload",
        )
        == "written"
    )
    assert writes == ["payload"]

    for document_ids, dataset_id, message in (
        ("document-cancel", "dataset-1", "canceled for deletion"),
        ("document-missing", "dataset-1", "unavailable"),
        ("document-live", "dataset-other", "scope rejected"),
    ):
        with pytest.raises((LookupError, PermissionError, RuntimeError), match=message):
            guarded_write(
                SimpleNamespace(model=Document),
                document_ids,
                dataset_id,
                write_operation,
                "late-payload",
            )
    assert writes == ["payload"]


def test_file_service_does_not_delete_tasks_before_document_lifecycle_cleanup():
    file_service_path = REPO_ROOT / "api" / "db" / "services" / "file_service.py"
    module = ast.parse(file_service_path.read_text(encoding="utf-8"))
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "FileService"
    )
    deletion = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef) and node.name == "delete_docs"
    )
    source = ast.get_source_segment(file_service_path.read_text(encoding="utf-8"), deletion)

    assert source is not None
    assert "DocumentService.remove_document" in source
    assert "TaskService.filter_delete" not in source


def test_non_image_document_chunk_cleanup_does_not_access_object_storage():
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    storage = MagicMock()
    search_results = iter(
        [
            {
                "hits": {
                    "hits": [
                        {"_id": "chunk-1", "_source": {"img_id": ""}},
                        {"_id": "chunk-2", "_source": {}},
                    ]
                }
            },
            {"hits": {"hits": []}},
        ]
    )
    doc_store = SimpleNamespace(
        search=lambda *_args: next(search_results),
        get_doc_ids=lambda result: [
            hit["_id"] for hit in result["hits"]["hits"]
        ],
        get_fields=lambda result, _fields: {
            hit["_id"]: hit.get("_source", {})
            for hit in result["hits"]["hits"]
        },
    )
    namespace = {
        "DB": SimpleNamespace(connection_context=lambda: lambda func: func),
        "OrderByExpr": object,
        "search": SimpleNamespace(index_name=lambda tenant_id: f"index-{tenant_id}"),
        "settings": SimpleNamespace(
            STORAGE_IMPL=storage,
            docStoreConn=doc_store,
        ),
    }
    delete_chunk_images = _load_class_method(
        document_service_path,
        "DocumentService",
        "delete_chunk_images",
        namespace,
    )

    delete_chunk_images(
        SimpleNamespace(),
        SimpleNamespace(id="document-1", kb_id="dataset-1"),
        "tenant-1",
    )

    storage.assert_not_called()
    assert storage.method_calls == []


def test_image_document_chunk_cleanup_uses_strict_storage_operations():
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    calls = []
    dataset_id = "90d4429c5e1f11f1ac9357bda2c79c71"

    class Storage:
        def obj_exist_strict(self, bucket, name, tenant_id=None):
            calls.append(("exists", bucket, name, tenant_id))
            return True

        def rm_strict(self, bucket, name, tenant_id=None):
            calls.append(("remove", bucket, name, tenant_id))
            raise OSError("image delete unavailable")

    search_results = iter(
        [
            {
                "hits": {
                    "hits": [
                        {
                            "_id": "chunk-image",
                            "_source": {
                                "img_id": f"{dataset_id}-chunk-image",
                            },
                        }
                    ]
                }
            },
            {"hits": {"hits": []}},
        ]
    )
    doc_store = SimpleNamespace(
        search=lambda *_args: next(search_results),
        get_doc_ids=lambda result: [
            hit["_id"] for hit in result["hits"]["hits"]
        ],
        get_fields=lambda result, _fields: {
            hit["_id"]: hit.get("_source", {})
            for hit in result["hits"]["hits"]
        },
    )
    namespace = {
        "DB": SimpleNamespace(connection_context=lambda: lambda func: func),
        "OrderByExpr": object,
        "search": SimpleNamespace(index_name=lambda tenant_id: f"index-{tenant_id}"),
        "settings": SimpleNamespace(
            STORAGE_IMPL=Storage(),
            docStoreConn=doc_store,
        ),
    }
    delete_chunk_images = _load_class_method(
        document_service_path,
        "DocumentService",
        "delete_chunk_images",
        namespace,
    )

    with pytest.raises(OSError, match="image delete unavailable"):
        delete_chunk_images(
            SimpleNamespace(),
            SimpleNamespace(id="document-1", kb_id=dataset_id),
            "tenant-1",
        )

    assert calls == [
        ("remove", dataset_id, "chunk-image", "tenant-1"),
    ]


@pytest.mark.asyncio
async def test_chunk_image_write_response_loss_rolls_back_before_es_insert():
    chunk_service_path = (
        REPO_ROOT
        / "rag"
        / "svr"
        / "task_executor_refactor"
        / "chunk_service.py"
    )
    calls = []

    class Storage:
        def __init__(self):
            self.objects = {}

        def obj_exist(self, bucket, name, tenant_id=None):
            return (bucket, name) in self.objects

        def obj_exist_strict(self, bucket, name, tenant_id=None):
            return self.obj_exist(bucket, name, tenant_id)

        def put(self, bucket, name, binary, tenant_id=None):
            self.objects[(bucket, name)] = bytes(binary)
            raise OSError("image write response lost")

        def rm_strict(self, bucket, name, tenant_id=None):
            calls.append(("rm_strict", bucket, name, tenant_id))
            self.objects.pop((bucket, name), None)

    class DocStore:
        def insert(self, *_args):
            calls.append(("es_insert",))

    class DocumentService:
        @staticmethod
        def execute_document_store_write(
            document_ids,
            dataset_id,
            write_operation,
            *args,
        ):
            calls.append(("document_gate", tuple(document_ids), dataset_id))
            return write_operation(*args)

    async def thread_pool_exec(operation, *args):
        return operation(*args)

    storage = Storage()
    namespace = {
        "Any": object,
        "DocumentService": DocumentService,
        "GRAPH_RAPTOR_FAKE_DOC_ID": "__dataset_raptor__",
        "settings": SimpleNamespace(
            STORAGE_IMPL=storage,
            docStoreConn=DocStore(),
        ),
        "thread_pool_exec": thread_pool_exec,
    }
    intercept_insert = _load_class_method(
        chunk_service_path,
        "ChunkService",
        "_intercept_doc_store_insert",
        namespace,
    )
    service = SimpleNamespace(
        _task_context=SimpleNamespace(
            write_interceptor=None,
            tenant_id="tenant-1",
            doc_id="document-1",
        ),
        _pending_images={"chunk-1": b"image-bytes"},
        _persisted_chunk_ids=set(),
        _persisted_image_ids=set(),
    )
    chunks = [{"id": "chunk-1", "doc_id": "document-1"}]

    with pytest.raises(OSError, match="image write response lost"):
        await intercept_insert(
            service,
            chunks,
            "ragflow_tenant_1",
            "dataset-1",
        )

    assert storage.objects == {}
    assert calls == [
        ("document_gate", ("document-1",), "dataset-1"),
        ("rm_strict", "dataset-1", "chunk-1", "tenant-1"),
    ]
    assert service._pending_images == {"chunk-1": b"image-bytes"}


@pytest.mark.parametrize(
    ("object_existed", "expected_remove_calls", "expected_object"),
    [
        (False, 1, None),
        (True, 0, b"image-bytes"),
    ],
)
@pytest.mark.asyncio
async def test_chunk_image_es_failure_only_rolls_back_new_objects(
    object_existed,
    expected_remove_calls,
    expected_object,
):
    chunk_service_path = (
        REPO_ROOT
        / "rag"
        / "svr"
        / "task_executor_refactor"
        / "chunk_service.py"
    )
    calls = []

    class Storage:
        def __init__(self):
            self.objects = {}
            if object_existed:
                self.objects[("dataset-1", "chunk-1")] = b"existing-image"

        def obj_exist_strict(self, bucket, name, tenant_id=None):
            return (bucket, name) in self.objects

        def put(self, bucket, name, binary, tenant_id=None):
            calls.append(("put", bucket, name, tenant_id))
            self.objects[(bucket, name)] = bytes(binary)

        def rm_strict(self, bucket, name, tenant_id=None):
            calls.append(("remove", bucket, name, tenant_id))
            self.objects.pop((bucket, name), None)

    class DocStore:
        def insert(self, *_args):
            calls.append(("es_insert",))
            return "es insert failed"

    class DocumentService:
        @staticmethod
        def execute_document_store_write(
            document_ids,
            dataset_id,
            write_operation,
            *args,
        ):
            calls.append(("document_gate", tuple(document_ids), dataset_id))
            return write_operation(*args)

    async def thread_pool_exec(operation, *args):
        return operation(*args)

    storage = Storage()
    namespace = {
        "Any": object,
        "DocumentService": DocumentService,
        "GRAPH_RAPTOR_FAKE_DOC_ID": "__dataset_raptor__",
        "settings": SimpleNamespace(
            STORAGE_IMPL=storage,
            docStoreConn=DocStore(),
        ),
        "thread_pool_exec": thread_pool_exec,
    }
    intercept_insert = _load_class_method(
        chunk_service_path,
        "ChunkService",
        "_intercept_doc_store_insert",
        namespace,
    )
    service = SimpleNamespace(
        _task_context=SimpleNamespace(
            write_interceptor=None,
            tenant_id="tenant-1",
            doc_id="document-1",
        ),
        _pending_images={"chunk-1": b"image-bytes"},
    )

    with pytest.raises(RuntimeError, match="es insert failed"):
        await intercept_insert(
            service,
            [{"id": "chunk-1", "doc_id": "document-1"}],
            "ragflow_tenant_1",
            "dataset-1",
        )

    assert storage.objects.get(("dataset-1", "chunk-1")) == expected_object
    assert sum(call[0] == "remove" for call in calls) == expected_remove_calls
    assert service._pending_images == {"chunk-1": b"image-bytes"}


@pytest.mark.asyncio
async def test_chunk_image_and_es_success_clear_pending_image():
    chunk_service_path = (
        REPO_ROOT
        / "rag"
        / "svr"
        / "task_executor_refactor"
        / "chunk_service.py"
    )

    class Storage:
        def __init__(self):
            self.objects = {}

        def obj_exist_strict(self, bucket, name, tenant_id=None):
            return (bucket, name) in self.objects

        def put(self, bucket, name, binary, tenant_id=None):
            self.objects[(bucket, name)] = bytes(binary)

        def rm_strict(self, bucket, name, tenant_id=None):
            self.objects.pop((bucket, name), None)

    class DocStore:
        @staticmethod
        def insert(*_args):
            return None

    class DocumentService:
        @staticmethod
        def execute_document_store_write(
            _document_ids,
            _dataset_id,
            write_operation,
            *args,
        ):
            return write_operation(*args)

    async def thread_pool_exec(operation, *args):
        return operation(*args)

    storage = Storage()
    namespace = {
        "Any": object,
        "DocumentService": DocumentService,
        "GRAPH_RAPTOR_FAKE_DOC_ID": "__dataset_raptor__",
        "settings": SimpleNamespace(
            STORAGE_IMPL=storage,
            docStoreConn=DocStore(),
        ),
        "thread_pool_exec": thread_pool_exec,
    }
    intercept_insert = _load_class_method(
        chunk_service_path,
        "ChunkService",
        "_intercept_doc_store_insert",
        namespace,
    )
    service = SimpleNamespace(
        _task_context=SimpleNamespace(
            write_interceptor=None,
            tenant_id="tenant-1",
            doc_id="document-1",
        ),
        _pending_images={"chunk-1": b"image-bytes"},
        _persisted_chunk_ids=set(),
        _persisted_image_ids=set(),
    )

    result = await intercept_insert(
        service,
        [{"id": "chunk-1", "doc_id": "document-1"}],
        "ragflow_tenant_1",
        "dataset-1",
    )

    assert result is None
    assert storage.objects == {("dataset-1", "chunk-1"): b"image-bytes"}
    assert service._pending_images == {}


@pytest.mark.asyncio
async def test_chunk_insert_without_pending_images_does_not_require_strict_storage():
    chunk_service_path = (
        REPO_ROOT
        / "rag"
        / "svr"
        / "task_executor_refactor"
        / "chunk_service.py"
    )
    calls = []

    class DocStore:
        def insert(self, chunks, index_name, dataset_id):
            calls.append(("es_insert", chunks, index_name, dataset_id))
            return None

    class DocumentService:
        @staticmethod
        def execute_document_store_write(
            document_ids,
            dataset_id,
            write_operation,
            *args,
        ):
            calls.append(("document_gate", tuple(document_ids), dataset_id))
            return write_operation(*args)

    async def thread_pool_exec(operation, *args):
        return operation(*args)

    namespace = {
        "Any": object,
        "DocumentService": DocumentService,
        "GRAPH_RAPTOR_FAKE_DOC_ID": "__dataset_raptor__",
        "settings": SimpleNamespace(
            STORAGE_IMPL=SimpleNamespace(),
            docStoreConn=DocStore(),
        ),
        "thread_pool_exec": thread_pool_exec,
    }
    intercept_insert = _load_class_method(
        chunk_service_path,
        "ChunkService",
        "_intercept_doc_store_insert",
        namespace,
    )
    chunks = [{"id": "chunk-1", "doc_id": "document-1"}]
    service = SimpleNamespace(
        _task_context=SimpleNamespace(
            write_interceptor=None,
            tenant_id="tenant-1",
            doc_id="document-1",
        ),
        _pending_images={},
        _persisted_chunk_ids=set(),
        _persisted_image_ids=set(),
    )

    result = await intercept_insert(
        service,
        chunks,
        "ragflow_tenant_1",
        "dataset-1",
    )

    assert result is None
    assert calls == [
        ("document_gate", ("document-1",), "dataset-1"),
        ("es_insert", chunks, "ragflow_tenant_1", "dataset-1"),
    ]


def test_minio_prefix_delete_is_exact_in_multi_bucket_mode():
    minio_path = REPO_ROOT / "rag" / "utils" / "minio_conn.py"
    storage_class = _load_class_subset(
        minio_path,
        "RAGFlowMinio",
        {"use_default_bucket", "use_prefix_path", "rm_prefix_strict"},
    )
    storage = object.__new__(storage_class)
    storage.bucket = None
    storage.prefix_path = None
    calls = []
    objects = [
        SimpleNamespace(object_name="_fuxi/document/generation/part-1"),
        SimpleNamespace(object_name="_fuxi/document/generation/manifest"),
    ]

    class Conn:
        def list_objects(self, bucket, *, prefix, recursive):
            calls.append(("list", bucket, prefix, recursive))
            return list(objects)

        def remove_object(self, bucket, object_name):
            calls.append(("remove", bucket, object_name))
            objects[:] = [
                obj for obj in objects if obj.object_name != object_name
            ]

    storage.conn = Conn()

    removed = storage.rm_prefix_strict(
        "dataset-1",
        "_fuxi/document/generation/",
        "tenant-1",
    )

    assert removed == 2
    assert calls == [
        ("list", "dataset-1", "_fuxi/document/generation/", True),
        (
            "remove",
            "dataset-1",
            "_fuxi/document/generation/part-1",
        ),
        (
            "remove",
            "dataset-1",
            "_fuxi/document/generation/manifest",
        ),
        ("list", "dataset-1", "_fuxi/document/generation/", True),
    ]


def test_minio_prefix_delete_preserves_default_bucket_dataset_scope():
    minio_path = REPO_ROOT / "rag" / "utils" / "minio_conn.py"
    storage_class = _load_class_subset(
        minio_path,
        "RAGFlowMinio",
        {"use_default_bucket", "use_prefix_path", "rm_prefix_strict"},
    )
    storage = object.__new__(storage_class)
    storage.bucket = "physical-bucket"
    storage.prefix_path = "ragflow"
    calls = []
    objects = [
        SimpleNamespace(
            object_name="ragflow/dataset-1/_fuxi/document/generation/part-1"
        )
    ]

    class Conn:
        def list_objects(self, bucket, *, prefix, recursive):
            calls.append(("list", bucket, prefix, recursive))
            return list(objects)

        def remove_object(self, bucket, object_name):
            calls.append(("remove", bucket, object_name))
            objects.clear()

    storage.conn = Conn()

    removed = storage.rm_prefix_strict(
        "dataset-1",
        "_fuxi/document/generation/",
        "tenant-1",
    )

    expected_prefix = "ragflow/dataset-1/_fuxi/document/generation/"
    assert removed == 1
    assert calls == [
        ("list", "physical-bucket", expected_prefix, True),
        ("remove", "physical-bucket", f"{expected_prefix}part-1"),
        ("list", "physical-bucket", expected_prefix, True),
    ]


def test_minio_prefix_delete_fails_when_objects_remain():
    minio_path = REPO_ROOT / "rag" / "utils" / "minio_conn.py"
    storage_class = _load_class_subset(
        minio_path,
        "RAGFlowMinio",
        {"use_default_bucket", "use_prefix_path", "rm_prefix_strict"},
    )
    storage = object.__new__(storage_class)
    storage.bucket = None
    storage.prefix_path = None
    objects = [SimpleNamespace(object_name="scope/part-1")]

    class Conn:
        @staticmethod
        def list_objects(_bucket, *, prefix, recursive):
            return list(objects)

        @staticmethod
        def remove_object(_bucket, _object_name):
            return None

    storage.conn = Conn()

    with pytest.raises(
        OSError,
        match="strict object prefix deletion was incomplete",
    ):
        storage.rm_prefix_strict("dataset-1", "scope/", "tenant-1")


def test_projection_object_delete_propagates_storage_failure(table_parser):
    class FailingStorage(_Storage):
        def rm_strict(self, bucket, name, tenant_id=None):
            raise OSError("object store unavailable")

    source_storage, projection, receipt = _stored_generation(table_parser)
    storage = FailingStorage()
    storage.objects.update(source_storage.objects)

    with pytest.raises(OSError, match="object store unavailable"):
        tabular_structure.delete_tabular_structure_projection_objects(
            storage,
            bucket="dataset-1",
            document_id="document-1",
            producer_generation_ref=projection["producer_generation_ref"],
            manifest_object_name=receipt["manifest_object_name"],
            manifest_sha256=receipt["manifest_sha256"],
            expected_part_count=receipt["part_count"],
            tenant_id="tenant-owner",
        )
    assert storage.obj_exist("dataset-1", receipt["manifest_object_name"])


def test_projection_object_delete_requires_strict_storage_capability(table_parser):
    source_storage, projection, receipt = _stored_generation(table_parser)
    source_storage.rm_strict = None

    with pytest.raises(RuntimeError, match="strict object deletion"):
        tabular_structure.delete_tabular_structure_projection_objects(
            source_storage,
            bucket="dataset-1",
            document_id="document-1",
            producer_generation_ref=projection["producer_generation_ref"],
            manifest_object_name=receipt["manifest_object_name"],
            manifest_sha256=receipt["manifest_sha256"],
            expected_part_count=receipt["part_count"],
            tenant_id="tenant-owner",
        )


def test_generation_purge_marks_scope_deleting_before_external_object_cleanup(
    service_module,
):
    module = ast.parse(SERVICE_MODULE_PATH.read_text(encoding="utf-8"))
    repository = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PeeweeTabularStructureRepository"
    )
    purge = next(
        node
        for node in repository.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "purge_document_generations"
    )
    source = ast.get_source_segment(
        SERVICE_MODULE_PATH.read_text(encoding="utf-8"), purge
    )

    assert source is not None
    assert 'status="deleting"' in source
    assert source.index('status="deleting"') < source.index(
        "delete_tabular_structure_projection_parts"
    )
    assert 'status="parts_deleted"' in source
    assert source.index("delete_tabular_structure_projection_parts") < source.index(
        'status="parts_deleted"'
    )
    assert source.index('status="parts_deleted"') < source.index(
        "delete_tabular_structure_projection_manifest"
    )
    assert source.index("delete_tabular_structure_projection_manifest") < source.index(
        "Generation.delete()"
    )


def test_generation_purge_retries_from_parts_deleted_without_manifest_inventory(
    service_module,
    table_parser,
    monkeypatch,
    request,
):
    from peewee import BooleanField, CharField, IntegerField, Model, ModelSelect, SqliteDatabase

    db_instance = SqliteDatabase(":memory:")
    request.addfinalizer(db_instance.close)

    class BaseModel(Model):
        class Meta:
            database = db_instance

    class Document(BaseModel):
        id = CharField(primary_key=True)
        kb_id = CharField()

    class Knowledgebase(BaseModel):
        id = CharField(primary_key=True)
        tenant_id = CharField()

    class Generation(BaseModel):
        producer_generation_ref = CharField(primary_key=True)
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        manifest_object_name = CharField()
        manifest_sha256 = CharField()
        part_count = IntegerField()
        status = CharField()

    class DatasetIndexState(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()
        index_revision = IntegerField()

    class TableIndex(BaseModel):
        tenant_id = CharField()
        kb_id = CharField()
        document_id = CharField()
        producer_generation_ref = CharField()
        table_ref = CharField()
        index_revision = IntegerField()
        active = BooleanField(default=True)

    db_instance.create_tables(
        [Document, Knowledgebase, Generation, DatasetIndexState, TableIndex]
    )
    Knowledgebase.create(id="dataset-1", tenant_id="tenant-owner")
    Document.create(id="document-1", kb_id="dataset-1")
    monkeypatch.setattr(
        service_module.PeeweeTabularStructureRepository,
        "_models",
        staticmethod(
            lambda: (
                db_instance,
                Document,
                Knowledgebase,
                Generation,
                DatasetIndexState,
                TableIndex,
            )
        ),
    )
    monkeypatch.setattr(ModelSelect, "for_update", lambda query: query)

    class ManifestFailureStorage(_Storage):
        fail_manifest = True

        def rm_strict(self, bucket, name, tenant_id=None):
            if self.fail_manifest and "/manifest-" in name:
                raise OSError("manifest storage unavailable")
            return super().rm_strict(bucket, name, tenant_id)

    source_storage, projection, receipt = _stored_generation(table_parser)
    storage = ManifestFailureStorage()
    storage.objects.update(source_storage.objects)
    Generation.create(
        producer_generation_ref=projection["producer_generation_ref"],
        tenant_id="tenant-owner",
        kb_id="dataset-1",
        document_id="document-1",
        manifest_object_name=receipt["manifest_object_name"],
        manifest_sha256=receipt["manifest_sha256"],
        part_count=receipt["part_count"],
        status="active",
    )
    TableIndex.create(
        tenant_id="tenant-owner",
        kb_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        table_ref="table-1",
        index_revision=3,
    )
    DatasetIndexState.create(
        tenant_id="tenant-owner",
        kb_id="dataset-1",
        index_revision=3,
    )

    with pytest.raises(OSError, match="manifest storage unavailable"):
        service_module.PeeweeTabularStructureRepository.purge_document_generations(
            storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
        )

    generation = Generation.get_by_id(projection["producer_generation_ref"])
    assert generation.status == "parts_deleted"
    assert storage.obj_exist("dataset-1", receipt["manifest_object_name"])
    assert all(
        not storage.obj_exist("dataset-1", part["object_name"])
        for part in json.loads(
            source_storage.get("dataset-1", receipt["manifest_object_name"])
        )["parts"]
    )
    assert TableIndex.select().where(TableIndex.document_id == "document-1").count() == 1

    storage.fail_manifest = False
    result = service_module.PeeweeTabularStructureRepository.purge_document_generations(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
    )

    assert result == {
        "generation_count": 1,
        "table_index_count": 1,
        "object_count": 1,
        "index_revision": 4,
    }
    assert not Generation.select().exists()
    assert not TableIndex.select().exists()
    assert not storage.obj_exist("dataset-1", receipt["manifest_object_name"])


def _load_class_method(source_path, class_name, method_name, namespace):
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    store_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in store_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    method = copy.deepcopy(method)
    method.decorator_list = []
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace[method_name]


def _load_class_subset(source_path, class_name, method_names):
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    source_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    selected = [
        copy.deepcopy(node)
        for node in source_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    loaded_class = ast.ClassDef(
        name=class_name,
        bases=[],
        keywords=[],
        body=selected,
        decorator_list=[],
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[loaded_class], type_ignores=[])
            ),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace[class_name]


def _load_function_subset(source_path, function_names, namespace=None):
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    selected = [
        copy.deepcopy(node)
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in function_names
    ]
    assert {node.name for node in selected} == set(function_names)
    namespace = dict(namespace or {})
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace


class _FakeQuery:
    def __init__(self, kind, **values):
        self.kind = kind
        self.values = values
        self.filter = []
        self.must = []
        self.must_not = []

    def to_dict(self):
        if self.kind == "bool":
            return {
                "bool": {
                    "filter": [value.to_dict() for value in self.filter],
                    "must": [value.to_dict() for value in self.must],
                    "must_not": [value.to_dict() for value in self.must_not],
                }
            }
        return {self.kind: self.values}


class _FakeSearch:
    def query(self, query):
        self._query = query
        return self

    def to_dict(self):
        return {"query": self._query.to_dict()}


def test_search_document_store_strict_delete_propagates_failures():
    class MissingError(Exception):
        pass

    class TimeoutError(Exception):
        pass

    namespace = {
        "ATTEMPT_TIME": 2,
        "ConnectionTimeout": TimeoutError,
        "NotFoundError": MissingError,
        "_response_value": lambda response, key, default=None: response.get(
            key, default
        ) if hasattr(response, "get") else default,
        "Q": _FakeQuery,
        "Search": _FakeSearch,
        "logger": MagicMock(),
    }
    delete_strict = _load_class_method(
        REPO_ROOT / "rag" / "utils" / "es_conn.py",
        "ESConnection",
        "delete_strict",
        namespace,
    )
    store = SimpleNamespace(logger=MagicMock(), _connect=MagicMock())
    client = MagicMock()
    store.es = client

    client.delete_by_query.return_value = {"deleted": 3}
    assert delete_strict(
        store, {"doc_id": "document-1"}, "tenant-index", "dataset-1"
    ) == 3
    request = client.delete_by_query.call_args.kwargs
    assert request["index"] == "tenant-index"
    assert '"doc_id": "document-1"' in json.dumps(request["body"])
    assert '"kb_id": "dataset-1"' in json.dumps(request["body"])

    for incomplete_result in (
        {"deleted": 2, "timed_out": True, "failures": []},
        {"deleted": 2, "timed_out": False, "failures": [{"cause": "blocked"}]},
        {"deleted": 2, "timed_out": False, "failures": [], "version_conflicts": 1},
        {"timed_out": False, "failures": [], "version_conflicts": 0},
        {"deleted": True, "timed_out": False, "failures": [], "version_conflicts": 0},
        {"deleted": -1, "timed_out": False, "failures": [], "version_conflicts": 0},
        {"deleted": 2, "total": 3, "timed_out": False, "failures": [], "version_conflicts": 0},
        {"deleted": 2, "total": 2, "noops": 1, "timed_out": False, "failures": [], "version_conflicts": 0},
    ):
        client.delete_by_query.return_value = incomplete_result
        with pytest.raises(RuntimeError, match="strict delete incomplete"):
            delete_strict(
                store, {"doc_id": "document-1"}, "tenant-index", "dataset-1"
            )

    client.delete_by_query.side_effect = MissingError("index missing")
    assert delete_strict(
        store, {"doc_id": "document-1"}, "tenant-index", "dataset-1"
    ) == 0

    client.delete_by_query.side_effect = TimeoutError("backend timeout")
    with pytest.raises(TimeoutError, match="backend timeout"):
        delete_strict(
            store, {"doc_id": "document-1"}, "tenant-index", "dataset-1"
        )

    client.delete_by_query.side_effect = RuntimeError("delete rejected")
    with pytest.raises(RuntimeError, match="delete rejected"):
        delete_strict(
            store, {"doc_id": "document-1"}, "tenant-index", "dataset-1"
        )


def test_search_document_store_strict_delete_accepts_elasticsearch_mapping_response():
    namespace = {
        "ATTEMPT_TIME": 2,
        "ConnectionTimeout": TimeoutError,
        "NotFoundError": KeyError,
        "_response_value": lambda response, key, default=None: response.get(
            key, default
        ),
        "Q": _FakeQuery,
        "Search": _FakeSearch,
        "logger": MagicMock(),
    }
    delete_strict = _load_class_method(
        REPO_ROOT / "rag" / "utils" / "es_conn.py",
        "ESConnection",
        "delete_strict",
        namespace,
    )
    store = SimpleNamespace(logger=MagicMock(), _connect=MagicMock())
    client = MagicMock()
    store.es = client
    client.delete_by_query.return_value = ObjectApiResponse(
        {
            "deleted": 3,
            "total": 3,
            "noops": 0,
            "timed_out": False,
            "failures": [],
            "version_conflicts": 0,
        },
        ApiResponseMeta(
            200,
            "1.1",
            HttpHeaders(),
            0.0,
            NodeConfig("http", "localhost", 9200),
        ),
    )

    assert delete_strict(
        store, {"doc_id": "document-1"}, "tenant-index", "dataset-1"
    ) == 3


def test_search_document_store_strict_index_exists_propagates_failures():
    index_exist_strict = _load_class_method(
        REPO_ROOT / "rag" / "utils" / "es_conn.py",
        "ESConnection",
        "index_exist_strict",
        {},
    )
    store = SimpleNamespace(
        es=SimpleNamespace(
            indices=SimpleNamespace(exists=lambda **kwargs: kwargs["index"] == "present")
        )
    )

    assert index_exist_strict(store, "present", "dataset-1") is True
    assert index_exist_strict(store, "missing", "dataset-1") is False

    def reject_exists(**_kwargs):
        raise RuntimeError("index probe failed")

    store.es.indices.exists = reject_exists
    with pytest.raises(RuntimeError, match="index probe failed"):
        index_exist_strict(store, "present", "dataset-1")


def test_strict_document_cleanup_uses_fail_closed_index_probe():
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    source = document_service_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DocumentService"
    )
    methods = {
        node.name: ast.get_source_segment(source, node)
        for node in service.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "delete_document_and_update_kb_counts",
            "remove_artifact_products",
        }
    }

    assert "index_exist_strict" in methods[
        "delete_document_and_update_kb_counts"
    ]
    assert "index_exist_strict" in methods["remove_artifact_products"]


def test_strict_document_thumbnail_delete_has_no_best_effort_preflight():
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    source = document_service_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DocumentService"
    )
    remove_document = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef) and node.name == "remove_document"
    )
    method_source = ast.get_source_segment(source, remove_document)

    assert method_source is not None
    assert "settings.STORAGE_IMPL.rm_strict" in method_source
    assert "settings.STORAGE_IMPL.obj_exist" not in method_source


def test_strict_metadata_delete_is_idempotent_without_preflight_get():
    metadata_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "doc_metadata_service.py"
    )
    source = metadata_service_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DocMetadataService"
    )
    deletion = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "delete_document_metadata"
    )
    method_source = ast.get_source_segment(source, deletion)

    assert method_source is not None
    strict_start = method_source.index("if strict:")
    strict_branch = method_source[strict_start:]
    assert "index_exist_strict" in strict_branch
    assert "deleted_count not in (0, 1)" in strict_branch
    assert strict_start < method_source.index("settings.docStoreConn.get(")


def test_document_delete_rejects_tenant_mismatch_before_external_cleanup():
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    source = document_service_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DocumentService"
    )
    remove_document = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef) and node.name == "remove_document"
    )
    method_source = ast.get_source_segment(source, remove_document)

    assert method_source is not None
    assert "expected_tenant_id" in method_source
    assert "document tenant scope mismatch" in method_source
    assert method_source.index("document tenant scope mismatch") < method_source.index(
        "delete_document_and_update_kb_counts"
    )


def test_dataset_nav_sync_delete_forwards_strict_mode():
    nav_path = REPO_ROOT / "rag" / "advanced_rag" / "knowlege_compile" / "dataset_nav.py"
    source = nav_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    wrapper = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "remove_dataset_nav_doc_sync"
    )
    method_source = ast.get_source_segment(source, wrapper)

    assert method_source is not None
    assert "strict=strict" in method_source
    assert "remove_dataset_nav_doc(" in method_source


@pytest.mark.asyncio
async def test_dataset_nav_delete_retry_keeps_doc_anchor_until_parent_detach(monkeypatch):
    document_id = "document-1"
    dataset_id = "dataset-1"
    module_path = (
        REPO_ROOT / "rag" / "advanced_rag" / "knowlege_compile" / "dataset_nav.py"
    )
    namespace = _load_function_subset(
        module_path,
        {
            "remove_dataset_nav_doc",
            "_cleanup_empty_cluster",
        },
        {
            "asyncio": asyncio,
            "json": json,
            "logging": __import__("logging"),
            "_nav_doc_id": lambda value: f"doc:{value}",
            "_nav_cluster_id": lambda kb_id, value: f"cluster:{kb_id}:{value}",
            "_nav_lock_key": lambda value: f"lock:{value}",
            "_COMPILE_KWD": "dataset_nav",
            "_LOCK_TIMEOUT_S": 30,
            "_LOCK_BLOCKING_TIMEOUT_S": 5,
        },
    )
    nav_doc_id = namespace["_nav_doc_id"](document_id)
    cluster_id = namespace["_nav_cluster_id"](dataset_id, "leaf")
    rows = {
        nav_doc_id: {
            "id": nav_doc_id,
            "doc_id": document_id,
            "parent_kwd": "leaf",
            "type_kwd": "nav_doc",
        },
        cluster_id: {
            "id": cluster_id,
            "name": "leaf",
            "parent_kwd": "root",
            "doc_ids_kwd": [document_id, "document-2"],
            "doc_count_int": 2,
            "type_kwd": "nav_cluster",
        },
    }
    fail_parent_detach = True

    class Lock:
        async def spin_acquire(self):
            return None

        def release(self):
            return None

    async def store_get(_tenant_id, _kb_id, row_id, **_kwargs):
        row = rows.get(row_id)
        return copy.deepcopy(row) if row else None

    async def store_upsert(_tenant_id, _kb_id, row, **_kwargs):
        nonlocal fail_parent_detach
        if row["id"] == cluster_id and fail_parent_detach:
            fail_parent_detach = False
            rows[row["id"]] = copy.deepcopy(row)
            raise OSError("parent update unavailable")
        rows[row["id"]] = copy.deepcopy(row)

    async def store_delete(_tenant_id, _kb_id, row_id, **_kwargs):
        rows.pop(row_id, None)

    async def store_search(_tenant_id, _kb_id, condition, _fields, **_kwargs):
        parent_name = (condition.get("parent_kwd") or [None])[0]
        return [
            copy.deepcopy(row)
            for row in rows.values()
            if row.get("parent_kwd") == parent_name
        ]

    namespace.update(
        {
            "RedisDistributedLock": lambda *_args, **_kwargs: Lock(),
            "_store_get": store_get,
            "_store_search": store_search,
            "_store_upsert": store_upsert,
            "_store_delete": store_delete,
        }
    )

    with pytest.raises(OSError, match="parent update unavailable"):
        await namespace["remove_dataset_nav_doc"](
            "tenant-1",
            dataset_id,
            document_id,
            strict=True,
        )

    assert nav_doc_id in rows
    assert rows[cluster_id]["doc_ids_kwd"] == ["document-2"]

    await namespace["remove_dataset_nav_doc"](
        "tenant-1",
        dataset_id,
        document_id,
        strict=True,
    )

    assert nav_doc_id not in rows
    assert rows[cluster_id]["doc_ids_kwd"] == ["document-2"]
    assert rows[cluster_id]["doc_count_int"] == 1


@pytest.mark.asyncio
async def test_dataset_nav_delete_retry_preserves_ancestor_path_until_root_cleanup(monkeypatch):
    document_id = "document-1"
    dataset_id = "dataset-1"
    module_path = (
        REPO_ROOT / "rag" / "advanced_rag" / "knowlege_compile" / "dataset_nav.py"
    )
    namespace = _load_function_subset(
        module_path,
        {
            "remove_dataset_nav_doc",
            "_cleanup_empty_cluster",
        },
        {
            "asyncio": asyncio,
            "json": json,
            "logging": __import__("logging"),
            "_nav_doc_id": lambda value: f"doc:{value}",
            "_nav_cluster_id": lambda kb_id, value: f"cluster:{kb_id}:{value}",
            "_nav_lock_key": lambda value: f"lock:{value}",
            "_COMPILE_KWD": "dataset_nav",
            "_LOCK_TIMEOUT_S": 30,
            "_LOCK_BLOCKING_TIMEOUT_S": 5,
        },
    )
    nav_doc_id = namespace["_nav_doc_id"](document_id)
    leaf_id = namespace["_nav_cluster_id"](dataset_id, "leaf")
    parent_id = namespace["_nav_cluster_id"](dataset_id, "parent")
    rows = {
        nav_doc_id: {
            "id": nav_doc_id,
            "doc_id": document_id,
            "parent_kwd": "leaf",
            "type_kwd": "nav_doc",
        },
        leaf_id: {
            "id": leaf_id,
            "name": "leaf",
            "parent_kwd": "parent",
            "doc_ids_kwd": [document_id],
            "doc_count_int": 1,
            "type_kwd": "nav_cluster",
        },
        parent_id: {
            "id": parent_id,
            "name": "parent",
            "parent_kwd": "root",
            "doc_ids_kwd": [],
            "doc_count_int": 0,
            "type_kwd": "nav_cluster",
        },
    }
    fail_root_cleanup = True

    class Lock:
        async def spin_acquire(self):
            return None

        def release(self):
            return None

    async def store_get(_tenant_id, _kb_id, row_id, **_kwargs):
        row = rows.get(row_id)
        return copy.deepcopy(row) if row else None

    async def store_search(_tenant_id, _kb_id, condition, _fields, **_kwargs):
        parent_name = (condition.get("parent_kwd") or [None])[0]
        return [
            copy.deepcopy(row)
            for row in rows.values()
            if row.get("parent_kwd") == parent_name
        ]

    async def store_upsert(_tenant_id, _kb_id, row, **_kwargs):
        rows[row["id"]] = copy.deepcopy(row)

    async def store_delete(_tenant_id, _kb_id, row_id, **_kwargs):
        nonlocal fail_root_cleanup
        if row_id == parent_id and fail_root_cleanup:
            fail_root_cleanup = False
            rows.pop(row_id, None)
            raise OSError("ancestor delete unavailable")
        rows.pop(row_id, None)

    namespace.update(
        {
            "RedisDistributedLock": lambda *_args, **_kwargs: Lock(),
            "_store_get": store_get,
            "_store_search": store_search,
            "_store_upsert": store_upsert,
            "_store_delete": store_delete,
        }
    )

    with pytest.raises(OSError, match="ancestor delete unavailable"):
        await namespace["remove_dataset_nav_doc"](
            "tenant-1",
            dataset_id,
            document_id,
            strict=True,
        )

    assert nav_doc_id in rows
    assert leaf_id not in rows
    assert parent_id not in rows

    await namespace["remove_dataset_nav_doc"](
        "tenant-1",
        dataset_id,
        document_id,
        strict=True,
    )

    assert rows == {}


def test_knowledge_graph_cleanup_retry_retains_graph_source_anchor(monkeypatch):
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    rows = {
        "graph": {
            "id": "graph",
            "knowledge_graph_kwd": "graph",
            "source_id": ["document-1", "document-2"],
            "removed_kwd": "N",
        },
        "entity-owned": {
            "id": "entity-owned",
            "knowledge_graph_kwd": "entity",
            "source_id": ["document-1"],
        },
        "entity-shared": {
            "id": "entity-shared",
            "knowledge_graph_kwd": "entity",
            "source_id": ["document-1", "document-2"],
        },
    }
    fail_orphan_delete = True

    def matches(row, condition):
        for key, value in condition.items():
            if key == "kb_id":
                continue
            if key == "must_not":
                if value == {"exists": "source_id"} and row.get("source_id"):
                    return False
                continue
            actual = row.get(key)
            expected = value if isinstance(value, list) else [value]
            if isinstance(actual, list):
                if not any(item in actual for item in expected):
                    return False
            elif actual not in expected:
                return False
        return True

    class DocStore:
        def search(
            self,
            fields,
            _highlight,
            condition,
            _match,
            _order,
            _offset,
            _limit,
            _index,
            _datasets,
        ):
            return {
                "rows": {
                    row_id: {
                        field: copy.deepcopy(row.get(field))
                        for field in fields
                        if row.get(field) is not None
                    }
                    for row_id, row in rows.items()
                    if matches(row, condition)
                }
            }

        @staticmethod
        def get_fields(result, _fields):
            return result["rows"]

        def update(self, condition, new_value, _index, _dataset):
            for row in rows.values():
                if not matches(row, condition):
                    continue
                for field, value in new_value.items():
                    if field == "remove":
                        for target_field, target_value in value.items():
                            values = row.get(target_field) or []
                            row[target_field] = [
                                item for item in values if item != target_value
                            ]
                    else:
                        row[field] = value
            return True

        def delete_strict(self, condition, _index, _dataset):
            nonlocal fail_orphan_delete
            if (
                condition.get("id") == ["entity-owned"]
                and fail_orphan_delete
                and any(matches(row, condition) for row in rows.values())
            ):
                fail_orphan_delete = False
                raise OSError("orphan delete unavailable")
            deleted = [
                row_id for row_id, row in rows.items() if matches(row, condition)
            ]
            for row_id in deleted:
                del rows[row_id]
            return len(deleted)

    cleanup = _load_class_method(
        document_service_path,
        "DocumentService",
        "cleanup_knowledge_graph_products",
        {
            "settings": SimpleNamespace(docStoreConn=DocStore()),
            "search": SimpleNamespace(index_name=lambda tenant_id: f"index-{tenant_id}"),
            "OrderByExpr": object,
        },
    )
    document = SimpleNamespace(id="document-1", kb_id="dataset-1")

    with pytest.raises(OSError, match="orphan delete unavailable"):
        cleanup(SimpleNamespace(), document, "tenant-1")

    assert "document-1" in rows["graph"]["source_id"]

    cleanup(SimpleNamespace(), document, "tenant-1")

    assert "entity-owned" not in rows
    assert rows["entity-shared"]["source_id"] == ["document-2"]
    assert rows["graph"]["source_id"] == ["document-2"]
    assert rows["graph"]["removed_kwd"] == "Y"


def test_knowledge_graph_cleanup_deletes_sole_owner_graph_row():
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    rows = {
        "graph": {
            "id": "graph",
            "knowledge_graph_kwd": "graph",
            "source_id": ["document-1"],
            "removed_kwd": "N",
        },
        "entity-owned": {
            "id": "entity-owned",
            "knowledge_graph_kwd": "entity",
            "source_id": ["document-1"],
        },
    }

    def matches(row, condition):
        for key, value in condition.items():
            if key == "kb_id":
                continue
            actual = row.get(key)
            expected = value if isinstance(value, list) else [value]
            if isinstance(actual, list):
                if not any(item in actual for item in expected):
                    return False
            elif actual not in expected:
                return False
        return True

    class DocStore:
        def search(
            self,
            fields,
            _highlight,
            condition,
            _match,
            _order,
            _offset,
            _limit,
            _index,
            _datasets,
        ):
            return {
                "rows": {
                    row_id: {
                        field: copy.deepcopy(row.get(field))
                        for field in fields
                        if row.get(field) is not None
                    }
                    for row_id, row in rows.items()
                    if matches(row, condition)
                }
            }

        @staticmethod
        def get_fields(result, _fields):
            return result["rows"]

        def update(self, condition, new_value, _index, _dataset):
            for row in rows.values():
                if not matches(row, condition):
                    continue
                for field, value in new_value.items():
                    if field == "remove":
                        for target_field, target_value in value.items():
                            row[target_field] = [
                                item
                                for item in (row.get(target_field) or [])
                                if item != target_value
                            ]
                    else:
                        row[field] = value
            return True

        def delete_strict(self, condition, _index, _dataset):
            deleted = [
                row_id for row_id, row in rows.items() if matches(row, condition)
            ]
            for row_id in deleted:
                del rows[row_id]
            return len(deleted)

    cleanup = _load_class_method(
        document_service_path,
        "DocumentService",
        "cleanup_knowledge_graph_products",
        {
            "settings": SimpleNamespace(docStoreConn=DocStore()),
            "search": SimpleNamespace(index_name=lambda tenant_id: f"index-{tenant_id}"),
            "OrderByExpr": object,
        },
    )

    cleanup(
        SimpleNamespace(),
        SimpleNamespace(id="document-1", kb_id="dataset-1"),
        "tenant-1",
    )

    assert rows == {}


def test_artifact_shared_owner_cleanup_requires_readback_confirmation():
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    rows = {
        "shared": {
            "id": "shared",
            "compile_kwd": "artifact_entity",
            "source_doc_ids": ["document-1", "document-2"],
        }
    }

    class DocStore:
        @staticmethod
        def index_exist(_index, _dataset):
            return True

        @staticmethod
        def index_exist_strict(_index, _dataset):
            return True

        def search(
            self,
            fields,
            _highlight,
            condition,
            _match,
            _order,
            _offset,
            _limit,
            _index,
            _datasets,
        ):
            matched = {}
            for row_id, row in rows.items():
                compile_kwds = condition.get("compile_kwd") or []
                if compile_kwds and row.get("compile_kwd") not in compile_kwds:
                    continue
                owner = condition.get("source_doc_ids")
                owners = owner if isinstance(owner, list) else [owner]
                if owner and not any(
                    value in (row.get("source_doc_ids") or [])
                    for value in owners
                ):
                    continue
                matched[row_id] = {
                    field: copy.deepcopy(row.get(field))
                    for field in fields
                    if row.get(field) is not None
                }
            return {"rows": matched}

        @staticmethod
        def get_fields(result, _fields):
            return result["rows"]

        @staticmethod
        def delete_strict(_condition, _index, _dataset):
            return 0

        @staticmethod
        def update(_condition, _new_value, _index, _dataset):
            return True

    fake_wiki = type(sys)(
        "rag.svr.task_executor_refactor.dataset_wiki_generator"
    )
    fake_wiki.WIKI_MAP_COMPILE_KWD = "artifact_map_extract"
    fake_wiki.WIKI_DERIVED_COMPILE_KWDS = {"artifact_entity"}
    previous_module = sys.modules.get(fake_wiki.__name__)
    sys.modules[fake_wiki.__name__] = fake_wiki
    try:
        cleanup = _load_class_method(
            document_service_path,
            "DocumentService",
            "remove_artifact_products",
            {
                "settings": SimpleNamespace(docStoreConn=DocStore()),
                "search": SimpleNamespace(
                    index_name=lambda tenant_id: f"index-{tenant_id}"
                ),
                "OrderByExpr": object,
            },
        )

        with pytest.raises(
            RuntimeError,
            match="strict artifact ownership cleanup failed",
        ):
            cleanup(
                SimpleNamespace(),
                SimpleNamespace(id="document-1", kb_id="dataset-1"),
                "tenant-1",
                strict=True,
            )
    finally:
        if previous_module is None:
            sys.modules.pop(fake_wiki.__name__, None)
        else:
            sys.modules[fake_wiki.__name__] = previous_module


def test_file_service_strictly_removes_only_exclusive_knowledgebase_original_before_document_row_delete():
    file_service_path = REPO_ROOT / "api" / "db" / "services" / "file_service.py"
    source = file_service_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "FileService"
    )
    deletion = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef) and node.name == "delete_docs"
    )
    method_source = ast.get_source_segment(source, deletion)

    assert method_source is not None
    assert "owns_original = False" in method_source
    assert "owns_original = not document_links" not in method_source
    assert "FileSource.KNOWLEDGEBASE" in method_source
    assert "File2DocumentService.get_by_file_id" in method_source
    assert "len(file_links) == 1" in method_source
    assert "settings.STORAGE_IMPL.rm_strict" in method_source
    strict_original_delete = method_source.index("settings.STORAGE_IMPL.rm_strict")
    document_delete = method_source.index("DocumentService.remove_document")
    assert strict_original_delete < document_delete
    assert "strict_cleanup=delete_exclusive_original" in method_source
    assert "settings.STORAGE_IMPL.rm(b, n)" not in method_source


def test_document_row_delete_waits_for_post_chunk_strict_cleanup_hook():
    document_service_path = (
        REPO_ROOT / "api" / "db" / "services" / "document_service.py"
    )
    source = document_service_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DocumentService"
    )
    deletion = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "delete_document_and_update_kb_counts"
    )
    method_source = ast.get_source_segment(source, deletion)

    assert method_source is not None
    assert "strict_cleanup" in method_source
    cleanup_hook = method_source.index("strict_cleanup(chunk_index_exists)")
    delete_tasks = method_source.index("TaskService.filter_delete")
    delete_document = method_source.index("cls.model.delete()")
    assert cleanup_hook < delete_tasks < delete_document
def test_projection_object_inventory_is_manifest_bound(service_module, table_parser):
    storage, projection, receipt = _stored_generation(table_parser)

    inventory = tabular_structure.list_tabular_structure_projection_objects(
        storage,
        bucket="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        manifest_object_name=receipt["manifest_object_name"],
        manifest_sha256=receipt["manifest_sha256"],
        expected_part_count=receipt["part_count"],
        tenant_id="tenant-owner",
    )

    assert len(inventory["part_object_names"]) == receipt["part_count"]
    assert inventory["manifest_object_name"] == receipt["manifest_object_name"]
    assert all(
        name.startswith(
            receipt["manifest_object_name"].rsplit("/", 1)[0] + "/part-"
        )
        for name in inventory["part_object_names"]
    )
    with pytest.raises(service_module.StructureSnapshotChanged, match="document scope"):
        tabular_structure.list_tabular_structure_projection_objects(
            storage,
            bucket="dataset-1",
            document_id="document-other",
            producer_generation_ref=projection["producer_generation_ref"],
            manifest_object_name=receipt["manifest_object_name"],
            manifest_sha256=receipt["manifest_sha256"],
            expected_part_count=receipt["part_count"],
            tenant_id="tenant-owner",
        )


def test_projection_object_inventory_accepts_historical_manifest_for_delete_only(
    service_module,
    table_parser,
):
    storage, projection, receipt = _stored_generation_with_historical_delete_manifest(
        table_parser
    )

    inventory = tabular_structure.list_tabular_structure_projection_objects(
        storage,
        bucket="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        manifest_object_name=receipt["manifest_object_name"],
        manifest_sha256=receipt["manifest_sha256"],
        expected_part_count=receipt["part_count"],
        tenant_id="tenant-owner",
    )

    assert inventory["object_names"] == [
        *inventory["part_object_names"],
        receipt["manifest_object_name"],
    ]
    assert tabular_structure.delete_tabular_structure_projection_objects(
        storage,
        bucket="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        manifest_object_name=receipt["manifest_object_name"],
        manifest_sha256=receipt["manifest_sha256"],
        expected_part_count=receipt["part_count"],
        tenant_id="tenant-owner",
    ) == receipt["part_count"] + 1
    assert storage.objects == {}


def test_current_projection_reader_rejects_historical_delete_manifest(
    service_module,
    table_parser,
):
    storage, projection, receipt = _stored_generation_with_historical_delete_manifest(
        table_parser
    )

    with pytest.raises(service_module.StructureSnapshotChanged, match="manifest schema changed"):
        tabular_structure.load_tabular_structure_projection(
            storage,
            bucket="dataset-1",
            document_id="document-1",
            producer_generation_ref=projection["producer_generation_ref"],
            manifest_object_name=receipt["manifest_object_name"],
            manifest_sha256=receipt["manifest_sha256"],
            expected_part_count=receipt["part_count"],
            tenant_id="tenant-owner",
        )


def test_projection_object_inventory_rejects_unknown_historical_manifest_schema(
    service_module,
    table_parser,
):
    storage, projection, receipt = _stored_generation_with_historical_delete_manifest(
        table_parser
    )
    old_manifest_name = receipt["manifest_object_name"]
    manifest = json.loads(storage.objects[("dataset-1", old_manifest_name)])
    manifest["unexpected_contract_field"] = "unknown"
    manifest_payload = tabular_structure._canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    manifest_object_name = (
        old_manifest_name.rsplit("manifest-", 1)[0]
        + f"manifest-{manifest_sha256}.json"
    )
    storage.put("dataset-1", manifest_object_name, manifest_payload)

    with pytest.raises(service_module.StructureSnapshotChanged, match="manifest schema changed"):
        tabular_structure.list_tabular_structure_projection_objects(
            storage,
            bucket="dataset-1",
            document_id="document-1",
            producer_generation_ref=projection["producer_generation_ref"],
            manifest_object_name=manifest_object_name,
            manifest_sha256=manifest_sha256,
            expected_part_count=receipt["part_count"],
            tenant_id="tenant-owner",
        )


def test_peewee_discovery_sql_binds_every_placeholder_in_statement_order(
    service_module,
    monkeypatch,
):
    class Field:
        def __eq__(self, other):
            return ("eq", other)

    class State:
        tenant_id = Field()
        kb_id = Field()
        index_revision = 9
        backfill_status = "complete"

        @classmethod
        def get_or_none(cls, *conditions):
            return cls()

    class Cursor:
        def __init__(self, row=None, rows=None):
            self.row = row
            self.rows = rows or []

        def fetchone(self):
            return self.row

        def fetchall(self):
            return self.rows

    class Database:
        def __init__(self):
            self.calls = []

        def atomic(self):
            class Atomic:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

            return Atomic()

        def execute_sql(self, sql, params):
            self.calls.append((sql, params))
            assert sql.count("%s") == len(params)
            if sql.startswith("SELECT 1"):
                return Cursor(row=None)
            return Cursor(
                rows=[
                    (
                        "document-a",
                        "generation-a",
                        "table-a",
                        1,
                        "b" * 64,
                        500_000,
                    )
                ]
            )

    database = Database()
    repository = service_module.PeeweeTabularStructureRepository()
    monkeypatch.setattr(
        repository,
        "_models",
        lambda: (database, object, object, object, State, object),
    )

    result = repository.discover_active_tables(
        "tenant-a",
        "dataset-a",
        "anonymous register",
        ("0000000050000000", "a" * 64),
        3,
    )

    sql, params = database.calls[1]
    assert params == (
        "anonymous register",
        "tenant-a",
        "dataset-a",
        "anonymous register",
        "anonymous register",
        50_000_000,
        "anonymous register",
        50_000_000,
        "a" * 64,
        3,
    )
    assert "INNER JOIN document" in sql
    assert "g.status = 'active'" in sql
    assert result["records"][0]["score_encoded"] == "0000000000500000"


def test_peewee_discovery_binds_revision_and_rows_inside_one_transaction_snapshot(
    service_module,
    monkeypatch,
):
    class Field:
        def __eq__(self, other):
            return ("eq", other)

    class Database:
        def __init__(self):
            self.in_transaction = False
            self.sql_calls = 0

        def atomic(self):
            database = self

            class Atomic:
                def __enter__(self):
                    assert database.in_transaction is False
                    database.in_transaction = True

                def __exit__(self, exc_type, exc, traceback):
                    database.in_transaction = False

            return Atomic()

        def execute_sql(self, sql, params):
            assert self.in_transaction is True
            self.sql_calls += 1

            class Cursor:
                def fetchone(self):
                    return None

                def fetchall(self):
                    return []

            return Cursor()

    database = Database()

    class State:
        tenant_id = Field()
        kb_id = Field()
        index_revision = 9
        backfill_status = "complete"
        reads = 0

        @classmethod
        def get_or_none(cls, *conditions):
            assert database.in_transaction is True
            cls.reads += 1
            return cls()

    repository = service_module.PeeweeTabularStructureRepository()
    monkeypatch.setattr(
        repository,
        "_models",
        lambda: (database, object, object, object, State, object),
    )

    result = repository.discover_active_tables(
        "tenant-a",
        "dataset-a",
        "anonymous register",
        None,
        1,
    )

    assert result["index_revision"] == 9
    assert State.reads == 2
    assert database.sql_calls == 2
    assert database.in_transaction is False


def test_peewee_discovery_discards_rows_when_revision_changes_during_read(
    service_module,
    monkeypatch,
):
    class Field:
        def __eq__(self, other):
            return ("eq", other)

    class State:
        tenant_id = Field()
        kb_id = Field()
        backfill_status = "complete"
        reads = 0

        @classmethod
        def get_or_none(cls, *conditions):
            cls.reads += 1
            instance = cls()
            instance.index_revision = 9 if cls.reads == 1 else 10
            return instance

    class Database:
        def atomic(self):
            class Atomic:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

            return Atomic()

        def execute_sql(self, sql, params):
            class Cursor:
                def fetchone(self):
                    return None

                def fetchall(self):
                    return [
                        (
                            "document-a",
                            "generation-a",
                            "table-a",
                            1,
                            "a" * 64,
                            500_000,
                        )
                    ]

            return Cursor()

    repository = service_module.PeeweeTabularStructureRepository()
    monkeypatch.setattr(
        repository,
        "_models",
        lambda: (Database(), object, object, object, State, object),
    )

    result = repository.discover_active_tables(
        "tenant-a",
        "dataset-a",
        "anonymous register",
        None,
        1,
    )

    assert result == {
        "index_revision": 10,
        "backfill_status": "complete",
        "unsafe": False,
        "records": [],
    }


def test_discovery_query_normalization_preserves_semantic_punctuation(service_module):
    normalized = service_module.normalize_tabular_discovery_query(
        "  DVP＆R　2D／3D  ABC－123．4  "
    )

    assert normalized == "dvp&r 2d/3d abc-123.4"


def test_table_index_discovery_is_generation_bound_and_content_free(
    service_module,
    generation_repository,
    table_parser,
):
    storage, projection, receipt = _stored_generation(table_parser)
    service_module.TabularStructureService.register_shadow_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        receipt=receipt,
        repository=generation_repository,
    )
    service_module.TabularStructureService.activate_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        expected_active_generation_ref=None,
        repository=generation_repository,
    )
    generation_repository.complete_backfill("tenant-owner", "dataset-1")
    indexed, unsafe_reason = service_module._table_search_text(
        projection["tables"][0],
    )
    assert unsafe_reason is None
    query = indexed.split()[0]

    result = service_module.TabularStructureService.discover_active_tables(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        query=query,
        cursor=None,
        page_size=10,
        max_pages=2,
        max_evidence_bytes=10_000,
        max_evidence_tokens=10_000,
        deadline_ms=1_000,
        repository=generation_repository,
    )

    assert result["incomplete"] is False
    assert result["index_revision"] >= 1
    assert result["query_digest"] == hashlib.sha256(query.encode()).hexdigest()
    assert len(result["seeds"]) == 1
    assert result["seeds"][0] == {
        "document_id": "document-1",
        "producer_generation_ref": projection["producer_generation_ref"],
        "table_ref": projection["tables"][0]["table_ref"],
        "table_ordinal": 1,
        "retrieval_rule": "bm25-ngram/v1",
        "score_encoded": result["seeds"][0]["score_encoded"],
        "identity_hash": result["seeds"][0]["identity_hash"],
    }
    assert "search_text" not in result["seeds"][0]


def test_table_index_projection_excludes_non_authoritative_fragments(service_module):
    projection = {
        "producer_generation_ref": "generation-a",
        "tables": [
            {
                "table_ref": "table-authoritative",
                "table_ordinal": 1,
                "enumeration_status": "supported_complete",
                "table_label": "Control plan",
                "table_context": [],
                "ordered_columns": [
                    {"header_path": ["Process"], "name": "Process"},
                ],
            },
            {
                "table_ref": "table-fragment",
                "table_ordinal": 2,
                "enumeration_status": "not_guaranteed_explained",
                "table_label": "Control plan metadata",
                "table_context": [],
                "ordered_columns": [
                    {"header_path": ["Process"], "name": "Process"},
                ],
            },
        ],
    }

    records = service_module.build_tabular_discovery_index_projection(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        projection=projection,
    )

    assert [record["table_ref"] for record in records] == ["table-authoritative"]


def test_discovery_revision_is_dataset_scoped(service_module):
    repository = service_module.InMemoryTabularStructureRepository()
    repository.add_authorization_scope("tenant-owner", "dataset-a", "document-a")
    repository.add_authorization_scope("tenant-owner", "dataset-b", "document-b")
    repository.seed_discovery_index(
        tenant_id="tenant-owner",
        dataset_id="dataset-a",
        document_id="document-a",
        producer_generation_ref="generation-a",
        table_ref="table-a",
        search_text="anonymous register",
    )
    repository.seed_discovery_index(
        tenant_id="tenant-owner",
        dataset_id="dataset-a",
        document_id="document-a",
        producer_generation_ref="generation-a",
        table_ref="table-a-2",
        search_text="anonymous register secondary",
        table_ordinal=2,
    )
    repository.seed_discovery_index(
        tenant_id="tenant-owner",
        dataset_id="dataset-b",
        document_id="document-b",
        producer_generation_ref="generation-b",
        table_ref="table-b",
        search_text="anonymous register",
    )
    repository.complete_backfill("tenant-owner", "dataset-a")
    repository.complete_backfill("tenant-owner", "dataset-b")

    first = service_module.TabularStructureService.discover_active_tables(
        tenant_id="tenant-owner",
        dataset_id="dataset-a",
        query="anonymous register",
        cursor=None,
        page_size=1,
        max_pages=2,
        max_evidence_bytes=10_000,
        max_evidence_tokens=10_000,
        deadline_ms=1_000,
        repository=repository,
    )
    repository.advance_dataset_revision("tenant-owner", "dataset-b")
    assert first["next_cursor"] is not None
    replay = service_module.TabularStructureService.discover_active_tables(
        tenant_id="tenant-owner",
        dataset_id="dataset-a",
        query="anonymous register",
        cursor=first["next_cursor"],
        page_size=1,
        max_pages=2,
        max_evidence_bytes=10_000,
        max_evidence_tokens=10_000,
        deadline_ms=1_000,
        repository=repository,
    )

    assert replay["incomplete"] is False
    assert replay["index_revision"] == first["index_revision"]


def test_discovery_cursor_binds_page_ordinal_and_marks_results_beyond_window(
    service_module,
):
    repository = service_module.InMemoryTabularStructureRepository()
    for ordinal in range(1, 4):
        repository.add_authorization_scope(
            "tenant-owner", "dataset-1", f"document-{ordinal}"
        )
        repository.seed_discovery_index(
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id=f"document-{ordinal}",
            producer_generation_ref=f"generation-{ordinal}",
            table_ref=f"table-{ordinal}",
            search_text="anonymous register",
            table_ordinal=ordinal,
        )
    repository.complete_backfill("tenant-owner", "dataset-1")

    first = service_module.TabularStructureService.discover_active_tables(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        query="anonymous register",
        cursor=None,
        page_size=1,
        max_pages=2,
        max_evidence_bytes=10_000,
        max_evidence_tokens=10_000,
        deadline_ms=1_000,
        repository=repository,
    )
    second = service_module.TabularStructureService.discover_active_tables(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        query="anonymous register",
        cursor=first["next_cursor"],
        page_size=1,
        max_pages=2,
        max_evidence_bytes=10_000,
        max_evidence_tokens=10_000,
        deadline_ms=1_000,
        repository=repository,
    )

    assert first["has_more_in_window"] is True
    assert first["has_more_beyond_window"] is False
    assert second["has_more_in_window"] is False
    assert second["has_more_beyond_window"] is True
    assert second["next_cursor"] is None

    with pytest.raises(ValueError, match="cursor"):
        service_module.TabularStructureService.discover_active_tables(
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            query="anonymous register",
            cursor=first["next_cursor"],
            page_size=1,
            max_pages=3,
            max_evidence_bytes=10_000,
            max_evidence_tokens=10_000,
            deadline_ms=1_000,
            repository=repository,
        )


def test_discovery_fails_closed_for_pending_or_unsafe_dataset(service_module):
    repository = service_module.InMemoryTabularStructureRepository()
    repository.add_authorization_scope("tenant-owner", "dataset-1", "document-1")
    repository.seed_discovery_index(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref="generation-a",
        table_ref="table-a",
        search_text="",
        projection_status="unsafe",
        unsafe_reason="unsafe_control_character",
    )

    pending = service_module.TabularStructureService.discover_active_tables(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        query="anonymous register",
        cursor=None,
        page_size=10,
        max_pages=2,
        max_evidence_bytes=10_000,
        max_evidence_tokens=10_000,
        deadline_ms=1_000,
        repository=repository,
    )
    assert pending["incomplete_cause"] == "backfill_pending"

    repository.complete_backfill("tenant-owner", "dataset-1")
    unsafe = service_module.TabularStructureService.discover_active_tables(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        query="anonymous register",
        cursor=None,
        page_size=10,
        max_pages=2,
        max_evidence_bytes=10_000,
        max_evidence_tokens=10_000,
        deadline_ms=1_000,
        repository=repository,
    )
    assert unsafe["incomplete"] is True
    assert unsafe["incomplete_cause"] == "projection_unsafe"
    assert unsafe["seeds"] == []
    assert "unsafe_control_character" not in json.dumps(unsafe)


def test_discovery_deadline_discards_late_repository_results(service_module):
    class SlowRepository(service_module.InMemoryTabularStructureRepository):
        def discover_active_tables(self, *args, **kwargs):
            time.sleep(0.01)
            return super().discover_active_tables(*args, **kwargs)

    repository = SlowRepository()
    repository.add_authorization_scope("tenant-owner", "dataset-1", "document-1")
    repository.seed_discovery_index(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref="generation-a",
        table_ref="table-a",
        search_text="anonymous register",
    )
    repository.complete_backfill("tenant-owner", "dataset-1")

    result = service_module.TabularStructureService.discover_active_tables(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        query="anonymous register",
        cursor=None,
        page_size=10,
        max_pages=2,
        max_evidence_bytes=10_000,
        max_evidence_tokens=10_000,
        deadline_ms=1,
        repository=repository,
    )

    assert result["incomplete"] is True
    assert result["incomplete_cause"] == "deadline"
    assert result["seeds"] == []


def test_backfill_is_resumable_and_rechecks_active_before_committing(
    service_module,
    table_parser,
):
    repository = service_module.InMemoryTabularStructureRepository()
    storage = _Storage()
    projections = []
    for ordinal in range(1, 3):
        document_id = f"document-{ordinal}"
        repository.add_authorization_scope("tenant-owner", "dataset-1", document_id)
        document_storage, projection, receipt = _stored_generation(
            table_parser,
            generation_ref=str(uuid.uuid4()),
            document_id=document_id,
        )
        storage.objects.update(document_storage.objects)
        record = {
            "producer_generation_ref": projection["producer_generation_ref"],
            "tenant_id": "tenant-owner",
            "kb_id": "dataset-1",
            "document_id": document_id,
            "projection_version": projection["version"],
            "producer_schema_version": projection["producer_schema_version"],
            "manifest_object_name": receipt["manifest_object_name"],
            "manifest_sha256": receipt["manifest_sha256"],
            "source_sha256": projection["source_sha256"],
            "row_count": len(projection["rows"]),
            "part_count": receipt["part_count"],
            "status": "active",
            "safe_error_code": None,
            "activated_at": None,
            "retained_at": None,
        }
        repository.inject(record)
        projections.append((record, projection))
    repository.mark_backfill_pending("tenant-owner", "dataset-1")

    first = service_module.TabularStructureService.backfill_active_generation_indexes(
        storage,
        batch_size=1,
        max_batches=1,
        repository=repository,
    )
    assert first == {"batches": 1, "documents": 1, "datasets_completed": 0}
    assert repository.backfill_state("tenant-owner", "dataset-1") == {
        "status": "pending",
        "cursor": "document-1",
    }

    second = service_module.TabularStructureService.backfill_active_generation_indexes(
        storage,
        batch_size=1,
        max_batches=1,
        repository=repository,
    )
    assert second == {"batches": 1, "documents": 1, "datasets_completed": 1}
    assert repository.backfill_state("tenant-owner", "dataset-1") == {
        "status": "complete",
        "cursor": None,
    }

    repository.mark_backfill_pending("tenant-owner", "dataset-1")
    repository.set_backfill_commit_hook(
        lambda: repository.remove_active_generation(
            projections[0][0]["producer_generation_ref"]
        )
    )
    with pytest.raises(service_module.StructureSnapshotChanged):
        service_module.TabularStructureService.backfill_active_generation_indexes(
            storage,
            batch_size=1,
            max_batches=1,
            repository=repository,
        )
    assert repository.backfill_state("tenant-owner", "dataset-1") == {
        "status": "pending",
        "cursor": None,
    }


def test_backfill_ignores_active_generations_outside_the_current_contract(
    service_module,
    table_parser,
):
    repository = service_module.InMemoryTabularStructureRepository()
    storage, projection, receipt = _stored_generation(
        table_parser,
        document_id="document-current",
    )
    current = {
        "producer_generation_ref": projection["producer_generation_ref"],
        "tenant_id": "tenant-owner",
        "kb_id": "dataset-1",
        "document_id": "document-current",
        "projection_version": projection["version"],
        "producer_schema_version": projection["producer_schema_version"],
        "manifest_object_name": receipt["manifest_object_name"],
        "manifest_sha256": receipt["manifest_sha256"],
        "source_sha256": projection["source_sha256"],
        "row_count": len(projection["rows"]),
        "part_count": receipt["part_count"],
        "status": "active",
        "safe_error_code": None,
        "activated_at": None,
        "retained_at": None,
    }
    legacy = {
        **current,
        "producer_generation_ref": str(uuid.uuid4()),
        "document_id": "document-legacy",
        "projection_version": "tabular-structure-projection/legacy",
        "producer_schema_version": "table-producer/legacy",
        "manifest_object_name": "legacy-manifest-is-not-readable",
        "manifest_sha256": "0" * 64,
    }
    repository.inject(legacy)
    repository.inject(current)
    repository.mark_backfill_pending("tenant-owner", "dataset-1")

    result = service_module.TabularStructureService.backfill_active_generation_indexes(
        storage,
        repository=repository,
    )

    assert result == {"batches": 1, "documents": 1, "datasets_completed": 1}
    assert repository.backfill_state("tenant-owner", "dataset-1") == {
        "status": "complete",
        "cursor": None,
    }


def test_backfill_validates_known_historical_inner_contract_and_indexes_current_only(
    service_module,
    table_parser,
):
    repository = service_module.InMemoryTabularStructureRepository()
    current_storage, current_projection, current_receipt = _stored_generation(
        table_parser,
        document_id="document-current",
    )
    historical_storage, historical_projection, historical_receipt = (
        _stored_generation_with_contract(
            table_parser,
            document_id="document-historical",
            structure_algorithm_version="region-producer/v10",
            enumeration_rule_version="enumeration-rules/v3",
        )
    )
    storage = _Storage()
    storage.objects.update(current_storage.objects)
    storage.objects.update(historical_storage.objects)
    repository.inject(
        _active_generation_record(
            current_projection,
            current_receipt,
            document_id="document-current",
        )
    )
    repository.inject(
        _active_generation_record(
            historical_projection,
            historical_receipt,
            document_id="document-historical",
        )
    )
    repository.mark_backfill_pending("tenant-owner", "dataset-1")

    result = service_module.TabularStructureService.backfill_active_generation_indexes(
        storage,
        repository=repository,
    )
    discovery = service_module.TabularStructureService.discover_active_tables(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        query="Inspection",
        cursor=None,
        page_size=10,
        max_pages=1,
        max_evidence_bytes=10_000,
        max_evidence_tokens=10_000,
        deadline_ms=10_000,
        repository=repository,
    )

    assert result == {"batches": 1, "documents": 2, "datasets_completed": 1}
    assert discovery["incomplete"] is False
    assert {
        seed["producer_generation_ref"] for seed in discovery["seeds"]
    } == {current_projection["producer_generation_ref"]}


def test_backfill_with_only_known_historical_inner_contract_finishes_without_candidates(
    service_module,
    table_parser,
):
    repository = service_module.InMemoryTabularStructureRepository()
    storage, projection, receipt = _stored_generation_with_contract(
        table_parser,
        document_id="document-historical",
        structure_algorithm_version="region-producer/v10",
        enumeration_rule_version="enumeration-rules/v3",
    )
    repository.inject(
        _active_generation_record(
            projection,
            receipt,
            document_id="document-historical",
        )
    )
    repository.mark_backfill_pending("tenant-owner", "dataset-1")

    result = service_module.TabularStructureService.backfill_active_generation_indexes(
        storage,
        repository=repository,
    )
    discovery = service_module.TabularStructureService.discover_active_tables(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        query="Inspection",
        cursor=None,
        page_size=10,
        max_pages=1,
        max_evidence_bytes=10_000,
        max_evidence_tokens=10_000,
        deadline_ms=10_000,
        repository=repository,
    )

    assert result == {"batches": 1, "documents": 1, "datasets_completed": 1}
    assert discovery["incomplete"] is False
    assert discovery["seeds"] == []


def test_backfill_rejects_unknown_inner_contract_instead_of_broadly_skipping(
    service_module,
    table_parser,
):
    repository = service_module.InMemoryTabularStructureRepository()
    storage, projection, receipt = _stored_generation_with_contract(
        table_parser,
        document_id="document-unknown",
        structure_algorithm_version="region-producer/v999",
        enumeration_rule_version="enumeration-rules/v999",
    )
    repository.inject(
        _active_generation_record(
            projection,
            receipt,
            document_id="document-unknown",
        )
    )
    repository.mark_backfill_pending("tenant-owner", "dataset-1")

    with pytest.raises(
        service_module.StructureSnapshotChanged,
        match="manifest version changed",
    ):
        service_module.TabularStructureService.backfill_active_generation_indexes(
            storage,
            repository=repository,
        )


@pytest.mark.parametrize(
    "corruption",
    ["manifest_digest", "manifest_generation", "manifest_scope", "part_count", "part_digest"],
)
def test_historical_inner_contract_backfill_keeps_snapshot_corruption_fail_closed(
    service_module,
    table_parser,
    corruption,
):
    repository = service_module.InMemoryTabularStructureRepository()
    storage, projection, receipt = _stored_generation_with_contract(
        table_parser,
        document_id="document-historical",
        structure_algorithm_version="region-producer/v10",
        enumeration_rule_version="enumeration-rules/v3",
    )
    if corruption == "manifest_digest":
        key = ("dataset-1", receipt["manifest_object_name"])
        storage.objects[key] += b" "
    elif corruption == "manifest_generation":
        old_name = receipt["manifest_object_name"]
        manifest = json.loads(storage.objects[("dataset-1", old_name)])
        manifest["producer_generation_ref"] = str(uuid.uuid4())
        payload = tabular_structure._canonical_json(manifest)
        digest = hashlib.sha256(payload).hexdigest()
        receipt["manifest_sha256"] = digest
        receipt["manifest_object_name"] = old_name.rsplit("manifest-", 1)[0] + f"manifest-{digest}.json"
        storage.objects[("dataset-1", receipt["manifest_object_name"])] = payload
    elif corruption == "manifest_scope":
        receipt["manifest_object_name"] = "_fuxi/tabular-structure/v1/wrong/manifest.json"
    elif corruption == "part_count":
        receipt["part_count"] = 2
    else:
        manifest = json.loads(
            storage.objects[("dataset-1", receipt["manifest_object_name"])]
        )
        part_name = manifest["parts"][0]["object_name"]
        storage.objects[("dataset-1", part_name)] += b" "
    repository.inject(
        _active_generation_record(
            projection,
            receipt,
            document_id="document-historical",
        )
    )
    repository.mark_backfill_pending("tenant-owner", "dataset-1")

    with pytest.raises(service_module.StructureSnapshotChanged):
        service_module.TabularStructureService.backfill_active_generation_indexes(
            storage,
            repository=repository,
        )


def test_backfill_has_no_default_batch_count_truncation(service_module):
    signature = inspect.signature(
        service_module.TabularStructureService.backfill_active_generation_indexes
    )

    assert signature.parameters["max_batches"].default is None


def test_peewee_backfill_query_filters_legacy_contract_before_manifest_io(
    service_module,
    monkeypatch,
):
    peewee = pytest.importorskip("peewee")
    test_database = peewee.SqliteDatabase(":memory:")

    class Base(peewee.Model):
        class Meta:
            database = test_database

    class Document(Base):
        id = peewee.CharField(primary_key=True)
        kb_id = peewee.CharField()

    class Generation(Base):
        producer_generation_ref = peewee.CharField(primary_key=True)
        tenant_id = peewee.CharField()
        kb_id = peewee.CharField()
        document_id = peewee.CharField()
        projection_version = peewee.CharField()
        producer_schema_version = peewee.CharField()
        status = peewee.CharField()

        def to_dict(self):
            return dict(self.__data__)

    test_database.create_tables([Document, Generation])
    Document.insert_many(
        [
            {"id": "document-current", "kb_id": "dataset-1"},
            {"id": "document-legacy", "kb_id": "dataset-1"},
        ]
    ).execute()
    Generation.insert_many(
        [
            {
                "producer_generation_ref": "generation-current",
                "tenant_id": "tenant-owner",
                "kb_id": "dataset-1",
                "document_id": "document-current",
                "projection_version": tabular_structure.PROJECTION_VERSION,
                "producer_schema_version": tabular_structure.PRODUCER_SCHEMA_VERSION,
                "status": "active",
            },
            {
                "producer_generation_ref": "generation-legacy",
                "tenant_id": "tenant-owner",
                "kb_id": "dataset-1",
                "document_id": "document-legacy",
                "projection_version": "tabular-structure-projection/legacy",
                "producer_schema_version": "table-producer/legacy",
                "status": "active",
            },
        ]
    ).execute()
    repository = service_module.PeeweeTabularStructureRepository()
    monkeypatch.setattr(
        repository,
        "_models",
        lambda: (test_database, Document, object, Generation, object, object),
    )

    try:
        records = repository.list_active_generations_for_backfill(
            "tenant-owner",
            "dataset-1",
            None,
            100,
        )

        assert [record["producer_generation_ref"] for record in records] == [
            "generation-current"
        ]
    finally:
        test_database.close()


def test_discovery_api_is_post_only_and_rejects_client_authorization_scope():
    source = CHUNK_API_PATH.read_text(encoding="utf-8")

    assert (
        '@manager.route("/datasets/<dataset_id>/tabular-structure/discovery/v1", methods=["POST"])'
        in source
    )
    assert "tenant_id" in source
    assert "provider_scope" in source
    assert "unexpected discovery request field" in source
    assert "discover_active_tables" in source


def test_authorized_active_generation_projects_document_name_separately_from_manifest():
    module = ast.parse(CHUNK_API_PATH.read_text(encoding="utf-8"))
    route = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "get_active_tabular_structure_generation"
    )
    source_name_helper = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_authorized_document_source_name"
    )
    string_values = {
        node.value
        for node in ast.walk(route)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    helper_calls = [
        node
        for node in ast.walk(route)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_authorized_document_source_name"
    ]
    query_calls = [
        node
        for node in ast.walk(source_name_helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "query"
    ]

    assert "document_name" in string_values
    assert helper_calls
    assert query_calls, "the authorized document control record must provide source identity"


def test_active_table_route_binds_document_generation_table_and_source_identity():
    module = ast.parse(CHUNK_API_PATH.read_text(encoding="utf-8"))
    route = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "get_active_tabular_structure_table"
    )
    source = ast.unparse(route)

    assert "_authorized_structure_owner(tenant_id, dataset_id, document_id)" in source
    assert "request.args.get('generation_ref')" in source
    assert "read_active_table" in source
    assert "producer_generation_ref=generation_ref" in source
    assert "table_ref=table_ref" in source
    assert "_authorized_document_source_name(dataset_id, document_id)" in source


@pytest.mark.parametrize(
    ("documents", "message"),
    [
        ([], "unavailable"),
        ([object(), object()], "unavailable"),
        ([type("Document", (), {"name": None})()], "unavailable"),
        ([type("Document", (), {"name": "unsafe\u0000name.xlsx"})()], "invalid"),
        ([type("Document", (), {"name": "x" * 1025})()], "invalid"),
    ],
)
def test_authorized_document_source_name_rejects_ambiguous_or_unsafe_identity(
    documents,
    message,
):
    class StubDocumentService:
        @staticmethod
        def query(**kwargs):
            assert kwargs == {"id": "document-1", "kb_id": "dataset-1"}
            return documents

    helper = _load_authorized_document_source_name(StubDocumentService)

    with pytest.raises(ValueError, match=message):
        helper("dataset-1", "document-1")


def test_authorized_document_source_name_returns_bounded_nfc_identity():
    decomposed = "  cafe\u0301.xlsx  "

    class StubDocumentService:
        @staticmethod
        def query(**kwargs):
            assert kwargs == {"id": "document-1", "kb_id": "dataset-1"}
            return [type("Document", (), {"name": decomposed})()]

    helper = _load_authorized_document_source_name(StubDocumentService)

    assert helper("dataset-1", "document-1") == "caf\u00e9.xlsx"


def test_peewee_repository_locks_document_and_switches_inside_one_transaction():
    module = ast.parse(SERVICE_MODULE_PATH.read_text(encoding="utf-8"))
    repository = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "PeeweeTabularStructureRepository")
    activate = next(node for node in repository.body if isinstance(node, ast.FunctionDef) and node.name == "activate")
    calls = [node for node in ast.walk(activate) if isinstance(node, ast.Call)]
    attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}

    assert "atomic" in attributes
    assert "for_update" in attributes
    assert "execute" in attributes

    assigned_names = {
        target.id
        for node in ast.walk(activate)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert {"retained_count", "activated_count"} <= assigned_names

    conflict_messages = {
        argument.value
        for node in ast.walk(activate)
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
        for argument in node.exc.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }
    assert "active generation compare-and-swap failed" in conflict_messages
    assert "shadow generation compare-and-swap failed" in conflict_messages


def test_peewee_shadow_registration_collapses_concurrent_duplicate_insert():
    module = ast.parse(SERVICE_MODULE_PATH.read_text(encoding="utf-8"))
    repository = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "PeeweeTabularStructureRepository")
    add_shadow = next(node for node in repository.body if isinstance(node, ast.FunctionDef) and node.name == "add_shadow")

    caught_names = {
        handler.type.id
        for node in ast.walk(add_shadow)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if isinstance(handler.type, ast.Name)
    }
    assert "IntegrityError" in caught_names

    attributes = {
        node.func.attr
        for node in ast.walk(add_shadow)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "atomic" in attributes
    assert "for_update" in attributes

    source = ast.get_source_segment(
        SERVICE_MODULE_PATH.read_text(encoding="utf-8"), add_shadow
    )
    assert source is not None
    assert "Document.select()" in source
    assert 'record["document_id"]' in source
    assert 'record["kb_id"]' in source

    get_calls = [
        node
        for node in ast.walk(add_shadow)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_by_id"
    ]
    assert len(get_calls) >= 2


def test_stored_projection_reader_rejects_tampered_manifest_and_part(service_module, table_parser):
    storage, projection, receipt = _stored_generation(table_parser)

    loaded = tabular_structure.load_tabular_structure_projection(
        storage,
        bucket="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        manifest_object_name=receipt["manifest_object_name"],
        manifest_sha256=receipt["manifest_sha256"],
        tenant_id="tenant-owner",
    )
    assert loaded == projection

    tampered_manifest = copy.deepcopy(storage.objects)
    manifest_key = ("dataset-1", receipt["manifest_object_name"])
    storage.objects[manifest_key] = tampered_manifest[manifest_key] + b" "
    with pytest.raises(service_module.StructureSnapshotChanged, match="manifest digest"):
        tabular_structure.load_tabular_structure_projection(
            storage,
            bucket="dataset-1",
            document_id="document-1",
            producer_generation_ref=projection["producer_generation_ref"],
            manifest_object_name=receipt["manifest_object_name"],
            manifest_sha256=receipt["manifest_sha256"],
            tenant_id="tenant-owner",
        )


def test_stored_projection_preserves_and_validates_enumeration_rule_version(
    service_module,
    table_parser,
):
    storage, projection, receipt = _stored_generation(table_parser)
    manifest_key = ("dataset-1", receipt["manifest_object_name"])
    manifest = json.loads(storage.objects[manifest_key])

    assert manifest["enumeration_rule_version"] == "enumeration-rules/v9"
    loaded = tabular_structure.load_tabular_structure_projection(
        storage,
        bucket="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        manifest_object_name=receipt["manifest_object_name"],
        manifest_sha256=receipt["manifest_sha256"],
        tenant_id="tenant-owner",
    )
    assert loaded["enumeration_rule_version"] == projection["enumeration_rule_version"]

    manifest["enumeration_rule_version"] = "enumeration-rules/unknown"
    tampered_payload = tabular_structure._canonical_json(manifest)
    tampered_sha256 = hashlib.sha256(tampered_payload).hexdigest()
    tampered_name = receipt["manifest_object_name"].rsplit("manifest-", 1)[0] + f"manifest-{tampered_sha256}.json"
    storage.objects[("dataset-1", tampered_name)] = tampered_payload

    with pytest.raises(service_module.StructureSnapshotChanged, match="manifest version"):
        tabular_structure.load_tabular_structure_projection(
            storage,
            bucket="dataset-1",
            document_id="document-1",
            producer_generation_ref=projection["producer_generation_ref"],
            manifest_object_name=tampered_name,
            manifest_sha256=tampered_sha256,
            tenant_id="tenant-owner",
        )


def test_stored_projection_reader_binds_objects_to_document_and_generation(service_module, table_parser):
    storage, projection, receipt = _stored_generation(table_parser)

    with pytest.raises(service_module.StructureSnapshotChanged, match="document scope"):
        tabular_structure.load_tabular_structure_projection(
            storage,
            bucket="dataset-1",
            document_id="document-2",
            producer_generation_ref=projection["producer_generation_ref"],
            manifest_object_name=receipt["manifest_object_name"],
            manifest_sha256=receipt["manifest_sha256"],
            tenant_id="tenant-owner",
        )

    part_key = next(key for key in storage.objects if "/part-" in key[1])
    storage.objects[part_key] += b" "
    with pytest.raises(service_module.StructureSnapshotChanged, match="part digest"):
        tabular_structure.load_tabular_structure_projection(
            storage,
            bucket="dataset-1",
            document_id="document-1",
            producer_generation_ref=projection["producer_generation_ref"],
            manifest_object_name=receipt["manifest_object_name"],
            manifest_sha256=receipt["manifest_sha256"],
            tenant_id="tenant-owner",
        )


def test_projection_validator_rejects_unbounded_or_inconsistent_manifest_context(table_parser):
    _storage, projection, _receipt = _stored_generation(table_parser)

    oversized = copy.deepcopy(projection)
    oversized["rows"][0]["table_label_kwd"] = "x" * 129
    with pytest.raises(ValueError, match="table label exceeds"):
        tabular_structure.validate_tabular_structure_projection(oversized)

    unsafe = copy.deepcopy(projection)
    unsafe["rows"][0]["table_context_list"] = json.dumps([{"name": "context", "value": "unsafe\u202evalue"}])
    with pytest.raises(ValueError, match="table context contains unsafe controls"):
        tabular_structure.validate_tabular_structure_projection(unsafe)

    inconsistent = copy.deepcopy(projection)
    table_ref = inconsistent["rows"][0]["table_ref_kwd"]
    same_table_rows = [row for row in inconsistent["rows"] if row["table_ref_kwd"] == table_ref]
    same_table_rows[1]["table_context_list"] = json.dumps([{"name": "context", "value": "different"}])
    with pytest.raises(ValueError, match="table context is not generation-wide"):
        tabular_structure.validate_tabular_structure_projection(inconsistent)

def test_generation_bound_page_has_stable_order_total_and_explicit_end(table_parser):
    _storage, projection, _receipt = _stored_generation(table_parser)
    table_ref = projection["tables"][0]["table_ref"]

    first = tabular_structure.page_tabular_structure_rows(projection, table_ref=table_ref, cursor=0, page_size=2)
    second = tabular_structure.page_tabular_structure_rows(projection, table_ref=table_ref, cursor=first["next_cursor"], page_size=2)

    combined = first["rows"] + second["rows"]
    assert [row["data_row_index_int"] for row in combined] == [1, 2, 3]
    assert first["producer_generation_ref"] == projection["producer_generation_ref"]
    assert first["total"] == 3
    assert first["source_total_count"] == 3
    assert first["has_more"] is True
    assert second["has_more"] is False
    assert second["next_cursor"] is None


def test_generation_bound_compact_page_is_lossless_and_smaller(table_parser):
    _storage, projection, _receipt = _stored_generation(table_parser)
    table_ref = projection["tables"][0]["table_ref"]

    verbose = tabular_structure.page_tabular_structure_rows(
        projection,
        table_ref=table_ref,
        cursor=0,
        page_size=3,
    )
    compact = tabular_structure.page_tabular_structure_rows(
        projection,
        table_ref=table_ref,
        cursor=0,
        page_size=3,
        row_transport_version="tabular-row-page-compact/v1",
    )

    assert compact["row_transport_version"] == "tabular-row-page-compact/v1"
    assert {
        key: compact[key]
        for key in (
            "producer_generation_ref",
            "table_ref",
            "producer_schema_version",
            "projection_version",
            "structure_algorithm_version",
            "enumeration_rule_version",
            "total",
            "source_total_count",
            "has_more",
            "next_cursor",
        )
    } == {
        key: verbose[key]
        for key in (
            "producer_generation_ref",
            "table_ref",
            "producer_schema_version",
            "projection_version",
            "structure_algorithm_version",
            "enumeration_rule_version",
            "total",
            "source_total_count",
            "has_more",
            "next_cursor",
        )
    }
    assert compact["rows"] == [
        [
            row["row_ordinal_int"],
            row["data_row_index_int"],
            row["row_role_kwd"],
            [
                [field["column_ordinal"], field["value"]]
                for field in json.loads(row["ordered_fields_list"])
            ],
        ]
        for row in verbose["rows"]
    ]
    assert len(tabular_structure._canonical_json(compact)) < len(
        tabular_structure._canonical_json(verbose)
    )


def test_shadow_registration_is_scope_checked_and_idempotent(service_module, generation_repository, table_parser):
    storage, projection, receipt = _stored_generation(table_parser)

    registered = service_module.TabularStructureService.register_shadow_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        receipt=receipt,
        repository=generation_repository,
    )
    repeated = service_module.TabularStructureService.register_shadow_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        receipt=receipt,
        repository=generation_repository,
    )

    assert registered["status"] == "shadow"
    assert repeated == registered
    assert registered["producer_generation_ref"] == projection["producer_generation_ref"]
    assert "manifest_object_name" not in registered
    mismatched_part_receipt = {**receipt, "part_count": receipt["part_count"] + 1}
    with pytest.raises(service_module.StructureSnapshotChanged, match="part count"):
        service_module.TabularStructureService.register_shadow_generation(
            storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
            receipt=mismatched_part_receipt,
            repository=generation_repository,
        )
    with pytest.raises(PermissionError, match="authorization scope"):
        service_module.TabularStructureService.register_shadow_generation(
            storage,
            tenant_id="other-tenant",
            dataset_id="dataset-1",
            document_id="document-1",
            receipt=receipt,
            repository=generation_repository,
        )


def test_activation_atomically_retains_old_generation_and_checks_expected_pointer(service_module, generation_repository, table_parser):
    first_storage, first_projection, first_receipt = _stored_generation(table_parser)
    second_storage, second_projection, second_receipt = _stored_generation(table_parser)
    for storage, receipt in ((first_storage, first_receipt), (second_storage, second_receipt)):
        service_module.TabularStructureService.register_shadow_generation(
            storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
            receipt=receipt,
            repository=generation_repository,
        )

    activated_first = service_module.TabularStructureService.activate_generation(
        first_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=first_projection["producer_generation_ref"],
        expected_active_generation_ref=None,
        repository=generation_repository,
    )
    assert activated_first["status"] == "active"

    with pytest.raises(service_module.StructureSnapshotChanged, match="active generation changed") as changed:
        service_module.TabularStructureService.activate_generation(
            second_storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
            producer_generation_ref=second_projection["producer_generation_ref"],
            expected_active_generation_ref=None,
            repository=generation_repository,
        )
    assert changed.value.active_generation_ref == first_projection["producer_generation_ref"]

    activated_second = service_module.TabularStructureService.activate_generation(
        second_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=second_projection["producer_generation_ref"],
        expected_active_generation_ref=first_projection["producer_generation_ref"],
        repository=generation_repository,
    )
    assert activated_second["status"] == "active"
    assert activated_second["producer_generation_ref"] == second_projection["producer_generation_ref"]
    assert activated_second["projection_version"] == second_projection["version"]
    assert activated_second["producer_schema_version"] == second_projection["producer_schema_version"]
    assert activated_second["structure_algorithm_version"] == second_projection["structure_algorithm_version"]
    assert activated_second["enumeration_rule_version"] == second_projection["enumeration_rule_version"]
    assert activated_second["row_count"] == len(second_projection["rows"])
    assert generation_repository.get(first_projection["producer_generation_ref"])["status"] == "retained"
    active = service_module.TabularStructureService.get_active_generation(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        repository=generation_repository,
    )
    assert active["producer_generation_ref"] == second_projection["producer_generation_ref"]


def test_activation_rejects_cross_document_generation_before_storage_read(service_module, generation_repository, table_parser):
    storage, projection, receipt = _stored_generation(table_parser)
    generation_repository.add_authorization_scope("tenant-owner", "dataset-1", "document-2")
    service_module.TabularStructureService.register_shadow_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        receipt=receipt,
        repository=generation_repository,
    )
    storage.get_calls.clear()

    with pytest.raises(service_module.StructureSnapshotMissing):
        service_module.TabularStructureService.activate_generation(
            storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-2",
            producer_generation_ref=projection["producer_generation_ref"],
            expected_active_generation_ref=None,
            repository=generation_repository,
        )

    assert storage.get_calls == []


def test_active_reads_require_exact_generation_and_never_fallback(service_module, generation_repository, table_parser):
    storage, projection, receipt = _stored_generation(table_parser)
    service_module.TabularStructureService.register_shadow_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        receipt=receipt,
        repository=generation_repository,
    )

    with pytest.raises(service_module.StructureSnapshotMissing):
        service_module.TabularStructureService.read_active_manifest(
            storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
            producer_generation_ref=projection["producer_generation_ref"],
            repository=generation_repository,
        )

    service_module.TabularStructureService.activate_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        expected_active_generation_ref=None,
        repository=generation_repository,
    )
    with pytest.raises(service_module.StructureSnapshotChanged):
        service_module.TabularStructureService.read_active_manifest(
            storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
            producer_generation_ref=str(uuid.uuid4()),
            repository=generation_repository,
        )

    manifest = service_module.TabularStructureService.read_active_manifest(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        repository=generation_repository,
    )
    assert manifest["producer_generation_ref"] == projection["producer_generation_ref"]
    assert manifest["enumeration_rule_version"] == "enumeration-rules/v9"
    assert manifest["tables"]
    assert manifest["tables"][0]["table_label"] == "Inspection"
    assert isinstance(manifest["tables"][0]["table_context"], list)
    assert "manifest_object_name" not in manifest
    assert "source_sha256" not in manifest


def test_exact_generation_reads_support_shadow_active_and_retained_with_scope_binding(
    service_module,
    generation_repository,
    table_parser,
):
    first_storage, first_projection, first_receipt = _stored_generation(table_parser)
    second_storage, second_projection, second_receipt = _stored_generation(table_parser)
    for storage, receipt in (
        (first_storage, first_receipt),
        (second_storage, second_receipt),
    ):
        service_module.TabularStructureService.register_shadow_generation(
            storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
            receipt=receipt,
            repository=generation_repository,
        )

    shadow = service_module.TabularStructureService.read_generation_manifest(
        first_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=first_projection["producer_generation_ref"],
        repository=generation_repository,
    )
    assert shadow["producer_generation_ref"] == first_projection["producer_generation_ref"]
    assert shadow["row_count"] == len(first_projection["rows"])

    service_module.TabularStructureService.activate_generation(
        first_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=first_projection["producer_generation_ref"],
        expected_active_generation_ref=None,
        repository=generation_repository,
    )
    active = service_module.TabularStructureService.read_generation_manifest(
        first_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=first_projection["producer_generation_ref"],
        repository=generation_repository,
    )
    assert active == shadow

    service_module.TabularStructureService.activate_generation(
        second_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=second_projection["producer_generation_ref"],
        expected_active_generation_ref=first_projection["producer_generation_ref"],
        repository=generation_repository,
    )
    retained = service_module.TabularStructureService.read_generation_manifest(
        first_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=first_projection["producer_generation_ref"],
        repository=generation_repository,
    )
    assert retained == shadow

    first_storage.get_calls.clear()
    with pytest.raises(service_module.StructureSnapshotMissing):
        service_module.TabularStructureService.read_generation_manifest(
            first_storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="other-document",
            producer_generation_ref=first_projection["producer_generation_ref"],
            repository=generation_repository,
        )
    assert first_storage.get_calls == []


def test_exact_generation_row_read_forwards_compact_transport_after_scope_binding(
    service_module,
    generation_repository,
    table_parser,
):
    storage, projection, receipt = _stored_generation(table_parser)
    service_module.TabularStructureService.register_shadow_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        receipt=receipt,
        repository=generation_repository,
    )

    table_ref = projection["tables"][0]["table_ref"]
    result = service_module.TabularStructureService.read_generation_rows(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        table_ref=table_ref,
        row_transport_version="tabular-row-page-compact/v1",
        repository=generation_repository,
    )

    assert result["row_transport_version"] == "tabular-row-page-compact/v1"
    assert result["producer_generation_ref"] == projection["producer_generation_ref"]
    assert result["table_ref"] == table_ref
    assert result["rows"][0] == [
        projection["rows"][0]["row_ordinal_int"],
        projection["rows"][0]["data_row_index_int"],
        projection["rows"][0]["row_role_kwd"],
        [
            [field["column_ordinal"], field["value"]]
            for field in json.loads(projection["rows"][0]["ordered_fields_list"])
        ],
    ]


def test_active_table_read_returns_only_the_generation_bound_table_metadata(
    service_module,
    generation_repository,
    table_parser,
):
    storage, projection, receipt = _stored_generation(table_parser)
    service_module.TabularStructureService.register_shadow_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        receipt=receipt,
        repository=generation_repository,
    )
    service_module.TabularStructureService.activate_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        expected_active_generation_ref=None,
        repository=generation_repository,
    )

    table_ref = projection["tables"][0]["table_ref"]
    result = service_module.TabularStructureService.read_active_table(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        table_ref=table_ref,
        repository=generation_repository,
    )

    assert result["producer_generation_ref"] == projection["producer_generation_ref"]
    assert result["table"] == projection["tables"][0]
    assert set(result) == {
        "producer_generation_ref",
        "producer_schema_version",
        "projection_version",
        "structure_algorithm_version",
        "enumeration_rule_version",
        "table",
    }


def test_managed_generation_receipts_preserve_validated_source_identity(
    service_module,
    generation_repository,
    table_parser,
):
    first_storage, first_projection, first_receipt = _stored_generation(table_parser)
    second_storage, second_projection, second_receipt = _stored_generation(table_parser)
    for storage, receipt in (
        (first_storage, first_receipt),
        (second_storage, second_receipt),
    ):
        service_module.TabularStructureService.register_shadow_generation(
            storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
            receipt=receipt,
            repository=generation_repository,
        )

    shadow = service_module.TabularStructureService.read_generation(
        first_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=first_projection["producer_generation_ref"],
        repository=generation_repository,
    )
    first_active = service_module.TabularStructureService.activate_generation(
        first_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=first_projection["producer_generation_ref"],
        expected_active_generation_ref=None,
        repository=generation_repository,
    )
    second_active = service_module.TabularStructureService.activate_generation(
        second_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=second_projection["producer_generation_ref"],
        expected_active_generation_ref=first_projection["producer_generation_ref"],
        repository=generation_repository,
    )
    restored = service_module.TabularStructureService.restore_retained_generation(
        first_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        retained_generation_ref=first_projection["producer_generation_ref"],
        expected_active_generation_ref=second_projection["producer_generation_ref"],
        repository=generation_repository,
    )

    assert shadow["source_sha256"] == first_projection["source_sha256"]
    assert first_active["source_sha256"] == first_projection["source_sha256"]
    assert second_active["source_sha256"] == second_projection["source_sha256"]
    assert restored["source_sha256"] == first_projection["source_sha256"]


def test_active_generation_public_identity_preserves_source_sha256(
    service_module,
    generation_repository,
    table_parser,
):
    storage, projection, receipt = _stored_generation(table_parser)
    service_module.TabularStructureService.register_shadow_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        receipt=receipt,
        repository=generation_repository,
    )
    service_module.TabularStructureService.activate_generation(
        storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=projection["producer_generation_ref"],
        expected_active_generation_ref=None,
        repository=generation_repository,
    )

    active = service_module.TabularStructureService.get_active_generation(
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        repository=generation_repository,
    )

    assert active["source_sha256"] == projection["source_sha256"]


def test_retained_restore_atomically_switches_expected_active_generation(
    service_module,
    generation_repository,
    table_parser,
):
    first_storage, first_projection, first_receipt = _stored_generation(table_parser)
    second_storage, second_projection, second_receipt = _stored_generation(table_parser)
    for storage, receipt in (
        (first_storage, first_receipt),
        (second_storage, second_receipt),
    ):
        service_module.TabularStructureService.register_shadow_generation(
            storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
            receipt=receipt,
            repository=generation_repository,
        )
    service_module.TabularStructureService.activate_generation(
        first_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=first_projection["producer_generation_ref"],
        expected_active_generation_ref=None,
        repository=generation_repository,
    )
    service_module.TabularStructureService.activate_generation(
        second_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        producer_generation_ref=second_projection["producer_generation_ref"],
        expected_active_generation_ref=first_projection["producer_generation_ref"],
        repository=generation_repository,
    )

    with pytest.raises(service_module.StructureSnapshotChanged, match="active generation changed"):
        service_module.TabularStructureService.restore_retained_generation(
            first_storage,
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
            retained_generation_ref=first_projection["producer_generation_ref"],
            expected_active_generation_ref=str(uuid.uuid4()),
            repository=generation_repository,
        )
    assert generation_repository.get(second_projection["producer_generation_ref"])["status"] == "active"
    assert generation_repository.get(first_projection["producer_generation_ref"])["status"] == "retained"

    restored = service_module.TabularStructureService.restore_retained_generation(
        first_storage,
        tenant_id="tenant-owner",
        dataset_id="dataset-1",
        document_id="document-1",
        retained_generation_ref=first_projection["producer_generation_ref"],
        expected_active_generation_ref=second_projection["producer_generation_ref"],
        repository=generation_repository,
    )
    assert restored["status"] == "active"
    assert restored["producer_generation_ref"] == first_projection["producer_generation_ref"]
    assert restored["projection_version"] == first_projection["version"]
    assert restored["producer_schema_version"] == first_projection["producer_schema_version"]
    assert restored["structure_algorithm_version"] == first_projection["structure_algorithm_version"]
    assert restored["enumeration_rule_version"] == first_projection["enumeration_rule_version"]
    assert restored["row_count"] == len(first_projection["rows"])
    assert generation_repository.get(second_projection["producer_generation_ref"])["status"] == "retained"


def test_multiple_active_rows_fail_closed_instead_of_picking_one(service_module, generation_repository):
    for generation_ref in (str(uuid.uuid4()), str(uuid.uuid4())):
        generation_repository.inject(
            {
                "producer_generation_ref": generation_ref,
                "tenant_id": "tenant-owner",
                "kb_id": "dataset-1",
                "document_id": "document-1",
                "projection_version": "tabular-structure-projection/v2",
                "producer_schema_version": "table-producer/v4",
                "manifest_object_name": f"internal/{generation_ref}",
                "manifest_sha256": "a" * 64,
                "source_sha256": "b" * 64,
                "row_count": 1,
                "part_count": 1,
                "status": "active",
            }
        )

    with pytest.raises(service_module.StructureGenerationConflict, match="multiple active"):
        service_module.TabularStructureService.get_active_generation(
            tenant_id="tenant-owner",
            dataset_id="dataset-1",
            document_id="document-1",
            repository=generation_repository,
        )
