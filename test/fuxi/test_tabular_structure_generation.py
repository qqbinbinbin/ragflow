import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import sys
import time
import uuid
from pathlib import Path

import pytest

from rag.app import tabular_structure
from test.fuxi.test_table_semantic_rows import _load_table_module
from test.fuxi.test_tabular_structure_projection import _workbook_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_MODULE_PATH = REPO_ROOT / "api" / "db" / "services" / "tabular_structure_service.py"
CHUNK_API_PATH = REPO_ROOT / "api" / "apps" / "restful_apis" / "chunk_api.py"


class _Storage:
    def __init__(self):
        self.objects = {}
        self.get_calls = []

    def put(self, bucket, name, binary, tenant_id=None):
        self.objects[(bucket, name)] = bytes(binary)

    def get(self, bucket, name, tenant_id=None):
        self.get_calls.append((bucket, name, tenant_id))
        return self.objects.get((bucket, name))

    def obj_exist(self, bucket, name, tenant_id=None):
        return (bucket, name) in self.objects


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

    assert manifest["enumeration_rule_version"] == "enumeration-rules/v7"
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
    assert manifest["enumeration_rule_version"] == "enumeration-rules/v7"
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
