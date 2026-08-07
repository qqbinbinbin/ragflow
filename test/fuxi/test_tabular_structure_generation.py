import ast
import copy
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import pytest

from rag.app import tabular_structure
from test.fuxi.test_table_semantic_rows import _load_table_module
from test.fuxi.test_tabular_structure_projection import _workbook_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_MODULE_PATH = REPO_ROOT / "api" / "db" / "services" / "tabular_structure_service.py"


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


def _stored_generation(table_parser, *, generation_ref=None, rows_per_part=2):
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
        document_id="document-1",
        projection=projection,
        rows_per_part=rows_per_part,
        tenant_id="tenant-owner",
    )
    return storage, projection, receipt


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

    assert manifest["enumeration_rule_version"] == "enumeration-rules/v4"
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
    assert manifest["enumeration_rule_version"] == "enumeration-rules/v4"
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
