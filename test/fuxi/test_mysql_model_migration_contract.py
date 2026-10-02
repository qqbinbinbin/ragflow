from pathlib import Path
import ast
from contextlib import nullcontext
import inspect
import os
import re
import sys
import types
import uuid

import pytest
import tools.scripts.mysql_migration as migration_module


class _Field:
    def __init__(self, *args, **kwargs):
        pass


class _Model:
    pass


try:
    import peewee  # noqa: F401
    import playhouse.migrate  # noqa: F401
except ImportError:
    peewee = types.ModuleType("peewee")
    for name in (
        "CharField",
        "IntegerField",
        "BigIntegerField",
        "DateTimeField",
        "PrimaryKeyField",
        "TextField",
    ):
        setattr(peewee, name, _Field)
    peewee.Model = _Model
    peewee.MySQLDatabase = type("MySQLDatabase", (), {})
    sys.modules.setdefault("peewee", peewee)

    playhouse = types.ModuleType("playhouse")
    playhouse_migrate = types.ModuleType("playhouse.migrate")
    playhouse_migrate.MySQLMigrator = type("MySQLMigrator", (), {})
    sys.modules.setdefault("playhouse", playhouse)
    sys.modules.setdefault("playhouse.migrate", playhouse_migrate)

from tools.scripts.mysql_migration import (
    MIGRATION_STAGES,
    MigrationConfig,
    MigrationDatabase,
    TabularStructureDiscoveryIndexStage,
    TenantModelContractPreflightStage,
    TenantModelIdMigrationStage,
    TenantModelInstanceStage,
    TenantModelStage,
)


ROOT = Path(__file__).resolve().parents[2]


def test_ocr_add_column_width_matches_current_model_contract():
    source = (ROOT / "api/db/db_models.py").read_text(encoding="utf-8")
    models = migration_module.load_declarative_orm_models(
        source, MigrationDatabase(MigrationConfig()).db, ["tenant"])
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "migrate_db")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "alter_db_add_column"
             and len(node.args) == 4 and isinstance(node.args[1], ast.Constant)
             and isinstance(node.args[2], ast.Constant)
             and (node.args[1].value, node.args[2].value) == ("tenant", "ocr_id")]
    assert len(calls) == 1
    width = next(ast.literal_eval(keyword.value) for keyword in calls[0].args[3].keywords
                 if keyword.arg == "max_length")
    assert width == models["tenant"]._meta.fields["ocr_id"].max_length


def test_declarative_models_compile_defaults_and_composite_keys_without_execution():
    source = '''
raise RuntimeError("model source must not execute")
class BaseModel(Model):
    create_time = BigIntegerField(null=True, index=True)
class Example(DataBaseModel):
    owner = CharField(max_length=32)
    name = CharField(max_length=128, default="")
    config = JSONField(null=False, default=dict)
    class Meta:
        db_table = "example"
        primary_key = CompositeKey("owner", "name")
        indexes = ((("name",), True),)
'''
    database = MigrationDatabase(MigrationConfig()).db
    models = migration_module.load_declarative_orm_models(source, database, ["example"])
    model = models["example"]
    assert model._meta.primary_key.field_names == ("owner", "name")
    assert model._meta.indexes == [(('name',), True)]
    assert model._meta.fields["name"].default == ""
    assert model._meta.fields["config"].default is dict
    sql, params = model._schema._create_table().query()
    assert "LONGTEXT NOT NULL" in sql
    assert "PRIMARY KEY (`owner`, `name`)" in sql
    assert not params


def test_declarative_models_reject_dynamic_schema_expression():
    source = '''
class BaseModel(Model):
    create_time = BigIntegerField(null=True)
class Example(DataBaseModel):
    id = CharField(max_length=fetch_secret(), primary_key=True)
'''
    with pytest.raises(RuntimeError, match="unreviewed_orm_expression"):
        migration_module.load_declarative_orm_models(source, MigrationDatabase(MigrationConfig()).db, ["example"])


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("chat", 1),
        ("embedding", 2),
        ("asr", 4),
        ("speech2text", 4),
        ("vision", 8),
        ("image2text", 8),
        ("rerank", 16),
        ("tts", 32),
        ("ocr", 64),
    ],
)
def test_integer_model_schema_uses_reviewed_bit_flags(model_type, expected):
    assert TenantModelStage.model_type_for_storage(model_type, "int") == expected


def test_unknown_model_type_fails_closed_for_integer_schema():
    with pytest.raises(ValueError, match="unsupported tenant model type"):
        TenantModelStage.model_type_for_storage("unreviewed", "int")


def test_missing_model_instance_aborts_instead_of_silently_skipping():
    records = [(7, "anonymous-model", "provider-id", "embedding", "1", "secret")]
    with pytest.raises(RuntimeError, match="tenant_model_instance mapping is incomplete"):
        TenantModelStage._resolve_instance_ids(records, {})


def test_legacy_numeric_reference_uses_exact_source_model_mapping():
    exact_mapping = {
        ("tenant-a", "7", "embedding"): "new-model-id",
    }
    assert (
        TenantModelIdMigrationStage.resolve_legacy_reference(
            exact_mapping,
            tenant_id="tenant-a",
            legacy_reference=7,
            model_type="embedding",
        )
        == "new-model-id"
    )


def test_missing_or_cross_tenant_legacy_reference_fails_closed():
    exact_mapping = {
        ("tenant-a", "7", "embedding"): "new-model-id",
    }
    with pytest.raises(RuntimeError, match="legacy tenant model reference is unresolved"):
        TenantModelIdMigrationStage.resolve_legacy_reference(
            exact_mapping,
            tenant_id="tenant-b",
            legacy_reference=7,
            model_type="embedding",
        )


def test_legacy_model_name_uses_exact_tenant_provider_and_type_mapping():
    exact_mapping = {
        ("tenant-a", "anonymous-model", "provider-a", "embedding"): "new-model-id",
    }
    assert (
        TenantModelIdMigrationStage.resolve_legacy_model_name(
            exact_mapping,
            tenant_id="tenant-a",
            legacy_model_name="anonymous-model@provider-a",
            model_type="embedding",
        )
        == "new-model-id"
    )
    with pytest.raises(RuntimeError, match="legacy tenant model name is unresolved"):
        TenantModelIdMigrationStage.resolve_legacy_model_name(
            exact_mapping,
            tenant_id="tenant-b",
            legacy_model_name="anonymous-model@provider-a",
            model_type="embedding",
        )


def test_instance_metadata_preserves_endpoint_without_exposing_credentials():
    assert TenantModelInstanceStage.build_instance_extra("https://example.invalid/v1") == (
        '{"base_url": "https://example.invalid/v1"}'
    )


def test_instance_identity_binds_provider_credentials_and_endpoint():
    first = TenantModelInstanceStage.instance_identity(
        "provider-id",
        '{"api_key": "secret", "is_tools": true}',
        "https://first.example.invalid/v1",
    )
    same = TenantModelInstanceStage.instance_identity(
        "provider-id",
        '{"api_key": "secret", "is_tools": false}',
        "https://first.example.invalid/v1",
    )
    other_endpoint = TenantModelInstanceStage.instance_identity(
        "provider-id",
        '{"api_key": "secret", "is_tools": false}',
        "https://second.example.invalid/v1",
    )
    assert first == same
    assert first != other_endpoint


def test_model_metadata_preserves_capacity_and_tool_capability():
    assert TenantModelStage.build_model_extra(
        api_key='{"api_key": "secret", "is_tools": true}',
        max_tokens=4096,
    ) == '{"is_tools": true, "max_tokens": 4096}'


def test_model_metadata_merge_preserves_fields_not_owned_by_legacy_source():
    assert TenantModelStage.merge_model_extra(
        '{"region": "anonymous", "max_tokens": 1}',
        '{"is_tools": true, "max_tokens": 4096}',
    ) == '{"is_tools": true, "max_tokens": 4096, "region": "anonymous"}'


def test_all_enabled_legacy_models_are_migration_candidates():
    condition = TenantModelStage.build_status_condition([])
    assert "tl.status = '1'" in condition
    assert "tl.status = '0'" not in condition


def test_service_entrypoint_migrates_model_contract_before_database_init_and_webserver():
    source = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    startup = source[
        source.index("tools/scripts/run_migrations.sh") :
        source.index('if [[ "${ENABLE_WEBSERVER}" -eq 1 ]]')
    ]
    assert re.search(r"tools/scripts/run_migrations\.sh\s+ensure_db_init", startup)
    assert "INIT_MODEL_PROVIDER_TABLES" not in startup.split("ensure_db_init", 1)[0]


def test_service_migration_does_not_skip_contract_repair_from_version_marker():
    source = (ROOT / "tools" / "scripts" / "run_migrations.sh").read_text(
        encoding="utf-8"
    )
    migration_calls = source.split("tools/scripts/mysql_migration.py")[1:]
    assert migration_calls
    for call in migration_calls:
        if "--mark-database-version" in call:
            continue
        assert "--database-version" not in call


def test_model_preflight_precedes_discovery_writes():
    source = (ROOT / "tools/scripts/run_migrations.sh").read_text(encoding="utf-8")
    assert source.index("--stages tenant_model_contract_preflight ") < source.index(
        "--stages tabular_structure_discovery_index"
    )


@pytest.mark.parametrize("dry_run", [True, False])
def test_foundation_stage_creates_only_missing_table(dry_run):
    stage_class = MIGRATION_STAGES.get("tabular_structure_foundation")
    assert stage_class is not None

    class Database:
        def __init__(self):
            self.statements = []

        def table_exists(self, table):
            return False

        def execute_sql(self, sql, params=None):
            self.statements.append(sql)
            class Cursor:
                def fetchone(self):
                    return ("8.0.40",) if sql == "SELECT VERSION()" else ("ACTIVE",)
            return Cursor()

    database = Database()
    stage = stage_class(database, dry_run=dry_run)
    assert stage.check() is True
    assert stage.execute()[0] == 0
    writes = [sql for sql in database.statements if not sql.startswith("SELECT ")]
    assert len(writes) == (0 if dry_run else 1)
    if not dry_run:
        assert writes[0].lstrip().startswith("CREATE TABLE IF NOT EXISTS tabular_structure_generation")


def test_foundation_stage_precedes_discovery_and_follows_model_preflight():
    source = (ROOT / "tools/scripts/run_migrations.sh").read_text(encoding="utf-8")
    assert source.index("--stages tenant_model_contract_preflight ") < source.index(
        "--stages tabular_structure_foundation"
    ) < source.index("--stages tabular_structure_discovery_index")


def test_foundation_contract_matches_authoritative_orm_fields():
    tree = ast.parse((ROOT / "api/db/db_models.py").read_text(encoding="utf-8"))
    expected = {}
    field_types = {"CharField": "varchar", "BigIntegerField": "bigint",
                   "IntegerField": "int", "DateTimeField": "datetime"}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("BaseModel", "TabularStructureGeneration"):
            for statement in node.body:
                if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
                    continue
                call = statement.value
                if not isinstance(call.func, ast.Name) or call.func.id not in field_types:
                    continue
                options = {key.arg: ast.literal_eval(key.value) for key in call.keywords}
                expected[statement.targets[0].id] = (
                    field_types[call.func.id], options.get("max_length"),
                    "YES" if options.get("null", False) else "NO",
                    "PRI" if options.get("primary_key", False) else "",
                )
    assert len(expected) == 19
    assert MIGRATION_STAGES["tabular_structure_foundation"].COLUMN_CONTRACT == expected


@pytest.fixture
def isolated_model_mysql():
    socket = os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_SOCKET")
    if not socket:
        pytest.skip("explicit isolated MySQL socket is not configured")
    name = "fuxi_model_fixture_" + uuid.uuid4().hex
    admin = MigrationDatabase(MigrationConfig(database="mysql"))
    admin.db.connect_params["unix_socket"] = socket
    admin.connect()
    database = None
    try:
        admin.execute_sql(f"CREATE DATABASE `{name}` CHARACTER SET utf8mb4")
        database = MigrationDatabase(MigrationConfig(database=name))
        database.db.connect_params["unix_socket"] = socket
        database.connect()
        yield database
    finally:
        if database is not None:
            database.close()
        admin.execute_sql(f"DROP DATABASE IF EXISTS `{name}`")
        admin.close()


def test_complete_legacy_model_chain_preserves_owners_endpoints_and_identities(isolated_model_mysql):
    db = isolated_model_mysql
    db.execute_sql("CREATE TABLE tenant_llm (id INT PRIMARY KEY, tenant_id VARCHAR(32), "
                   "llm_factory VARCHAR(128), llm_name VARCHAR(128), model_type VARCHAR(16), "
                   "api_key TEXT, api_base TEXT, max_tokens INT, status CHAR(1))")
    db.execute_sql("CREATE TABLE tenant (id VARCHAR(32) PRIMARY KEY, tenant_llm_id INT, "
                   "tenant_embd_id INT, tenant_rerank_id INT)")
    db.execute_sql("CREATE TABLE knowledgebase (id VARCHAR(32) PRIMARY KEY, tenant_id VARCHAR(32), "
                   "status CHAR(1), embd_id VARCHAR(256), tenant_embd_id INT)")
    db.execute_sql("CREATE TABLE dialog (id VARCHAR(32) PRIMARY KEY, tenant_id VARCHAR(32), tenant_llm_id INT)")
    db.execute_sql("CREATE TABLE document (id VARCHAR(32) PRIMARY KEY, kb_id VARCHAR(32))")
    db.execute_sql("CREATE TABLE file (id VARCHAR(32) PRIMARY KEY)")
    models = [
        (1, "tenant-a", "OpenAI-API-Compatible", "bge-large-zh-v1.5", "embedding", "customer", "https://customer.example.invalid/v1", 8192, "1"),
        (2, "tenant-a", "OpenAI-API-Compatible", "BAAI/bge-reranker-v2-m3", "rerank", "rerank", "https://rerank.example.invalid/v1", 8192, "1"),
        (3, "tenant-a", "OpenAI-API-Compatible", "qwen3.6-plus", "chat", "chat", "https://chat.example.invalid/v1", 8192, "1"),
        (4, "tenant-a", "OpenAI-API-Compatible", "text-embedding-v3", "embedding", "chat", "https://chat.example.invalid/v1", 8192, "1"),
        (5, "tenant-a", "Tongyi-Qianwen", "qwen-turbo", "chat", "chat", "https://chat.example.invalid/v1", 8192, "1"),
        (6, "tenant-b", "OpenAI-API-Compatible", "bge-large-zh-v1.5", "embedding", "disabled", "https://customer.example.invalid/v1", 8192, "0"),
    ]
    for row in models:
        db.execute_sql("INSERT INTO tenant_llm VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", row)
    db.execute_sql("INSERT INTO tenant VALUES ('tenant-a',3,1,2),('tenant-b',NULL,NULL,NULL)")
    db.execute_sql("INSERT INTO knowledgebase VALUES "
                   "('kb-active','tenant-a','1','bge-large-zh-v1.5@OpenAI-API-Compatible',1),"
                   "('kb-disabled','tenant-a','0','BAAI/bge-large-zh-v1.5@OpenAI-API-Compatible',NULL)")
    db.execute_sql("INSERT INTO dialog VALUES ('dialog-a','tenant-a',3)")
    db.execute_sql("INSERT INTO document VALUES ('doc-a','kb-active'),('doc-b','kb-active')")
    db.execute_sql("INSERT INTO file VALUES ('file-a'),('file-b')")
    def identities():
        return {table: db.execute_sql(f"SELECT id FROM `{table}` ORDER BY id").fetchall()
                for table in ("tenant", "knowledgebase", "document", "file", "tenant_llm")}
    baseline = identities()
    legacy_rows = db.execute_sql("SELECT * FROM tenant_llm ORDER BY id").fetchall()
    before_tables = db.execute_sql("SHOW TABLES").fetchall()
    with pytest.raises(RuntimeError, match="legacy tenant model name is unresolved"):
        TenantModelContractPreflightStage(db, dry_run=False).execute()
    assert db.execute_sql("SHOW TABLES").fetchall() == before_tables
    assert identities() == baseline
    # Synthetic fixture decision only; this does not authorize a customer binding change.
    db.execute_sql("UPDATE knowledgebase SET embd_id=%s WHERE id='kb-disabled'",
                   ("bge-large-zh-v1.5@OpenAI-API-Compatible",))
    stages = ["tenant_model_contract_preflight", "tenant_model_provider", "tenant_model_instance",
              "tenant_model", "tenant_model_id_migration"]
    snapshots = []
    for _ in range(2):
        for name in stages:
            stage = MIGRATION_STAGES[name](db, dry_run=False)
            if stage.check():
                stage.execute()
        snapshots.append({table: db.execute_sql(f"SELECT * FROM `{table}` ORDER BY id").fetchall()
                          for table in ("tenant_model_provider", "tenant_model_instance", "tenant_model", "tenant", "knowledgebase", "dialog")})
    assert snapshots[0] == snapshots[1]
    assert identities() == baseline
    assert db.execute_sql("SELECT * FROM tenant_llm ORDER BY id").fetchall() == legacy_rows
    assert db.execute_sql("SELECT COUNT(*) FROM tenant_model").fetchone()[0] == 5
    binding = db.execute_sql("SELECT k.status,p.tenant_id,m.model_name,i.extra "
                            "FROM knowledgebase k JOIN tenant_model m ON m.id=k.tenant_embd_id "
                            "JOIN tenant_model_provider p ON p.id=m.provider_id "
                            "JOIN tenant_model_instance i ON i.id=m.instance_id "
                            "WHERE k.id='kb-disabled'").fetchone()
    assert binding[:3] == ("0", "tenant-a", "bge-large-zh-v1.5")
    assert "https://customer.example.invalid/v1" in binding[3]


def test_all_reviewed_missing_tables_preserve_primary_and_unique_indexes_in_mysql(isolated_model_mysql):
    db = isolated_model_mysql
    names = ["chat_channel", "compilation_template", "compilation_template_group",
             "file_commit", "file_commit_item", "tabular_structure_generation", "tenant_model",
             "tenant_model_group", "tenant_model_group_mapping", "tenant_model_instance", "tenant_model_provider"]
    source = (ROOT / "api/db/db_models.py").read_text(encoding="utf-8")
    models = migration_module.load_declarative_orm_models(source, db.db, names)
    db.execute_sql("CREATE TABLE protected_business (id VARCHAR(32) PRIMARY KEY)")
    db.execute_sql("INSERT INTO protected_business VALUES ('preserve-me')")
    for _ in range(2):
        for name in names:
            models[name].create_table(safe=True)
    for name, model in models.items():
        assert {column.name for column in db.db.get_columns(name)} == set(model._meta.fields)
        primary = model._meta.primary_key
        expected_primary = list(primary.field_names) if hasattr(primary, "field_names") else [primary.name]
        assert db.db.get_primary_keys(name) == expected_primary
        actual_unique = {tuple(index.columns) for index in db.db.get_indexes(name) if index.unique}
        for columns, unique in model._meta.indexes:
            if unique:
                assert tuple(columns) in actual_unique
        assert db.execute_sql(f"SELECT COUNT(*) FROM `{name}`").fetchone()[0] == 0
    assert list(db.execute_sql("SELECT id FROM protected_business").fetchall()) == [("preserve-me",)]
    db.execute_sql("INSERT INTO file_commit_item (id,commit_id,file_id,operation) VALUES ('a','commit','file','add')")
    with pytest.raises(peewee.IntegrityError):
        db.execute_sql("INSERT INTO file_commit_item (id,commit_id,file_id,operation) VALUES ('b','commit','file','modify')")


def test_reviewed_additive_fields_defaults_indexes_and_drift_in_mysql(isolated_model_mysql):
    db = isolated_model_mysql
    source = (ROOT / "api/db/db_models.py").read_text(encoding="utf-8")
    columns = {"knowledgebase": [f"{kind}_task_{suffix}" for kind in
               ("artifact", "skill", "structure_graph", "structure_mindmap", "timeline",
                "session_graph", "session_essence", "structure") for suffix in ("id", "finish_at")],
               "tenant": ["ocr_id", "tenant_ocr_id"], "sync_logs": ["task_type"], "user_canvas": ["tags"]}
    timestamps = ("create_time", "create_date", "update_time", "update_date")
    indexes = {name: [(field,) for field in timestamps] for name in
               ("tabular_structure_dataset_index_state", "tabular_structure_table_index")}
    selected = {**columns, **{table: list(timestamps) for table in indexes}}
    models = migration_module.load_declarative_orm_models(source, db.db, list(selected), field_names=selected)
    for table in columns:
        db.execute_sql(f"CREATE TABLE `{table}` (id VARCHAR(32) PRIMARY KEY)")
        db.execute_sql(f"INSERT INTO `{table}` VALUES ('preserved')")
    for table in indexes:
        db.execute_sql(f"CREATE TABLE `{table}` (id VARCHAR(32) PRIMARY KEY, create_time BIGINT NULL, "
                       "create_date DATETIME NULL, update_time BIGINT NULL, update_date DATETIME NULL)")
    execute = migration_module.apply_reviewed_additive_delta
    dry = execute(db.db, models, columns, indexes, dry_run=True)
    assert dry == {"columns_added": 20, "indexes_added": 8}
    assert len(db.db.get_columns("tenant")) == 1
    assert execute(db.db, models, columns, indexes) == dry
    assert execute(db.db, models, columns, indexes) == {"columns_added": 0, "indexes_added": 0}
    for table, fields in columns.items():
        assert set(fields) <= {column.name for column in db.db.get_columns(table)}
        assert db.execute_sql(f"SELECT id FROM `{table}`").fetchone() == ("preserved",)
    assert db.execute_sql("SELECT task_type FROM sync_logs").fetchone() == ("sync",)
    assert db.execute_sql("SELECT tags FROM user_canvas").fetchone() == ("",)
    assert db.execute_sql("SELECT ocr_id, tenant_ocr_id FROM tenant").fetchone() == (None, None)
    for table, fields in indexes.items():
        actual = {tuple(index.columns) for index in db.db.get_indexes(table) if not index.unique}
        assert set(fields) <= actual
    # Queue an earlier missing field, then reject a later drift before any DDL.
    db.execute_sql("ALTER TABLE knowledgebase DROP COLUMN artifact_task_finish_at")
    db.execute_sql("ALTER TABLE tenant MODIFY ocr_id VARCHAR(128) NULL")
    with pytest.raises(RuntimeError, match="reviewed_column_contract_mismatch"):
        execute(db.db, models, columns, indexes)
    assert "artifact_task_finish_at" not in {column.name for column in db.db.get_columns("knowledgebase")}
    assert db.execute_sql("SELECT id FROM tenant").fetchone() == ("preserved",)


@pytest.mark.parametrize("drift", [False, True])
def test_foundation_existing_schema_is_noop_or_rejected_without_ddl(drift):
    stage_class = MIGRATION_STAGES["tabular_structure_foundation"]

    class Cursor:
        def fetchall(self):
            rows = [(name, *contract) for name, contract in stage_class.COLUMN_CONTRACT.items()]
            return rows[:-1] if drift else rows

    class Database:
        config = type("Config", (), {"database": "anonymous"})()

        def table_exists(self, table):
            return True

        def execute_sql(self, sql, params=None):
            assert sql.startswith("SELECT ")
            if sql == "SELECT VERSION()" or "INFORMATION_SCHEMA.PLUGINS" in sql:
                class BackendCursor:
                    def fetchone(self):
                        return ("8.0.40",) if sql == "SELECT VERSION()" else ("ACTIVE",)
                return BackendCursor()
            return Cursor()

    stage = stage_class(Database(), dry_run=False)
    if drift:
        with pytest.raises(RuntimeError, match="foundation_schema_mismatch"):
            stage.execute()
    else:
        assert stage.check() is False
        assert stage.execute()[0] == 0


@pytest.mark.parametrize("method", ["check", "execute"])
@pytest.mark.parametrize("missing", TabularStructureDiscoveryIndexStage.source_tables)
def test_discovery_missing_dependency_rejects_before_any_sql(method, missing):
    class Database:
        def table_exists(self, table):
            return table != missing

        def execute_sql(self, sql, params=None):
            raise AssertionError("SQL reached before missing dependency rejection: " + sql)

    stage = TabularStructureDiscoveryIndexStage(Database(), dry_run=False)
    with pytest.raises(RuntimeError, match="discovery_missing_source_table:" + missing):
        getattr(stage, method)()


def test_contract_preflight_rejects_cross_tenant_legacy_reference():
    source_by_id, source_by_name = TenantModelContractPreflightStage.build_source_maps(
        [
            (7, "tenant-a", "provider-a", "anonymous-model", "embedding"),
        ]
    )

    with pytest.raises(RuntimeError, match="legacy tenant model reference is unresolved"):
        TenantModelContractPreflightStage.validate_reference(
            source_by_id=source_by_id,
            source_by_name=source_by_name,
            current_model_ids=set(),
            tenant_id="tenant-b",
            current_reference=7,
            legacy_model_name="anonymous-model@provider-a",
            model_type="embedding",
        )


def test_contract_preflight_rejects_ambiguous_legacy_model_name():
    with pytest.raises(RuntimeError, match="legacy tenant model name mapping is ambiguous"):
        TenantModelContractPreflightStage.build_source_maps(
            [
                (7, "tenant-a", "provider-a", "anonymous-model", "embedding"),
                (8, "tenant-a", "provider-a", "anonymous-model", "embedding"),
            ]
        )


def test_service_migration_preflights_references_before_any_write_stage():
    source = (ROOT / "tools" / "scripts" / "run_migrations.sh").read_text(
        encoding="utf-8"
    )
    stage_lists = re.findall(r"--stages\s+([^\s\\]+)", source)
    stages = next(
        value.split(",")
        for value in stage_lists
        if "tenant_model_provider" in value
    )
    assert stages == [
        "tenant_model_contract_preflight",
        "tenant_model_provider",
        "tenant_model_instance",
        "tenant_model",
        "tenant_model_id_migration",
    ]


def test_tabular_structure_discovery_index_uses_a_formal_mysql_ngram_migration_stage():
    assert "tabular_structure_discovery_index" in MIGRATION_STAGES
    stage = MIGRATION_STAGES["tabular_structure_discovery_index"]
    source = inspect.getsource(stage)

    assert "tabular_structure_dataset_index_state" in source
    assert "tabular_structure_table_index" in source
    assert "WITH PARSER ngram" in source
    assert "FULLTEXT" in source
    assert "SELECT VERSION()" in source
    assert "discovery_unsupported_backend" in source


def test_tabular_structure_discovery_index_preserves_full_opaque_table_identity():
    source = inspect.getsource(TabularStructureDiscoveryIndexStage)
    model_source = (ROOT / "api" / "db" / "db_models.py").read_text(
        encoding="utf-8"
    )
    service_source = (
        ROOT / "api" / "db" / "services" / "tabular_structure_service.py"
    ).read_text(encoding="utf-8")

    assert "table_ref VARCHAR(512) CHARACTER SET ascii COLLATE ascii_bin" in source
    assert "table_ref VARCHAR(96)" not in source
    assert "tabular-structure-index/v2" in source
    assert re.search(
        r"table_ref\s*=\s*CharField\(\s*max_length=512",
        model_source,
    )
    assert 'SQL("CHARACTER SET ascii")' in model_source
    assert '"ascii_bin" if settings.DATABASE_TYPE.lower() == "mysql"' in model_source
    assert 'settings.DATABASE_TYPE.lower() == "mysql"' in model_source
    assert (
        'TABULAR_DISCOVERY_INDEX_SCHEMA_VERSION = "tabular-structure-index/v2"'
        in service_source
    )


def test_discovery_migration_repairs_truncated_table_identity_before_backfill():
    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Database:
        config = type("Config", (), {"database": "anonymous"})()

        def __init__(self):
            self.statements = []

        def table_exists(self, table):
            return True

        def atomic(self):
            return nullcontext()

        def execute_sql(self, sql, params=None):
            normalized = " ".join(sql.split())
            self.statements.append((normalized, params))
            if sql == "SELECT VERSION()":
                return Cursor(("8.0.40",))
            if "INFORMATION_SCHEMA.PLUGINS" in sql:
                return Cursor(("ACTIVE",))
            if "information_schema.columns" in sql:
                return Cursor(("varchar", 96, "utf8mb4", "utf8mb4_0900_ai_ci"))
            if "information_schema.statistics" in sql:
                return Cursor((1,))
            if "LEFT JOIN tabular_structure_dataset_index_state" in sql:
                return Cursor(None)
            return Cursor(None)

    database = Database()
    stage = TabularStructureDiscoveryIndexStage(database, dry_run=False)

    assert stage.check() is True
    stage.execute()

    executed = "\n".join(sql for sql, _params in database.statements)
    assert "with self.db.atomic()" in inspect.getsource(
        TabularStructureDiscoveryIndexStage.execute
    )
    assert (
        "ALTER TABLE tabular_structure_table_index MODIFY table_ref VARCHAR(512) "
        "CHARACTER SET ascii COLLATE ascii_bin NOT NULL" in executed
    )
    assert "DELETE FROM tabular_structure_table_index" in executed
    assert "backfill_status = 'pending'" in executed
    assert "backfill_cursor = NULL" in executed
    assert "index_schema_version = 'tabular-structure-index/v2'" in executed


def test_discovery_migration_is_idempotent_after_identity_contract_repair():
    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Database:
        config = type("Config", (), {"database": "anonymous"})()

        def table_exists(self, table):
            return True

        def execute_sql(self, sql, params=None):
            if sql == "SELECT VERSION()":
                return Cursor(("8.0.40",))
            if "INFORMATION_SCHEMA.PLUGINS" in sql:
                return Cursor(("ACTIVE",))
            if "information_schema.columns" in sql:
                return Cursor(("varchar", 512, "ascii", "ascii_bin"))
            if "information_schema.statistics" in sql:
                return Cursor((1,))
            if "index_schema_version <>" in sql:
                return Cursor(None)
            if "LEFT JOIN tabular_structure_dataset_index_state" in sql:
                return Cursor(None)
            raise AssertionError(sql)

    stage = TabularStructureDiscoveryIndexStage(Database(), dry_run=False)

    assert stage.check() is False


def test_discovery_migration_reprojects_when_index_table_is_missing_but_state_exists():
    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Database:
        config = type("Config", (), {"database": "anonymous"})()

        def __init__(self):
            self.statements = []

        def table_exists(self, table):
            return table != "tabular_structure_table_index"

        def atomic(self):
            return nullcontext()

        def execute_sql(self, sql, params=None):
            normalized = " ".join(sql.split())
            self.statements.append((normalized, params))
            if sql == "SELECT VERSION()":
                return Cursor(("8.0.40",))
            if "INFORMATION_SCHEMA.PLUGINS" in sql:
                return Cursor(("ACTIVE",))
            if "index_schema_version <>" in sql:
                return Cursor(None)
            if "LEFT JOIN tabular_structure_dataset_index_state" in sql:
                return Cursor(None)
            if "information_schema.statistics" in sql:
                return Cursor((0,))
            return Cursor(None)

    database = Database()
    stage = TabularStructureDiscoveryIndexStage(database, dry_run=False)

    stage.execute()

    executed = "\n".join(sql for sql, _params in database.statements)
    assert "DELETE FROM tabular_structure_table_index" in executed
    assert "backfill_status = 'pending'" in executed


@pytest.mark.skipif(
    not (os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_PASSWORD") or os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_SOCKET")),
    reason="explicit isolated MySQL integration target is not configured",
)
def test_discovery_identity_migration_round_trips_opaque_refs_in_mysql():
    database_name = f"adr039_table_ref_{uuid.uuid4().hex}"
    password = os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_PASSWORD", "")
    socket = os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_SOCKET")
    host = os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_HOST", "mysql")
    port = int(os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_PORT", "3306"))
    user = os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_USER", "root")
    admin = MigrationDatabase(
        MigrationConfig(host=host, port=port, user=user, password=password, database="mysql")
    )
    target = None
    if socket:
        admin.db.connect_params["unix_socket"] = socket
    admin.connect()
    try:
        admin.execute_sql(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        target = MigrationDatabase(
            MigrationConfig(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database_name,
            )
        )
        if socket:
            target.db.connect_params["unix_socket"] = socket
        target.connect()
        for table in ("document", "knowledgebase"):
            target.execute_sql(f"CREATE TABLE `{table}` (id VARCHAR(32) PRIMARY KEY) ENGINE=InnoDB")
        foundation = MIGRATION_STAGES["tabular_structure_foundation"](target, dry_run=False)
        assert foundation.check() is True
        foundation.execute()
        assert foundation.check() is False
        foundation.execute()
        target.execute_sql(
            "CREATE TABLE tabular_structure_dataset_index_state ("
            "tenant_id VARCHAR(32) NOT NULL, kb_id VARCHAR(256) NOT NULL, "
            "index_revision BIGINT UNSIGNED NOT NULL DEFAULT 1, "
            "backfill_status VARCHAR(16) NOT NULL DEFAULT 'complete', "
            "backfill_cursor VARCHAR(32) NULL, index_schema_version VARCHAR(64) NOT NULL, "
            "create_time BIGINT NULL, create_date DATETIME NULL, update_time BIGINT NULL, "
            "update_date DATETIME NULL, PRIMARY KEY (tenant_id, kb_id)) "
            "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        )
        target.execute_sql(
            "CREATE TABLE tabular_structure_table_index ("
            "tenant_id VARCHAR(32) NOT NULL, kb_id VARCHAR(256) NOT NULL, "
            "document_id VARCHAR(32) NOT NULL, producer_generation_ref VARCHAR(36) NOT NULL, "
            "table_ref VARCHAR(96) NOT NULL, table_ordinal INT UNSIGNED NOT NULL, "
            "search_text TEXT NOT NULL, identity_hash CHAR(64) NOT NULL, "
            "index_revision BIGINT UNSIGNED NOT NULL, active BOOLEAN NOT NULL DEFAULT TRUE, "
            "projection_status VARCHAR(16) NOT NULL DEFAULT 'safe', unsafe_reason VARCHAR(64) NULL, "
            "create_time BIGINT NULL, create_date DATETIME NULL, update_time BIGINT NULL, "
            "update_date DATETIME NULL, "
            "PRIMARY KEY (tenant_id, kb_id, document_id, producer_generation_ref, table_ref), "
            "INDEX idx_tabular_structure_dataset_revision "
            "(tenant_id, kb_id, active, index_revision), "
            "INDEX idx_tabular_structure_document (document_id), "
            "INDEX idx_tabular_structure_identity (identity_hash), "
            "FULLTEXT INDEX ft_tabular_structure_search_text (search_text) WITH PARSER ngram) "
            "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        )
        target.execute_sql(
            "INSERT INTO tabular_structure_generation "
            "(producer_generation_ref,tenant_id,kb_id,document_id,projection_version,"
            "producer_schema_version,manifest_object_name,manifest_sha256,source_sha256,"
            "row_count,part_count,status) VALUES "
            "('11111111-1111-1111-1111-111111111111','tenant','dataset','document',"
            "'v1','v1','anonymous.json',REPEAT('a',64),REPEAT('b',64),1,1,'active')"
        )
        target.execute_sql(
            "INSERT INTO tabular_structure_dataset_index_state VALUES "
            "('tenant','dataset',1,'complete',NULL,'tabular-structure-index/v1',NULL,NULL,NULL,NULL)"
        )
        target.execute_sql(
            "INSERT INTO tabular_structure_table_index "
            "(tenant_id,kb_id,document_id,producer_generation_ref,table_ref,table_ordinal,"
            "search_text,identity_hash,index_revision) VALUES "
            "('tenant','dataset','document','11111111-1111-1111-1111-111111111111',"
            "REPEAT('a',96),1,'anonymous',REPEAT('b',64),1)"
        )

        stage = TabularStructureDiscoveryIndexStage(target, dry_run=False)
        assert stage.check() is True
        stage.execute()

        contract = target.execute_sql(
            "SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, CHARACTER_SET_NAME, COLLATION_NAME "
            "FROM information_schema.columns WHERE table_schema=%s "
            "AND table_name='tabular_structure_table_index' AND column_name='table_ref'",
            (database_name,),
        ).fetchone()
        assert contract == ("varchar", 512, "ascii", "ascii_bin")
        assert target.execute_sql(
            "SELECT COUNT(*) FROM tabular_structure_table_index"
        ).fetchone()[0] == 0
        assert target.execute_sql(
            "SELECT index_revision, backfill_status, backfill_cursor, index_schema_version "
            "FROM tabular_structure_dataset_index_state"
        ).fetchone() == (2, "pending", None, "tabular-structure-index/v2")

        refs = ["tbl_v2_" + "a" * 64 + "_" + "b" * 64, "x" * 512]
        for ordinal, table_ref in enumerate(refs, 1):
            target.execute_sql(
                "INSERT INTO tabular_structure_table_index "
                "(tenant_id,kb_id,document_id,producer_generation_ref,table_ref,table_ordinal,"
                "search_text,identity_hash,index_revision) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    "tenant",
                    "dataset",
                    "document",
                    "11111111-1111-1111-1111-111111111111",
                    table_ref,
                    ordinal,
                    "anonymous",
                    str(ordinal) * 64,
                    2,
                ),
            )
        assert [
            row[0]
            for row in target.execute_sql(
                "SELECT table_ref FROM tabular_structure_table_index ORDER BY table_ordinal"
            ).fetchall()
        ] == refs
        assert stage.check() is False
    finally:
        if target is not None:
            target.close()
        admin.execute_sql(f"DROP DATABASE IF EXISTS `{database_name}`")
        admin.close()


def test_discovery_migration_recovers_when_active_generations_lack_index_state():
    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Database:
        config = type("Config", (), {"database": "anonymous"})()

        def table_exists(self, table):
            return True

        def execute_sql(self, sql, params=None):
            if sql == "SELECT VERSION()":
                return Cursor(("8.0.40",))
            if "INFORMATION_SCHEMA.PLUGINS" in sql:
                return Cursor(("ACTIVE",))
            if "information_schema.columns" in sql:
                return Cursor(("varchar", 512, "ascii", "ascii_bin"))
            if "information_schema.statistics" in sql:
                return Cursor((1,))
            if "index_schema_version <>" in sql:
                return Cursor(None)
            if "LEFT JOIN tabular_structure_dataset_index_state" in sql:
                return Cursor((1,))
            raise AssertionError(sql)

    stage = TabularStructureDiscoveryIndexStage(Database(), dry_run=True)

    assert stage.check() is True


def test_tabular_structure_discovery_migration_runs_before_backend_start():
    source = (ROOT / "tools" / "scripts" / "run_migrations.sh").read_text(
        encoding="utf-8",
    )

    assert "--stages tabular_structure_discovery_index" in source
    assert "--backfill-tabular-structure-index" in source
    ddl = source.index("--stages tabular_structure_discovery_index")
    backfill = source.index("--backfill-tabular-structure-index")
    model_preflight = source.index("--stages tenant_model_contract_preflight ")
    model_contract = source.index("--stages tenant_model_contract_preflight,")
    version_marker = source.index("--mark-database-version")
    assert model_preflight < ddl < backfill < model_contract < version_marker
    assert source.count('--config "$CONFIG"') >= 3


def test_tabular_structure_backfill_cli_uses_the_service_layer_not_sql_reconstruction():
    source = (ROOT / "tools" / "scripts" / "mysql_migration.py").read_text(
        encoding="utf-8",
    )

    assert '"--backfill-tabular-structure-index"' in source
    assert "backfill_active_generation_indexes" in source
    assert "STORAGE_IMPL" in source
    assert 'DB.lock("tabular_structure_discovery_index_backfill"' in source
    assert "load_tabular_structure_projection" not in inspect.getsource(
        MIGRATION_STAGES["tabular_structure_discovery_index"]
    )
