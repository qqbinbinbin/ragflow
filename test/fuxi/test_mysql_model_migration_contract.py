from pathlib import Path
from contextlib import nullcontext
import inspect
import os
import re
import sys
import types
import uuid

import pytest


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
        if "tenant_model_contract_preflight" in value
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
    not os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_PASSWORD"),
    reason="explicit isolated MySQL integration target is not configured",
)
def test_discovery_identity_migration_round_trips_opaque_refs_in_mysql():
    database_name = f"adr039_table_ref_{uuid.uuid4().hex}"
    password = os.environ["FUXI_ADR039_MYSQL_INTEGRATION_PASSWORD"]
    host = os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_HOST", "mysql")
    port = int(os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_PORT", "3306"))
    user = os.getenv("FUXI_ADR039_MYSQL_INTEGRATION_USER", "root")
    admin = MigrationDatabase(
        MigrationConfig(host=host, port=port, user=user, password=password, database="mysql")
    )
    target = None
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
        target.connect()
        target.execute_sql(
            "CREATE TABLE tabular_structure_generation ("
            "producer_generation_ref VARCHAR(36) PRIMARY KEY, tenant_id VARCHAR(32) NOT NULL, "
            "kb_id VARCHAR(256) NOT NULL, document_id VARCHAR(32) NOT NULL, "
            "status VARCHAR(16) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        )
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
            "INSERT INTO tabular_structure_generation VALUES "
            "('11111111-1111-1111-1111-111111111111','tenant','dataset','document','active')"
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
    model_contract = source.index("--stages tenant_model_contract_preflight")
    version_marker = source.index("--mark-database-version")
    assert ddl < backfill < model_contract < version_marker
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
