from pathlib import Path
import re
import sys
import types

import pytest


class _Field:
    def __init__(self, *args, **kwargs):
        pass


class _Model:
    pass


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


def test_service_entrypoint_runs_model_migration_before_starting_webserver():
    source = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    startup = source[source.index("ensure_db_init") : source.index('if [[ "${ENABLE_WEBSERVER}" -eq 1 ]]')]
    assert re.search(r"ensure_db_init\s+tools/scripts/run_migrations\.sh", startup)
    assert "INIT_MODEL_PROVIDER_TABLES" not in startup.split("tools/scripts/run_migrations.sh", 1)[0]


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
    stage_list = re.search(r"--stages\s+([^\s\\]+)", source)
    assert stage_list is not None
    stages = stage_list.group(1).split(",")
    assert stages == [
        "tenant_model_contract_preflight",
        "tenant_model_provider",
        "tenant_model_instance",
        "tenant_model",
        "tenant_model_id_migration",
    ]
