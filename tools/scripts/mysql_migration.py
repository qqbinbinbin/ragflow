#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
MySQL Data Migration Script

This script provides a flexible MySQL data migration tool that supports:
1. MySQL configuration via config file or command line arguments
2. Direct peewee operations without importing api.db.services
3. Configurable migration stages via command line
4. Migration logging with table names, row counts, and duration
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid

from packaging.version import InvalidVersion, Version
from peewee import (
    CharField,
    IntegerField,
    BigIntegerField,
    DateTimeField,
    MySQLDatabase,
    Model,
    PrimaryKeyField,
    TextField,
)
from playhouse.migrate import MySQLMigrator

# Add project root to path for imports
PROJECT_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_BASE)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MIGRATION_DB_VERSION_MARKER = "mysql_migration.database.version"


class MigrationConfig:
    """Configuration for MySQL connection"""

    def __init__(self, host: str = "localhost", port: int = 3306, user: str = "root", password: str = "", database: str = "rag_flow"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    @classmethod
    def from_config_file(cls, config_path: str) -> "MigrationConfig":
        """Load configuration from YAML config file"""
        try:
            from ruamel.yaml import YAML

            yaml = YAML(typ="safe", pure=True)

            with open(config_path, "r") as f:
                config = yaml.load(f)

            # Try to get database config
            db_config = config.get("database", config.get("mysql", {}))

            return cls(
                host=db_config.get("host", "localhost"),
                port=db_config.get("port", 3306),
                user=db_config.get("user", "root"),
                password=db_config.get("password", ""),
                database=db_config.get("name", db_config.get("database", "rag_flow")),
            )
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}, using defaults")
            return cls()


class MigrationStats:
    """Track migration statistics"""

    def __init__(self):
        self.tables_operated = []
        self.rows_processed = 0
        self.start_time = None
        self.end_time = None
        self.stage_stats = []

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()

    def add_stage_stats(self, stage_name: str, tables: list, rows: int, duration: float):
        self.stage_stats.append({"stage": stage_name, "tables": tables, "rows": rows, "duration": duration})
        self.tables_operated.extend(tables)
        self.rows_processed += rows

    def print_summary(self):
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        logger.info("=" * 60)
        logger.info("Migration Summary")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {duration:.2f}s")
        logger.info(f"Total Rows Processed: {self.rows_processed}")
        logger.info(f"Tables Operated: {', '.join(set(self.tables_operated))}")
        logger.info("-" * 60)
        logger.info("Stage Details:")
        for stat in self.stage_stats:
            logger.info(f"  [{stat['stage']}] Tables: {', '.join(stat['tables'])}, Rows: {stat['rows']}, Duration: {stat['duration']:.2f}s")
        logger.info("=" * 60)


class MigrationDatabase:
    """Database wrapper for migrations"""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.db = MySQLDatabase(config.database, host=config.host, port=config.port, user=config.user, password=config.password, charset="utf8mb4")
        self.migrator = MySQLMigrator(self.db)

    def connect(self):
        self.db.connect()
        logger.info(f"Connected to MySQL database: {self.config.database}")

    def close(self):
        if not self.db.is_closed():
            self.db.close()
            logger.info("Database connection closed")

    def execute_sql(self, sql: str, params=None):
        return self.db.execute_sql(sql, params)

    def atomic(self):
        return self.db.atomic()

    def table_exists(self, table_name: str) -> bool:
        cursor = self.execute_sql("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s AND table_name = %s", (self.config.database, table_name))
        return cursor.fetchone()[0] > 0

    def column_exists(self, table_name: str, column_name: str) -> bool:
        cursor = self.execute_sql("SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = %s AND table_name = %s AND column_name = %s", (self.config.database, table_name, column_name))
        return cursor.fetchone()[0] > 0

    def get_column_type(self, table_name: str, column_name: str) -> str | None:
        """Get the DATA_TYPE of a column from information_schema, returns None if column does not exist"""
        cursor = self.execute_sql("SELECT DATA_TYPE FROM information_schema.columns WHERE table_schema = %s AND table_name = %s AND column_name = %s", (self.config.database, table_name, column_name))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_system_setting_value(self, name: str) -> str | None:
        if not self.table_exists("system_settings"):
            logger.info("Table 'system_settings' does not exist, migration marker is unavailable")
            return None
        cursor = self.execute_sql(
            "SELECT `value` FROM `system_settings` WHERE `name` = %s",
            (name,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def upsert_system_setting(self, name: str, value: str, source: str = "migration", data_type: str = "string"):
        if not self.table_exists("system_settings"):
            logger.warning("Table 'system_settings' does not exist, migration marker was not saved")
            return

        current_ts = int(time.time())
        self.execute_sql(
            """
            INSERT INTO `system_settings`
            (`name`, `source`, `data_type`, `value`, `create_time`, `create_date`, `update_time`, `update_date`)
            VALUES (%s, %s, %s, %s, %s, FROM_UNIXTIME(%s), %s, FROM_UNIXTIME(%s))
            ON DUPLICATE KEY UPDATE
              `source` = VALUES(`source`),
              `data_type` = VALUES(`data_type`),
              `value` = VALUES(`value`),
              `update_time` = VALUES(`update_time`),
              `update_date` = VALUES(`update_date`)
            """,
            (
                name,
                source,
                data_type,
                value,
                current_ts * 1000,
                current_ts,
                current_ts * 1000,
                current_ts,
            ),
        )

    def get_database_version(self) -> str | None:
        return self.get_system_setting_value(MIGRATION_DB_VERSION_MARKER)

    def set_database_version(self, version: str):
        self.upsert_system_setting(MIGRATION_DB_VERSION_MARKER, version)


def parse_migration_version(version: str | None) -> Version | None:
    if not version:
        return None
    normalized = version.strip()
    if normalized.startswith(("v", "V")):
        normalized = normalized[1:]
    try:
        return Version(normalized)
    except InvalidVersion:
        logger.warning("Invalid migration version format: %s", version)
        return None


def should_skip_migration(current_db_version: str | None, target_version: str) -> bool:
    current = parse_migration_version(current_db_version)
    target = parse_migration_version(target_version)
    if current is None or target is None:
        return False
    return current >= target


# Define model classes for migration (not importing from api.db.db_models)
class BaseModel(Model):
    """Base model for migration tables"""

    create_time = BigIntegerField(null=True, index=True)
    create_date = DateTimeField(null=True, index=True)
    update_time = BigIntegerField(null=True, index=True)
    update_date = DateTimeField(null=True, index=True)

    class Meta:
        database = None  # Will be set dynamically


class TenantLLM(BaseModel):
    """Tenant LLM model (source table)"""

    id = PrimaryKeyField()
    tenant_id = CharField(max_length=32, null=False, index=True)
    llm_factory = CharField(max_length=128, null=False, index=True)
    model_type = CharField(max_length=128, null=True, index=True)
    llm_name = CharField(max_length=128, null=True, default="", index=True)
    api_key = TextField(null=True)
    api_base = CharField(max_length=255, null=True)
    max_tokens = IntegerField(default=8192, index=True)
    used_tokens = IntegerField(default=0, index=True)
    status = CharField(max_length=1, null=False, default="1", index=True)

    class Meta:
        table_name = "tenant_llm"
        database = None


class TenantModelProvider(BaseModel):
    """Tenant Model Provider model (target table)"""

    id = CharField(max_length=32, primary_key=True)
    provider_name = CharField(max_length=128, null=False, index=True)
    tenant_id = CharField(max_length=32, null=False, index=True)

    class Meta:
        table_name = "tenant_model_provider"
        database = None


class MigrationStage:
    """Base class for migration stages"""

    name = "base_stage"
    description = "Base migration stage"
    source_tables = []
    target_tables = []

    def __init__(self, db: MigrationDatabase, dry_run: bool = True, create_table_only: bool = False):
        self.db = db
        self.dry_run = dry_run
        self.create_table_only = create_table_only
        self._noop_completes_migration = False

    def check(self) -> bool:
        """Check if migration is needed"""
        raise NotImplementedError

    def execute(self) -> tuple[int, list]:
        """Execute migration, returns (rows_affected, tables_operated)"""
        raise NotImplementedError

    def create_target_table(self):
        """Create target table (override in subclass if needed)"""
        pass

    def mark_noop_completes_migration(self):
        self._noop_completes_migration = True

    def noop_completes_migration(self) -> bool:
        return self._noop_completes_migration


class TenantModelContractPreflightStage(MigrationStage):
    """Validate every legacy model reference before model migration writes."""

    name = "tenant_model_contract_preflight"
    description = "Validate the complete legacy-to-current tenant model reference contract"
    source_tables = ["tenant_llm", "tenant", "knowledgebase", "dialog", "memory"]
    target_tables = []

    def check(self) -> bool:
        if not self.db.table_exists("tenant_llm"):
            raise RuntimeError("tenant_llm is required for tenant model migration")
        return True

    @staticmethod
    def build_source_maps(source_models: list) -> tuple[dict, dict]:
        source_by_id = {}
        source_by_name = {}
        for source_id, tenant_id, factory, model_name, model_type in source_models:
            canonical_type = TenantModelIdMigrationStage.canonical_model_type(model_type)
            identity = (str(tenant_id), str(factory), str(model_name), canonical_type)
            id_key = (str(tenant_id), str(source_id), canonical_type)
            previous = source_by_id.get(id_key)
            if previous and previous != identity:
                raise RuntimeError("legacy tenant model reference mapping is ambiguous")
            source_by_id[id_key] = identity

            name_key = (str(tenant_id), str(model_name), str(factory), canonical_type)
            previous_source_id = source_by_name.get(name_key)
            if previous_source_id is not None and previous_source_id != str(source_id):
                raise RuntimeError("legacy tenant model name mapping is ambiguous")
            source_by_name[name_key] = str(source_id)
        return source_by_id, source_by_name

    @staticmethod
    def validate_reference(
        *,
        source_by_id: dict,
        source_by_name: dict,
        current_model_ids: set,
        tenant_id: str,
        current_reference,
        legacy_model_name,
        model_type: str,
    ) -> None:
        canonical_type = TenantModelIdMigrationStage.canonical_model_type(model_type)
        current_value = str(current_reference).strip() if current_reference is not None else ""
        if current_value:
            current_key = (str(tenant_id), current_value, canonical_type)
            if current_key in current_model_ids or current_key in source_by_id:
                return
            raise RuntimeError(
                "legacy tenant model reference is unresolved: "
                f"tenant={tenant_id} reference={current_value} type={model_type}"
            )

        legacy_value = str(legacy_model_name).strip() if legacy_model_name is not None else ""
        if not legacy_value:
            return
        pure_name, _, provider_name = TenantModelIdMigrationStage._split_model_name_value(legacy_value)
        name_key = (str(tenant_id), pure_name, provider_name, canonical_type)
        if name_key not in source_by_name:
            raise RuntimeError(
                "legacy tenant model name is unresolved: "
                f"tenant={tenant_id} model={legacy_value} type={model_type}"
            )

    def _current_model_ids(self) -> set:
        required_tables = (
            "tenant_model_provider",
            "tenant_model_instance",
            "tenant_model",
        )
        if any(not self.db.table_exists(table) for table in required_tables):
            return set()
        cursor = self.db.execute_sql(
            "SELECT tm.id, tmp.tenant_id, tm.model_type "
            "FROM tenant_model tm "
            "INNER JOIN tenant_model_provider tmp ON tmp.id = tm.provider_id "
            "INNER JOIN tenant_model_instance tmi ON tmi.id = tm.instance_id "
            "WHERE tm.status = 'active' AND tmi.status = 'active'"
        )
        current_ids = set()
        for model_id, tenant_id, model_type in cursor.fetchall():
            for type_name, type_bit in TenantModelIdMigrationStage.MODEL_TYPE_TO_INT.items():
                if int(model_type) & type_bit:
                    current_ids.add(
                        (
                            str(tenant_id),
                            str(model_id),
                            TenantModelIdMigrationStage.canonical_model_type(type_name),
                        )
                    )
        return current_ids

    def execute(self) -> tuple[int, list]:
        cursor = self.db.execute_sql(
            "SELECT id, tenant_id, llm_factory, llm_name, model_type "
            "FROM tenant_llm WHERE status = '1' ORDER BY id"
        )
        source_by_id, source_by_name = self.build_source_maps(cursor.fetchall())
        current_model_ids = self._current_model_ids()

        checked = 0
        for table_name, fields in TenantModelIdMigrationStage.TENANT_ID_FIELDS.items():
            if not self.db.table_exists(table_name):
                continue
            tenant_expression = "id" if table_name == "tenant" else "tenant_id"
            for target_column, legacy_column, model_type in fields:
                target_exists = self.db.column_exists(table_name, target_column)
                legacy_exists = self.db.column_exists(table_name, legacy_column)
                if not target_exists and not legacy_exists:
                    continue
                target_expression = f"`{target_column}`" if target_exists else "NULL"
                legacy_expression = f"`{legacy_column}`" if legacy_exists else "NULL"
                cursor = self.db.execute_sql(
                    f"SELECT `{tenant_expression}`, {target_expression}, {legacy_expression} "
                    f"FROM `{table_name}`"
                )
                for tenant_id, current_reference, legacy_model_name in cursor.fetchall():
                    self.validate_reference(
                        source_by_id=source_by_id,
                        source_by_name=source_by_name,
                        current_model_ids=current_model_ids,
                        tenant_id=str(tenant_id),
                        current_reference=current_reference,
                        legacy_model_name=legacy_model_name,
                        model_type=model_type,
                    )
                    if current_reference not in (None, "") or legacy_model_name not in (None, ""):
                        checked += 1
        logger.info("Validated %s tenant model references before migration writes", checked)
        return 0, []


class TenantModelProviderStage(MigrationStage):
    """Migrate tenant_llm to tenant_model_provider"""

    name = "tenant_model_provider"
    description = "Migrate tenant_llm.llm_factory to tenant_model_provider.provider_name"
    source_tables = ["tenant_llm"]
    target_tables = ["tenant_model_provider"]

    def current_timestamp(self) -> int:
        return int(time.time())

    def generate_uuid(self) -> str:
        """Generate 32-character UUID1"""
        return uuid.uuid1().hex

    def check(self) -> bool:
        """Check if migration is needed"""
        # Check if source table exists
        if not self.db.table_exists("tenant_llm"):
            logger.warning("Source table 'tenant_llm' does not exist")
            return False

        # Check if target table exists
        if not self.db.table_exists("tenant_model_provider"):
            if self.dry_run:
                logger.info("[DRY RUN] Target table 'tenant_model_provider' does not exist. Use --execute to create and populate the table.")
                return False
            logger.info("Target table 'tenant_model_provider' does not exist, will create")
            return True

        # Check if there's data to migrate
        cursor = self.db.execute_sql(
            "SELECT COUNT(*) FROM tenant_llm t1 WHERE NOT EXISTS (  SELECT 1 FROM tenant_model_provider t2   WHERE t2.tenant_id = t1.tenant_id AND t2.provider_name = t1.llm_factory)"
        )
        count = cursor.fetchone()[0]

        if count == 0:
            self.mark_noop_completes_migration()
            logger.info("No new data to migrate from tenant_llm to tenant_model_provider")
            return False

        logger.info(f"Found {count} rows to migrate from tenant_llm to tenant_model_provider")
        return True

    def execute(self) -> tuple[int, list]:
        """Execute migration"""
        current_ts = self.current_timestamp()
        rows_inserted = 0

        # Check if target table exists
        if not self.db.table_exists("tenant_model_provider"):
            if self.dry_run:
                logger.info("[DRY RUN] Target table 'tenant_model_provider' does not exist. Use --execute to create and populate the table.")
                return 0, []
            logger.info("Target table 'tenant_model_provider' does not exist, will create")
            self.create_target_table()

        # If create_table_only mode, skip data migration
        if self.create_table_only:
            logger.info("[CREATE TABLE ONLY] Target table created/verified, skipping data migration")
            return 0, self.target_tables

        # Get distinct tenant_id, llm_factory pairs that don't exist in target
        cursor = self.db.execute_sql(
            "SELECT DISTINCT tenant_id, llm_factory FROM tenant_llm t1 "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM tenant_model_provider t2 "
            "  WHERE t2.tenant_id = t1.tenant_id AND t2.provider_name = t1.llm_factory"
            ")"
        )

        records = cursor.fetchall()

        if not records:
            logger.info("No records to migrate")
            return 0, []

        logger.info(f"Migrating {len(records)} unique tenant_id/llm_factory pairs...")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would insert {len(records)} records")
            return len(records), self.target_tables

        # Insert records in batches with parameterized SQL to avoid quote breakage/injection
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            placeholders = []
            params = []
            for tenant_id, llm_factory in batch:
                record_id = self.generate_uuid()
                placeholders.append("(%s, %s, %s, %s, FROM_UNIXTIME(%s), %s, FROM_UNIXTIME(%s))")
                params.extend(
                    [
                        record_id,
                        llm_factory,
                        tenant_id,
                        current_ts * 1000,
                        current_ts,
                        current_ts * 1000,
                        current_ts,
                    ]
                )
            insert_sql = f"""
                INSERT INTO tenant_model_provider
                (id, provider_name, tenant_id, create_time, create_date, update_time, update_date)
                VALUES {", ".join(placeholders)}
            """
            self.db.execute_sql(insert_sql, params)
            rows_inserted += len(batch)
            logger.info(f"Inserted batch {i // batch_size + 1}: {len(batch)} records")

        return rows_inserted, self.target_tables

    def create_target_table(self):
        """Create tenant_model_provider table"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS tenant_model_provider (
            id VARCHAR(32) NOT NULL PRIMARY KEY,
            provider_name VARCHAR(128) NOT NULL,
            tenant_id VARCHAR(32) NOT NULL,
            create_time BIGINT,
            create_date DATETIME,
            update_time BIGINT,
            update_date DATETIME,
            INDEX idx_provider_name (provider_name),
            INDEX idx_tenant_id (tenant_id),
            UNIQUE INDEX idx_tenant_provider_unique (tenant_id, provider_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        self.db.execute_sql(create_sql)
        logger.info("Created tenant_model_provider table")


class TenantModelInstanceStage(MigrationStage):
    """Migrate tenant_llm to tenant_model_instance"""

    name = "tenant_model_instance"
    description = "Migrate tenant_llm to tenant_model_instance with provider_id lookup"
    source_tables = ["tenant_llm", "tenant_model_provider"]
    target_tables = ["tenant_model_instance"]

    def current_timestamp(self) -> int:
        return int(time.time())

    def generate_uuid(self) -> str:
        """Generate 32-character UUID1"""
        return uuid.uuid1().hex

    @staticmethod
    def build_instance_extra(api_base: str | None) -> str:
        payload = {"base_url": api_base.strip()} if api_base and api_base.strip() else {}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @classmethod
    def instance_identity(cls, provider_id: str, api_key: str | None, api_base: str | None) -> tuple[str, str, str]:
        return (
            str(provider_id),
            cls._strip_is_tools_from_api_key(api_key or ""),
            (api_base or "").strip(),
        )

    @staticmethod
    def _base_url_from_extra(extra: str | None) -> str:
        if not extra:
            return ""
        try:
            payload = json.loads(extra)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("tenant_model_instance.extra is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("tenant_model_instance.extra must be a JSON object")
        base_url = payload.get("base_url", "")
        if base_url is None:
            return ""
        if not isinstance(base_url, str):
            raise RuntimeError("tenant_model_instance.extra.base_url must be a string")
        return base_url.strip()

    def check(self) -> bool:
        """Check if migration is needed"""
        # Check if source table exists
        if not self.db.table_exists("tenant_llm"):
            logger.warning("Source table 'tenant_llm' does not exist")
            return False

        # Check if tenant_model_provider exists (dependency)
        if not self.db.table_exists("tenant_model_provider"):
            if self.dry_run:
                logger.info("[DRY RUN] Dependency table 'tenant_model_provider' does not exist. Run 'tenant_model_provider' stage first or use --execute.")
                return False
            logger.warning("Dependency table 'tenant_model_provider' does not exist. Please run 'tenant_model_provider' stage first.")
            return False

        # Check if target table exists
        if not self.db.table_exists("tenant_model_instance"):
            if self.dry_run:
                logger.info("[DRY RUN] Target table 'tenant_model_instance' does not exist. Use --execute to create and populate the table.")
                return False
            logger.info("Target table 'tenant_model_instance' does not exist, will create")
            return True

        cursor = self.db.execute_sql("SELECT COUNT(*) FROM tenant_llm WHERE status = '1'")
        count = cursor.fetchone()[0]

        if count == 0:
            self.mark_noop_completes_migration()
            logger.info("No new data to migrate from tenant_llm to tenant_model_instance")
            return False

        logger.info(f"Found {count} rows to migrate from tenant_llm to tenant_model_instance")
        return True

    def execute(self) -> tuple[int, list]:
        """Execute migration"""
        current_ts = self.current_timestamp()
        rows_inserted = 0

        # Check if tenant_model_provider exists (dependency)
        if not self.db.table_exists("tenant_model_provider"):
            logger.error("Dependency table 'tenant_model_provider' does not exist. Please run 'tenant_model_provider' stage first.")
            return 0, []

        # Check if target table exists
        if not self.db.table_exists("tenant_model_instance"):
            if self.dry_run:
                logger.info("[DRY RUN] Target table 'tenant_model_instance' does not exist. Use --execute to create and populate the table.")
                return 0, []
            logger.info("Target table 'tenant_model_instance' does not exist, will create")
            self.create_target_table()

        # If create_table_only mode, skip data migration
        if self.create_table_only:
            logger.info("[CREATE TABLE ONLY] Target table created/verified, skipping data migration")
            return 0, self.target_tables

        cursor = self.db.execute_sql(
            "SELECT tl.tenant_id, tl.llm_factory, tl.api_key, tl.api_base, tl.status, tmp.id as provider_id "
            "FROM tenant_llm tl "
            "INNER JOIN tenant_model_provider tmp ON tmp.tenant_id = tl.tenant_id AND tmp.provider_name = tl.llm_factory "
            "WHERE tl.status = '1'"
        )

        source_records = cursor.fetchall()

        if not source_records:
            logger.info("No records to migrate")
            return 0, []

        cursor = self.db.execute_sql("SELECT id, provider_id, api_key, status, extra FROM tenant_model_instance")
        existing_by_identity = {}
        for instance_id, provider_id, api_key, status, extra in cursor.fetchall():
            identity = self.instance_identity(provider_id, api_key, self._base_url_from_extra(extra))
            previous = existing_by_identity.get(identity)
            if previous and previous != instance_id:
                raise RuntimeError("tenant_model_instance identity is ambiguous")
            existing_by_identity[identity] = instance_id

        source_by_identity = {}
        provider_identity_counts = {}
        for tenant_id, llm_factory, api_key, api_base, status, provider_id in source_records:
            identity = self.instance_identity(provider_id, api_key, api_base)
            source_by_identity.setdefault(identity, (tenant_id, llm_factory, api_key, api_base, status, provider_id))
        for identity in source_by_identity:
            provider_identity_counts[identity[0]] = provider_identity_counts.get(identity[0], 0) + 1

        records = [record for identity, record in source_by_identity.items() if identity not in existing_by_identity]

        if not records:
            logger.info("No new tenant_model_instance records to migrate")
            return 0, []

        logger.info("Migrating %s tenant_model_instance records...", len(records))

        if self.dry_run:
            logger.info(f"[DRY RUN] Would insert {len(records)} records")
            for tenant_id, llm_factory, api_key, api_base, status, provider_id in records[:5]:
                logger.info(f"  instance_name=default, provider_id={provider_id}, api_key=***")
            if len(records) > 5:
                logger.info(f"  ... and {len(records) - 5} more records")
            return len(records), self.target_tables

        # Insert records in batches
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            placeholders = []
            params = []
            for tenant_id, llm_factory, api_key, api_base, status, provider_id in batch:
                record_id = self.generate_uuid()
                identity = self.instance_identity(provider_id, api_key, api_base)
                digest = hashlib.sha256("\0".join(identity).encode("utf-8")).hexdigest()[:12]
                instance_name = "default" if provider_identity_counts[str(provider_id)] == 1 else f"legacy-{digest}"
                status_val = "active" if status in ["1", "active", "enable"] else "inactive"
                placeholders.append(
                    "(%s, %s, %s, %s, %s, %s, %s, FROM_UNIXTIME(%s), %s, FROM_UNIXTIME(%s))"
                )
                params.extend(
                    [
                        record_id,
                        instance_name,
                        provider_id,
                        api_key or "",
                        status_val,
                        self.build_instance_extra(api_base),
                        current_ts * 1000,
                        current_ts,
                        current_ts * 1000,
                        current_ts,
                    ]
                )

            insert_sql = f"""
                INSERT INTO tenant_model_instance
                (id, instance_name, provider_id, api_key, status, extra,
                 create_time, create_date, update_time, update_date)
                VALUES {", ".join(placeholders)}
            """
            self.db.execute_sql(insert_sql, params)
            rows_inserted += len(batch)
            logger.info(f"Inserted batch {i // batch_size + 1}: {len(batch)} records")

        return rows_inserted, self.target_tables

    @staticmethod
    def _strip_is_tools_from_api_key(api_key: str) -> str:
        """Strip is_tools from api_key for dedup comparison.

        Handles three api_key formats:
        1. Plain string (e.g. "sk-xxx" or "x") — returned as-is.
        2. JSON with only {"api_key": "...", "is_tools": true/false} — extract the inner api_key value.
        3. JSON with factory-specific fields + optional "is_tools" — remove only the "is_tools" key.

        For format 3, the factory-specific JSON structures are:
          VolcEngine:          {"ark_api_key": ..., "endpoint_id": ...}
          Tencent Cloud:       {"tencent_cloud_sid": ..., "tencent_cloud_sk": ...}
          Bedrock:             {"auth_mode": ..., "bedrock_ak": ..., "bedrock_sk": ..., "bedrock_region": ..., "aws_role_arn": ...}
          XunFei Spark (tts):  {"spark_app_id": ..., "spark_api_secret": ..., "spark_api_key": ...}
          BaiduYiyan:          {"yiyan_ak": ..., "yiyan_sk": ...}
          Fish Audio:          {"fish_audio_ak": ..., "fish_audio_refid": ...}
          Google Cloud:        {"google_project_id": ..., "google_region": ..., "google_service_account_key": ...}
          Azure-OpenAI:        {"api_key": ..., "api_version": ...}
          OpenRouter:          {"api_key": ..., "provider_order": ...}
          MinerU:              {"api_key": ..., "provider_order": ...}
          PaddleOCR:           {"api_key": ..., "provider_order": ...}
          OpenDataLoader:      {"api_key": ..., "provider_order": ...}
        """
        if not api_key:
            return api_key

        try:
            parsed = json.loads(api_key)
        except (json.JSONDecodeError, TypeError, ValueError):
            return api_key

        if not isinstance(parsed, dict):
            return api_key

        # Case 2: {"api_key": "...", "is_tools": true/false} — extract inner api_key
        if set(parsed.keys()) <= {"api_key", "is_tools"}:
            return parsed.get("api_key", "")

        # Case 3: factory-specific JSON with is_tools appended — remove is_tools key
        if "is_tools" in parsed:
            payload = {k: v for k, v in parsed.items() if k != "is_tools"}
            return json.dumps(payload, sort_keys=True)

        # Already a JSON dict without is_tools — return as-is
        return json.dumps(parsed, sort_keys=True)

    def _dedup_api_key_records(self, records: list) -> list:
        """Deduplicate records whose api_keys are logically identical after stripping is_tools.

        Groups by (tenant_id, llm_factory, provider_id). Within each group, if multiple
        records share the same canonical api_key (with is_tools removed), only one is kept.
        The kept record uses the original api_key value from the first occurrence; is_tools
        information is not needed in tenant_model_instance (it is stored in tenant_model instead).
        """
        from collections import defaultdict

        groups = defaultdict(list)
        for rec in records:
            tenant_id, llm_factory, api_key, status, provider_id = rec
            groups[(tenant_id, llm_factory, provider_id)].append(rec)

        deduped = []
        dup_count = 0
        for (tenant_id, llm_factory, provider_id), group in groups.items():
            if len(group) <= 1:
                deduped.extend(group)
                continue

            # Multiple records in group — dedup by canonical api_key
            seen = {}  # canonical_key -> first record
            for rec in group:
                _, _, api_key, _, _ = rec
                canonical = self._strip_is_tools_from_api_key(api_key)
                if canonical not in seen:
                    seen[canonical] = rec
                else:
                    dup_count += 1
                    logger.debug("Deduped equivalent API-key records for tenant=%s factory=%s provider=%s", tenant_id, llm_factory, provider_id)
            deduped.extend(seen.values())

        if dup_count > 0:
            logger.info(f"Deduplicated {dup_count} api_key records (is_tools-only differences)")

        return deduped

    def create_target_table(self):
        """Create tenant_model_instance table"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS tenant_model_instance (
            id VARCHAR(32) NOT NULL PRIMARY KEY,
            instance_name VARCHAR(128) NOT NULL,
            provider_id VARCHAR(32) NOT NULL,
            api_key VARCHAR(512) NOT NULL,
            status VARCHAR(32) DEFAULT 'active',
            extra VARCHAR(512) DEFAULT '{}',
            create_time BIGINT,
            create_date DATETIME,
            update_time BIGINT,
            update_date DATETIME,
            INDEX idx_provider_id (provider_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        self.db.execute_sql(create_sql)
        logger.info("Created tenant_model_instance table")


class TenantModelStage(MigrationStage):
    """Migrate tenant_llm to tenant_model"""

    name = "tenant_model"
    description = "Migrate every enabled tenant_llm model to tenant_model"
    source_tables = ["tenant_llm", "tenant_model_provider", "tenant_model_instance"]
    target_tables = ["tenant_model"]

    @staticmethod
    def _get_empty_llm_factories() -> list[str]:
        """Load factory names whose llm field is an empty list from conf/llm_factories.json"""
        conf_path = os.path.join(PROJECT_BASE, "conf", "llm_factories.json")
        with open(conf_path, "r") as f:
            data = json.load(f)
        factories = []
        for key, items in data.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        llm = item.get("llm")
                        if isinstance(llm, list) and len(llm) == 0:
                            factories.append(item["name"])
        return factories

    MODEL_TYPE_TO_INT = {
        "chat": 1,
        "embedding": 2,
        "asr": 4,
        "speech2text": 4,
        "vision": 8,
        "image2text": 8,
        "rerank": 16,
        "tts": 32,
        "ocr": 64,
    }

    @classmethod
    def model_type_for_storage(cls, model_type: str, storage_type: str):
        normalized = (model_type or "").strip().lower()
        if normalized not in cls.MODEL_TYPE_TO_INT:
            raise ValueError(f"unsupported tenant model type: {model_type}")
        if storage_type.lower() in ("int", "integer", "bigint"):
            return cls.MODEL_TYPE_TO_INT[normalized]
        return normalized

    @staticmethod
    def build_status_condition(_empty_factories: list[str] | None = None) -> str:
        return "tl.status = '1'"

    def _build_status_condition(self) -> str:
        return self.build_status_condition()

    @staticmethod
    def build_model_extra(api_key: str | None, max_tokens: int | None) -> str:
        payload = {}
        if api_key:
            try:
                parsed = json.loads(api_key)
            except (json.JSONDecodeError, TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("is_tools"), bool):
                payload["is_tools"] = parsed["is_tools"]
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def merge_model_extra(current_extra: str | None, source_extra: str) -> str:
        try:
            current = json.loads(current_extra or "{}")
            source = json.loads(source_extra)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("tenant_model.extra is not valid JSON") from exc
        if not isinstance(current, dict) or not isinstance(source, dict):
            raise RuntimeError("tenant_model.extra must be a JSON object")
        current.update(source)
        return json.dumps(current, ensure_ascii=False, sort_keys=True)

    def current_timestamp(self) -> int:
        return int(time.time())

    def generate_uuid(self) -> str:
        """Generate 32-character UUID1"""
        return uuid.uuid1().hex

    def check(self) -> bool:
        """Check if migration is needed"""
        # Check if source table exists
        if not self.db.table_exists("tenant_llm"):
            logger.warning("Source table 'tenant_llm' does not exist")
            return False

        # Check if tenant_model_provider exists (dependency)
        if not self.db.table_exists("tenant_model_provider"):
            if self.dry_run:
                logger.info("[DRY RUN] Dependency table 'tenant_model_provider' does not exist. Run 'tenant_model_provider' stage first or use --execute.")
                return False
            logger.warning("Dependency table 'tenant_model_provider' does not exist. Please run 'tenant_model_provider' stage first.")
            return False

        # Check if tenant_model_instance exists (dependency)
        if not self.db.table_exists("tenant_model_instance"):
            if self.dry_run:
                logger.info("[DRY RUN] Dependency table 'tenant_model_instance' does not exist. Run 'tenant_model_instance' stage first or use --execute.")
                return False
            logger.warning("Dependency table 'tenant_model_instance' does not exist. Please run 'tenant_model_instance' stage first.")
            return False

        # Check if target table exists
        if not self.db.table_exists("tenant_model"):
            if self.dry_run:
                logger.info("[DRY RUN] Target table 'tenant_model' does not exist. Use --execute to create and populate the table.")
                return False
            logger.info("Target table 'tenant_model' does not exist, will create")
            return True

        status_condition = self._build_status_condition()

        cursor = self.db.execute_sql(
            f"SELECT COUNT(*) FROM tenant_llm tl WHERE {status_condition}"
        )
        count = cursor.fetchone()[0]

        if count == 0:
            self.mark_noop_completes_migration()
            logger.info("No new data to migrate from tenant_llm to tenant_model")
            return False

        logger.info(f"Found {count} rows to migrate from tenant_llm to tenant_model")
        return True

    def execute(self) -> tuple[int, list]:
        """Execute migration"""
        current_ts = self.current_timestamp()
        rows_inserted = 0

        # Check if tenant_model_provider exists (dependency)
        if not self.db.table_exists("tenant_model_provider"):
            logger.error("Dependency table 'tenant_model_provider' does not exist. Please run 'tenant_model_provider' stage first.")
            return 0, []

        # Check if tenant_model_instance exists (dependency)
        if not self.db.table_exists("tenant_model_instance"):
            logger.error("Dependency table 'tenant_model_instance' does not exist. Please run 'tenant_model_instance' stage first.")
            return 0, []

        # Check if target table exists
        if not self.db.table_exists("tenant_model"):
            if self.dry_run:
                logger.info("[DRY RUN] Target table 'tenant_model' does not exist. Use --execute to create and populate the table.")
                return 0, []
            logger.info("Target table 'tenant_model' does not exist, will create")
            self.create_target_table()

        # If create_table_only mode, skip data migration
        if self.create_table_only:
            logger.info("[CREATE TABLE ONLY] Target table created/verified, skipping data migration")
            return 0, self.target_tables

        status_condition = self._build_status_condition()

        instance_lookup = self._build_instance_lookup()

        cursor = self.db.execute_sql(
            f"SELECT tl.id, tl.llm_name, tmp.id as provider_id, "
            f"       tl.model_type, tl.status, tl.api_key, tl.api_base, tl.max_tokens "
            f"FROM tenant_llm tl "
            f"INNER JOIN tenant_model_provider tmp ON tmp.tenant_id = tl.tenant_id AND tmp.provider_name = tl.llm_factory "
            f"WHERE {status_condition}"
        )

        records = cursor.fetchall()

        if not records:
            logger.info("No records to migrate")
            return 0, []

        # Resolve instance_id for each record using Python-level canonical matching
        resolved_records = self._resolve_instance_ids(records, instance_lookup)

        storage_type = self.db.get_column_type("tenant_model", "model_type") or "int"
        cursor = self.db.execute_sql("SELECT id, provider_id, instance_id, model_name, model_type, status, extra FROM tenant_model")
        existing = {}
        for row in cursor.fetchall():
            model_id, provider_id, instance_id, model_name, model_type, status, extra = row
            key = (provider_id, instance_id, model_name)
            if key in existing:
                raise RuntimeError("tenant_model identity is ambiguous")
            existing[key] = (model_id, int(model_type), status, extra or "{}")

        pending_inserts = []
        pending_updates = []
        for source_id, llm_name, provider_id, instance_id, model_type, status, api_key, api_base, max_tokens in resolved_records:
            type_value = self.model_type_for_storage(model_type, storage_type)
            if not isinstance(type_value, int):
                raise RuntimeError("tenant_model.model_type must use integer bit flags")
            extra = self.build_model_extra(api_key, max_tokens)
            key = (provider_id, instance_id, llm_name)
            current = existing.get(key)
            if current:
                model_id, current_type, current_status, current_extra = current
                merged_type = current_type | type_value
                merged_extra = self.merge_model_extra(current_extra, extra)
                if current_type != merged_type or current_status != "active" or current_extra != merged_extra:
                    pending_updates.append((merged_type, "active", merged_extra, model_id))
                continue
            pending_inserts.append((source_id, llm_name, provider_id, instance_id, type_value, "active", extra))

        if not pending_inserts and not pending_updates:
            logger.info("No new tenant_model records or metadata updates to migrate")
            return 0, []

        logger.info("Migrating %s tenant_model inserts and %s updates...", len(pending_inserts), len(pending_updates))

        if self.dry_run:
            return len(pending_inserts) + len(pending_updates), self.target_tables

        for model_type, status, extra, model_id in pending_updates:
            self.db.execute_sql(
                "UPDATE tenant_model SET model_type=%s, status=%s, extra=%s WHERE id=%s",
                (model_type, status, extra, model_id),
            )
            rows_inserted += 1

        batch_size = 100
        for i in range(0, len(pending_inserts), batch_size):
            batch = pending_inserts[i : i + batch_size]
            placeholders = []
            params = []
            for source_id, llm_name, provider_id, instance_id, model_type, status, extra in batch:
                record_id = self.generate_uuid()
                placeholders.append(
                    "(%s, %s, %s, %s, %s, %s, %s, %s, FROM_UNIXTIME(%s), %s, FROM_UNIXTIME(%s))"
                )
                params.extend(
                    [record_id, llm_name or "", provider_id, instance_id, model_type, status,
                     extra, current_ts * 1000, current_ts, current_ts * 1000, current_ts]
                )

            insert_sql = f"""
                INSERT INTO tenant_model
                (id, model_name, provider_id, instance_id, model_type, status, extra,
                 create_time, create_date, update_time, update_date)
                VALUES {", ".join(placeholders)}
            """
            self.db.execute_sql(insert_sql, params)
            rows_inserted += len(batch)
            logger.info(f"Inserted batch {i // batch_size + 1}: {len(batch)} records")

        return rows_inserted, self.target_tables

    def _build_instance_lookup(self) -> dict:
        """Load all tenant_model_instance records, indexed by (provider_id, canonical_api_key).

        The canonical_api_key is computed by stripping is_tools from the stored api_key,
        matching the dedup logic used during the instance migration stage.

        Returns:
            dict mapping (provider_id, canonical_api_key) -> instance_id
        """
        cursor = self.db.execute_sql("SELECT id, provider_id, api_key, extra FROM tenant_model_instance")
        lookup = {}
        for instance_id, provider_id, api_key, extra in cursor.fetchall():
            identity = TenantModelInstanceStage.instance_identity(
                provider_id,
                api_key,
                TenantModelInstanceStage._base_url_from_extra(extra),
            )
            if identity in lookup and lookup[identity] != instance_id:
                raise RuntimeError("tenant_model_instance identity is ambiguous")
            lookup[identity] = instance_id
        logger.info(f"Loaded {len(lookup)} instance records for lookup")
        return lookup

    @staticmethod
    def _resolve_instance_ids(records: list, instance_lookup: dict) -> list:
        """Resolve instance_id for each tenant_llm record using canonical api_key matching.

        Args:
            records: source tenant_llm rows with provider and endpoint evidence

        Returns:
            source rows with the exact matching instance_id inserted after provider_id.
        """
        resolved = []
        missing = []
        for record in records:
            source_id, llm_name, provider_id, model_type, status, api_key, *metadata = record
            api_base = metadata[0] if metadata else ""
            identity = TenantModelInstanceStage.instance_identity(provider_id, api_key, api_base)
            instance_id = instance_lookup.get(identity)
            if instance_id:
                resolved.append((source_id, llm_name, provider_id, instance_id, model_type, status, api_key, *metadata))
            else:
                missing.append((source_id, provider_id, llm_name))

        if missing:
            raise RuntimeError(
                "tenant_model_instance mapping is incomplete for tenant_llm ids: "
                + ",".join(str(item[0]) for item in missing)
            )

        return resolved

    @staticmethod
    def _extract_extra_from_api_key(api_key: str) -> str:
        """Extract is_tools from api_key JSON and return an extra JSON string for tenant_model.

        If api_key is a JSON dict containing "is_tools": true, return '{"is_tools": true}'.
        Otherwise return '{}' (empty dict).
        """
        if not api_key:
            return "{}"

        try:
            parsed = json.loads(api_key)
        except (json.JSONDecodeError, TypeError, ValueError):
            return "{}"

        if not isinstance(parsed, dict):
            return "{}"

        if parsed.get("is_tools") is True:
            return json.dumps({"is_tools": True})

        return "{}"

    def create_target_table(self):
        """Create tenant_model table"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS tenant_model (
            id VARCHAR(32) NOT NULL PRIMARY KEY,
            model_name VARCHAR(128),
            provider_id VARCHAR(32) NOT NULL,
            instance_id VARCHAR(32) NOT NULL,
            model_type INT NOT NULL,
            status VARCHAR(32) DEFAULT 'active',
            extra VARCHAR(1024) DEFAULT '{}',
            create_time BIGINT,
            create_date DATETIME,
            update_time BIGINT,
            update_date DATETIME,
            INDEX idx_instance_id (instance_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        self.db.execute_sql(create_sql)
        logger.info("Created tenant_model table")


class ModelIdConfigStage(MigrationStage):
    """Normalize stored model IDs from model@provider to model@default@provider."""

    name = "model_id_config"
    description = "Normalize stored model IDs in config columns to model@default@provider"
    source_tables = [
        "tenant",
        "knowledgebase",
        "document",
        "dialog",
        "memory",
        "search",
        "user_canvas",
        "canvas_template",
        "user_canvas_version",
        "api_4_conversation",
        "pipeline_operation_log",
        "connector",
        "evaluation_runs",
    ]
    target_tables = source_tables

    model_id_fields = {
        "llm_id",
        "embd_id",
        "embedding_model",
        "rerank_id",
        "asr_id",
        "img2txt_id",
        "tts_id",
        "ocr_id",
    }
    search_config_model_id_fields = {"chat_id"}
    scan_batch_size = 500
    string_columns = {
        "tenant": ("llm_id", "embd_id", "asr_id", "img2txt_id", "rerank_id", "tts_id", "ocr_id"),
        "knowledgebase": ("embd_id",),
        "dialog": ("llm_id", "rerank_id"),
        "memory": ("embd_id", "llm_id"),
    }
    json_columns = {
        "knowledgebase": ("parser_config",),
        "document": ("parser_config",),
        "search": ("search_config",),
        "user_canvas": ("dsl",),
        "canvas_template": ("dsl",),
        "user_canvas_version": ("dsl",),
        "api_4_conversation": ("dsl",),
        "pipeline_operation_log": ("dsl",),
        "connector": ("config",),
        "evaluation_runs": ("config_snapshot",),
    }

    def normalize_model_id(self, value):
        if not isinstance(value, str) or not value:
            return value, False

        parts = value.split("@")
        if len(parts) != 2:
            return value, False

        model_name, provider_name = parts
        if not model_name or not provider_name:
            return value, False

        return f"{model_name}@default@{provider_name}", True

    def normalize_config(self, value, path=None):
        path = path or ()

        if isinstance(value, dict):
            changed = False
            normalized = {}
            for key, item in value.items():
                key_path = path + (str(key),)
                should_normalize = key in self.model_id_fields or (key in self.search_config_model_id_fields and "search_config" in path)
                if should_normalize:
                    normalized_item, item_changed = self.normalize_model_id(item)
                else:
                    normalized_item, item_changed = self.normalize_config(item, key_path)
                normalized[key] = normalized_item
                changed = changed or item_changed
            return normalized, changed

        if isinstance(value, list):
            changed = False
            normalized = []
            for index, item in enumerate(value):
                normalized_item, item_changed = self.normalize_config(item, path + (str(index),))
                normalized.append(normalized_item)
                changed = changed or item_changed
            return normalized, changed

        return value, False

    def existing_columns(self, table_columns):
        for table_name, columns in table_columns.items():
            if not self.db.table_exists(table_name):
                logger.info("Table '%s' does not exist, skipping", table_name)
                continue
            for column_name in columns:
                if not self.db.column_exists(table_name, column_name):
                    logger.info("Column '%s.%s' does not exist, skipping", table_name, column_name)
                    continue
                yield table_name, column_name

    def load_json_value(self, raw_value, table_name, column_name, row_id):
        if raw_value in (None, ""):
            return None, False
        if isinstance(raw_value, (dict, list)):
            return raw_value, True
        try:
            return json.loads(raw_value), True
        except (TypeError, json.JSONDecodeError):
            logger.warning(
                "Failed to parse JSON in %s.%s id=%s, skipping",
                table_name,
                column_name,
                row_id,
            )
            return None, False

    def iter_string_changes(self):
        for table_name, column_name in self.existing_columns(self.string_columns):
            cursor = self.db.execute_sql(
                f"SELECT id, `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL AND `{column_name}` != '' AND `{column_name}` LIKE %s",
                ("%@%",),
            )
            while True:
                rows = cursor.fetchmany(self.scan_batch_size)
                if not rows:
                    break
                for row_id, value in rows:
                    normalized, changed = self.normalize_model_id(value)
                    if changed:
                        yield table_name, column_name, row_id, normalized

    def iter_json_changes(self):
        for table_name, column_name in self.existing_columns(self.json_columns):
            cursor = self.db.execute_sql(
                f"SELECT id, `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL AND `{column_name}` != '' AND `{column_name}` LIKE %s",
                ("%@%",),
            )
            while True:
                rows = cursor.fetchmany(self.scan_batch_size)
                if not rows:
                    break
                for row_id, raw_value in rows:
                    config, loaded = self.load_json_value(raw_value, table_name, column_name, row_id)
                    if not loaded:
                        continue
                    normalized, changed = self.normalize_config(config, (column_name,))
                    if changed:
                        normalized_json = json.dumps(
                            normalized,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        yield table_name, column_name, row_id, normalized_json

    def count_changes(self) -> tuple[int, set]:
        rows = 0
        tables = set()
        for table_name, _, _, _ in self.iter_string_changes():
            rows += 1
            tables.add(table_name)
        for table_name, _, _, _ in self.iter_json_changes():
            rows += 1
            tables.add(table_name)
        return rows, tables

    def check(self) -> bool:
        rows, tables = self.count_changes()
        if rows == 0:
            self.mark_noop_completes_migration()
            logger.info("No stored model IDs need normalization")
            return False
        logger.info(
            "Found %s rows to normalize across tables: %s",
            rows,
            ", ".join(sorted(tables)),
        )
        return True

    def execute(self) -> tuple[int, list]:
        if self.create_table_only:
            logger.info("[CREATE TABLE ONLY] No tables are created for this data migration")
            return 0, []

        rows_updated = 0
        tables_operated = set()

        for table_name, column_name, row_id, normalized in self.iter_string_changes():
            tables_operated.add(table_name)
            rows_updated += 1
            if rows_updated <= 10:
                logger.info(
                    "%s %s.%s id=%s -> %s",
                    "[DRY RUN] Would update" if self.dry_run else "Updating",
                    table_name,
                    column_name,
                    row_id,
                    normalized,
                )
            if not self.dry_run:
                self.db.execute_sql(
                    f"UPDATE `{table_name}` SET `{column_name}` = %s WHERE id = %s",
                    (normalized, row_id),
                )

        for table_name, column_name, row_id, normalized_json in self.iter_json_changes():
            tables_operated.add(table_name)
            rows_updated += 1
            if rows_updated <= 10:
                logger.info(
                    "%s %s.%s id=%s",
                    "[DRY RUN] Would update" if self.dry_run else "Updating",
                    table_name,
                    column_name,
                    row_id,
                )
            if not self.dry_run:
                self.db.execute_sql(
                    f"UPDATE `{table_name}` SET `{column_name}` = %s WHERE id = %s",
                    (normalized_json, row_id),
                )

        if rows_updated > 10:
            logger.info("... and %s more row updates", rows_updated - 10)

        if self.dry_run:
            logger.info("[DRY RUN] Would update %s rows", rows_updated)
        else:
            logger.info("Updated %s rows", rows_updated)

        return rows_updated, sorted(tables_operated)


class TenantModelSeedingStage(MigrationStage):
    """Seed tenant_model table with models from conf/llm_factories.json"""

    name = "tenant_model_seeding"
    description = "Seed tenant_model with models from conf/llm_factories.json for existing providers/instances"
    source_tables = ["tenant_model_provider", "tenant_model_instance", "tenant_model"]
    target_tables = ["tenant_model"]

    def current_timestamp(self) -> int:
        return int(time.time())

    def generate_uuid(self) -> str:
        """Generate 32-character UUID1"""
        return uuid.uuid1().hex

    def _load_llm_factories(self) -> list:
        """Load factory_llm_infos from conf/llm_factories.json"""
        conf_path = os.path.join(PROJECT_BASE, "conf", "llm_factories.json")
        with open(conf_path, "r") as f:
            data = json.load(f)
        return data.get("factory_llm_infos", [])

    def check(self) -> bool:
        """Check if migration is needed"""
        if not self.db.table_exists("tenant_model_provider"):
            logger.warning("Dependency table 'tenant_model_provider' does not exist")
            return False

        if not self.db.table_exists("tenant_model_instance"):
            logger.warning("Dependency table 'tenant_model_instance' does not exist")
            return False

        if not self.db.table_exists("tenant_model"):
            if self.dry_run:
                logger.info("[DRY RUN] Target table 'tenant_model' does not exist")
                return False
            return True

        # If model_type is already INT, the merge stage has been executed — seeding is not applicable
        model_type_dtype = self.db.get_column_type("tenant_model", "model_type")
        if model_type_dtype and model_type_dtype.lower() == "int":
            self.mark_noop_completes_migration()
            logger.info("tenant_model.model_type is already INT, seeding stage is not applicable (merge already done)")
            return False

        # Check if there are any models to seed
        factories = self._load_llm_factories()
        total_missing = 0
        for factory in factories:
            llm_list = factory.get("llm", [])
            if not llm_list:
                continue
            factory_name = factory["name"]
            # Determine the provider_name to query and optional instance extra filter
            if factory_name == "siliconflow_intl":
                provider_name_filter = "SILICONFLOW"
                instance_extra_include = '%"region": "intl"%'
                instance_extra_exclude = None
            elif factory_name == "SILICONFLOW":
                provider_name_filter = "SILICONFLOW"
                instance_extra_include = None
                instance_extra_exclude = '%"region": "intl"%'
            else:
                provider_name_filter = factory_name
                instance_extra_include = None
                instance_extra_exclude = None
            # Query provider for this factory
            cursor = self.db.execute_sql("SELECT id FROM tenant_model_provider WHERE provider_name = %s", (provider_name_filter,))
            providers = cursor.fetchall()
            for (provider_id,) in providers:
                # Query instances for this provider, with optional extra include/exclude filter
                if instance_extra_include and instance_extra_exclude:
                    cursor = self.db.execute_sql(
                        "SELECT id FROM tenant_model_instance WHERE provider_id = %s AND extra LIKE %s AND extra NOT LIKE %s", (provider_id, instance_extra_include, instance_extra_exclude)
                    )
                elif instance_extra_include:
                    cursor = self.db.execute_sql("SELECT id FROM tenant_model_instance WHERE provider_id = %s AND extra LIKE %s", (provider_id, instance_extra_include))
                elif instance_extra_exclude:
                    cursor = self.db.execute_sql("SELECT id FROM tenant_model_instance WHERE provider_id = %s AND extra NOT LIKE %s", (provider_id, instance_extra_exclude))
                else:
                    cursor = self.db.execute_sql("SELECT id FROM tenant_model_instance WHERE provider_id = %s", (provider_id,))
                instances = cursor.fetchall()
                for (instance_id,) in instances:
                    for llm in llm_list:
                        model_name = llm.get("llm_name", "")
                        model_types = llm.get("model_type", "chat")
                        if isinstance(model_types, str):
                            model_types = [model_types]
                        for mt in model_types:
                            cursor = self.db.execute_sql(
                                "SELECT COUNT(*) FROM tenant_model WHERE provider_id = %s AND instance_id = %s AND model_name = %s AND model_type = %s", (provider_id, instance_id, model_name, mt)
                            )
                            if cursor.fetchone()[0] == 0:
                                total_missing += 1

        if total_missing == 0:
            self.mark_noop_completes_migration()
            logger.info("No missing models to seed in tenant_model")
            return False

        logger.info(f"Found {total_missing} missing models to seed in tenant_model")
        return True

    def execute(self) -> tuple[int, list]:
        """Execute migration"""
        current_ts = self.current_timestamp()
        rows_inserted = 0

        if not self.db.table_exists("tenant_model_provider"):
            logger.error("Dependency table 'tenant_model_provider' does not exist")
            return 0, []

        if not self.db.table_exists("tenant_model_instance"):
            logger.error("Dependency table 'tenant_model_instance' does not exist")
            return 0, []

        if not self.db.table_exists("tenant_model"):
            if self.dry_run:
                logger.info("[DRY RUN] Target table 'tenant_model' does not exist")
                return 0, []
            logger.info("Target table 'tenant_model' does not exist, will create")
            self.create_target_table()

        if self.create_table_only:
            logger.info("[CREATE TABLE ONLY] Target table created/verified, skipping data seeding")
            return 0, self.target_tables

        # Pre-load existing extra values for model_names from tenant_model
        # to reuse non-default extra for same model_name across providers
        cursor = self.db.execute_sql("SELECT model_name, extra FROM tenant_model WHERE extra != '{}' AND extra IS NOT NULL")
        existing_extra_map = {}
        for model_name, extra in cursor.fetchall():
            if model_name and extra and extra != "{}":
                existing_extra_map[model_name] = extra

        # Pre-load existing status values for model_names from tenant_model
        # Prefer non-unsupported status when seeding new records for the same model_name
        cursor = self.db.execute_sql("SELECT model_name, status FROM tenant_model WHERE status != 'unsupported' AND status IS NOT NULL")
        existing_status_map = {}
        for model_name, status in cursor.fetchall():
            if model_name and status:
                existing_status_map[model_name] = status

        factories = self._load_llm_factories()
        all_records = []

        for factory in factories:
            llm_list = factory.get("llm", [])
            if not llm_list:
                continue
            factory_name = factory["name"]

            # Determine the provider_name to query and optional instance extra filter
            if factory_name == "siliconflow_intl":
                provider_name_filter = "SILICONFLOW"
                instance_extra_include = '%"region": "intl"%'
                instance_extra_exclude = None
            elif factory_name == "SILICONFLOW":
                provider_name_filter = "SILICONFLOW"
                instance_extra_include = None
                instance_extra_exclude = '%"region": "intl"%'
            else:
                provider_name_filter = factory_name
                instance_extra_include = None
                instance_extra_exclude = None

            # Query provider for this factory
            cursor = self.db.execute_sql("SELECT id FROM tenant_model_provider WHERE provider_name = %s", (provider_name_filter,))
            providers = cursor.fetchall()

            for (provider_id,) in providers:
                # Query instances for this provider, with optional extra include/exclude filter
                if instance_extra_include and instance_extra_exclude:
                    cursor = self.db.execute_sql(
                        "SELECT id FROM tenant_model_instance WHERE provider_id = %s AND extra LIKE %s AND extra NOT LIKE %s", (provider_id, instance_extra_include, instance_extra_exclude)
                    )
                elif instance_extra_include:
                    cursor = self.db.execute_sql("SELECT id FROM tenant_model_instance WHERE provider_id = %s AND extra LIKE %s", (provider_id, instance_extra_include))
                elif instance_extra_exclude:
                    cursor = self.db.execute_sql("SELECT id FROM tenant_model_instance WHERE provider_id = %s AND extra NOT LIKE %s", (provider_id, instance_extra_exclude))
                else:
                    cursor = self.db.execute_sql("SELECT id FROM tenant_model_instance WHERE provider_id = %s", (provider_id,))
                instances = cursor.fetchall()
                if not instances:
                    logger.warning(f"No instances found for provider '{provider_name_filter}' (id={provider_id}), skipping")
                    continue

                for (instance_id,) in instances:
                    for llm in llm_list:
                        model_name = llm.get("llm_name", "")
                        if not model_name:
                            continue
                        model_types = llm.get("model_type", "chat")
                        if isinstance(model_types, str):
                            model_types = [model_types]

                        # Determine extra: reuse existing non-default extra for same model_name;
                        # otherwise build default from is_tools and max_tokens in the llm entry
                        if model_name in existing_extra_map:
                            extra = existing_extra_map[model_name]
                        else:
                            extra_dict = {}
                            if llm.get("is_tools") is True:
                                extra_dict["is_tools"] = True
                            max_tokens = llm.get("max_tokens")
                            if max_tokens is not None:
                                extra_dict["max_tokens"] = max_tokens
                            extra = json.dumps(extra_dict) if extra_dict else "{}"

                        # Determine status: reuse existing non-unsupported status for same model_name, else "active"
                        status = existing_status_map.get(model_name, "active")

                        for mt in model_types:
                            # Check if record already exists
                            cursor = self.db.execute_sql(
                                "SELECT COUNT(*) FROM tenant_model WHERE provider_id = %s AND instance_id = %s AND model_name = %s AND model_type = %s", (provider_id, instance_id, model_name, mt)
                            )
                            if cursor.fetchone()[0] > 0:
                                continue
                            all_records.append((model_name, provider_id, instance_id, mt, status, extra))

        if not all_records:
            logger.info("No missing models to seed")
            return 0, []

        logger.info(f"Seeding {len(all_records)} tenant_model records...")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would insert {len(all_records)} records")
            for model_name, provider_id, instance_id, mt, status, extra in all_records[:5]:
                logger.info(f"  model_name={model_name}, provider_id={provider_id}, instance_id={instance_id}, model_type={mt}, status={status}, extra={extra}")
            if len(all_records) > 5:
                logger.info(f"  ... and {len(all_records) - 5} more records")
            return len(all_records), self.target_tables

        # Insert records in batches
        batch_size = 100
        for i in range(0, len(all_records), batch_size):
            batch = all_records[i : i + batch_size]
            values = []
            for model_name, provider_id, instance_id, mt, status, extra in batch:
                record_id = self.generate_uuid()
                model_name_escaped = model_name.replace("'", "''") if model_name else ""
                mt_escaped = mt.replace("'", "''") if mt else ""
                extra_escaped = extra.replace("'", "''") if extra else "{}"
                status_escaped = status.replace("'", "''") if status else "active"
                values.append(
                    f"('{record_id}', '{model_name_escaped}', '{provider_id}', "
                    f"'{instance_id}', '{mt_escaped}', '{status_escaped}', "
                    f"'{extra_escaped}', "
                    f"{current_ts * 1000}, FROM_UNIXTIME({current_ts}), "
                    f"{current_ts * 1000}, FROM_UNIXTIME({current_ts}))"
                )

            insert_sql = f"""
                INSERT INTO tenant_model
                (id, model_name, provider_id, instance_id, model_type, status, extra,
                 create_time, create_date, update_time, update_date)
                VALUES {", ".join(values)}
            """
            self.db.execute_sql(insert_sql)
            rows_inserted += len(batch)
            logger.info(f"Inserted batch {i // batch_size + 1}: {len(batch)} records")

        return rows_inserted, self.target_tables

    def create_target_table(self):
        """Create tenant_model table"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS tenant_model (
            id VARCHAR(32) NOT NULL PRIMARY KEY,
            model_name VARCHAR(128),
            provider_id VARCHAR(32) NOT NULL,
            instance_id VARCHAR(32) NOT NULL,
            model_type VARCHAR(32) NOT NULL,
            status VARCHAR(32) DEFAULT 'active',
            extra VARCHAR(1024) DEFAULT '{}',
            create_time BIGINT,
            create_date DATETIME,
            update_time BIGINT,
            update_date DATETIME,
            INDEX idx_instance_id (instance_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        self.db.execute_sql(create_sql)
        logger.info("Created tenant_model table")


class ModelTypeMergeStage(MigrationStage):
    """Merge tenant_model rows by (provider_id, instance_id, model_name), converting model_type to binary integer"""

    name = "model_type_merge"
    description = "Merge tenant_model rows by provider_id/instance_id/model_name, converting model_type to binary integer"
    source_tables = ["tenant_model"]
    target_tables = ["tenant_model_merge_tmp"]

    # Mapping from LLMType string values to ModelTypeBinary integer values
    MODEL_TYPE_STR_TO_INT = {
        "chat": 1,  # 0b0000001
        "embedding": 2,  # 0b0000010
        "asr": 4,  # 0b0000100
        "speech2text": 4,  # 0b0000100 legacy alias
        "vision": 8,  # 0b0001000
        "image2text": 8,  # 0b0001000 legacy alias
        "rerank": 16,  # 0b0010000
        "tts": 32,  # 0b0100000
        "ocr": 64,  # 0b1000000
    }

    def current_timestamp(self) -> int:
        return int(time.time())

    def generate_uuid(self) -> str:
        """Generate 32-character UUID1"""
        return uuid.uuid1().hex

    def check(self) -> bool:
        """Check if migration is needed"""
        if not self.db.table_exists("tenant_model"):
            logger.warning("Source table 'tenant_model' does not exist")
            return False

        # If model_type is already INT, the merge has already been executed — no work needed
        model_type_dtype = self.db.get_column_type("tenant_model", "model_type")
        if model_type_dtype and model_type_dtype.lower() == "int":
            self.mark_noop_completes_migration()
            logger.info("tenant_model.model_type is already INT, merge stage is not applicable (already done)")
            return False

        return True

    def execute(self) -> tuple[int, list]:
        """Execute migration"""
        current_ts = self.current_timestamp()
        rows_inserted = 0

        if not self.db.table_exists("tenant_model"):
            logger.error("Source table 'tenant_model' does not exist")
            return 0, []

        # Create temporary table with model_type as INTEGER
        self.create_target_table()

        if self.create_table_only:
            logger.info("[CREATE TABLE ONLY] Temporary table created, skipping data merge")
            return 0, self.target_tables

        # Load all records from tenant_model
        cursor = self.db.execute_sql("SELECT id, model_name, provider_id, instance_id, model_type, status, extra, create_time, create_date, update_time, update_date FROM tenant_model ORDER BY id")
        all_rows = cursor.fetchall()

        # Group by (provider_id, instance_id, model_name)
        from collections import defaultdict

        groups = defaultdict(list)
        for row in all_rows:
            id_, model_name, provider_id, instance_id, model_type_str, status, extra, create_time, create_date, update_time, update_date = row
            groups[(provider_id, instance_id, model_name)].append(row)

        # Merge each group into a single record
        merged_records = []
        for (provider_id, instance_id, model_name), rows in groups.items():
            # Compute binary model_type:
            #   OR all model_types where status != 'unsupported'
            #   Then AND with complement of OR of all model_types where status == 'unsupported'
            supported_bits = 0
            unsupported_bits = 0
            first_non_unsupported_status = None
            first_row = rows[0]

            for row in rows:
                _, _, _, _, model_type_str, status, _, _, _, _, _ = row
                type_bit = self.MODEL_TYPE_STR_TO_INT.get(model_type_str, 0)
                if status == "unsupported":
                    unsupported_bits |= type_bit
                else:
                    supported_bits |= type_bit
                    if first_non_unsupported_status is None:
                        first_non_unsupported_status = status

            merged_type = supported_bits & ~unsupported_bits
            merged_status = first_non_unsupported_status or "active"

            # Use first row's values for all other fields, but replace id, model_type, status
            id_, model_name, provider_id, instance_id, _, _, extra, create_time, create_date, update_time, update_date = first_row

            # Only include records whose merged status is "active"
            if merged_status != "active":
                continue

            merged_records.append((self.generate_uuid(), model_name, provider_id, instance_id, merged_type, merged_status, extra, create_time, create_date, update_time, update_date))

        if not merged_records:
            logger.info("No records to merge")
            return 0, []

        logger.info(f"Merging {len(all_rows)} rows into {len(merged_records)} rows...")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would insert {len(merged_records)} merged rows, then rename tables")
            for rec in merged_records[:5]:
                _, model_name, provider_id, instance_id, merged_type, status, extra, *_ = rec
                logger.info(f"  model_name={model_name}, provider_id={provider_id}, instance_id={instance_id}, model_type={merged_type}, status={status}, extra={extra}")
            if len(merged_records) > 5:
                logger.info(f"  ... and {len(merged_records) - 5} more rows")
            return len(merged_records), self.target_tables

        # Insert merged records into temporary table
        batch_size = 100
        for i in range(0, len(merged_records), batch_size):
            batch = merged_records[i : i + batch_size]
            values = []
            for rec in batch:
                id_, model_name, provider_id, instance_id, merged_type, status, extra, create_time, create_date, update_time, update_date = rec
                model_name_escaped = (model_name.replace("'", "''") if model_name else "") or None
                extra_escaped = extra.replace("'", "''") if extra else "{}"
                status_escaped = status.replace("'", "''") if status else "active"
                # Handle NULL date fields
                create_time_val = create_time if create_time is not None else current_ts * 1000
                update_time_val = update_time if update_time is not None else current_ts * 1000
                create_date_sql = f"FROM_UNIXTIME({int(create_time_val / 1000)})" if create_time_val else "NULL"
                update_date_sql = f"FROM_UNIXTIME({int(update_time_val / 1000)})" if update_time_val else "NULL"
                values.append(
                    f"('{id_}', '{model_name_escaped}', '{provider_id}', "
                    f"'{instance_id}', {merged_type}, '{status_escaped}', "
                    f"'{extra_escaped}', "
                    f"{create_time_val}, {create_date_sql}, "
                    f"{update_time_val}, {update_date_sql})"
                )

            insert_sql = f"""
                INSERT INTO tenant_model_merge_tmp
                (id, model_name, provider_id, instance_id, model_type, status, extra,
                 create_time, create_date, update_time, update_date)
                VALUES {", ".join(values)}
            """
            self.db.execute_sql(insert_sql)
            rows_inserted += len(batch)
            logger.info(f"Inserted batch {i // batch_size + 1}: {len(batch)} merged rows")

        # Rename original table to backup, then rename temp table to tenant_model
        logger.info("Renaming tables: tenant_model -> tenant_model_backup, tenant_model_merge_tmp -> tenant_model")
        self.db.execute_sql("DROP TABLE IF EXISTS tenant_model_backup")
        self.db.execute_sql("ALTER TABLE tenant_model RENAME TO tenant_model_backup")
        self.db.execute_sql("ALTER TABLE tenant_model_merge_tmp RENAME TO tenant_model")
        logger.info("Table rename completed successfully")

        return rows_inserted, ["tenant_model_merge_tmp", "tenant_model", "tenant_model_backup"]

    def create_target_table(self):
        """Create temporary table tenant_model_merge_tmp with model_type as INTEGER"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS tenant_model_merge_tmp (
            id VARCHAR(32) NOT NULL PRIMARY KEY,
            model_name VARCHAR(128),
            provider_id VARCHAR(32) NOT NULL,
            instance_id VARCHAR(32) NOT NULL,
            model_type INT NOT NULL,
            status VARCHAR(32) DEFAULT 'active',
            extra VARCHAR(1024) DEFAULT '{}',
            create_time BIGINT,
            create_date DATETIME,
            update_time BIGINT,
            update_date DATETIME,
            INDEX idx_instance_id (instance_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        self.db.execute_sql(create_sql)
        logger.info("Created tenant_model_merge_tmp table")


class TenantModelIdMigrationStage(MigrationStage):
    """Populate tenant_*_id columns in tenant, knowledgebase, dialog, memory with tenant_model.id values."""

    name = "tenant_model_id_migration"
    description = "Populate tenant_*_id columns with tenant_model.id, converting from IntegerField to CharField (VARCHAR(32))"
    source_tables = ["tenant_model", "tenant_model_provider", "tenant_model_instance", "tenant", "knowledgebase", "dialog", "memory"]
    target_tables = ["tenant", "knowledgebase", "dialog", "memory"]

    # Maps table -> list of (tenant_*_id column, legacy *_id column, model_type for lookup)
    # The legacy column holds the model_name (e.g. "gpt-4o@default@OpenAI"), we use it to find tenant_model.id
    TENANT_ID_FIELDS = {
        "tenant": [
            ("tenant_llm_id", "llm_id", "chat"),
            ("tenant_embd_id", "embd_id", "embedding"),
            ("tenant_asr_id", "asr_id", "speech2text"),
            ("tenant_img2txt_id", "img2txt_id", "image2text"),
            ("tenant_rerank_id", "rerank_id", "rerank"),
            ("tenant_tts_id", "tts_id", "tts"),
            ("tenant_ocr_id", "ocr_id", "ocr"),
        ],
        "knowledgebase": [
            ("tenant_embd_id", "embd_id", "embedding"),
        ],
        "dialog": [
            ("tenant_llm_id", "llm_id", "chat"),
            ("tenant_rerank_id", "rerank_id", "rerank"),
        ],
        "memory": [
            ("tenant_embd_id", "embd_id", "embedding"),
            ("tenant_llm_id", "llm_id", "chat"),
        ],
    }

    # model_type string -> binary integer mapping
    MODEL_TYPE_TO_INT = {
        "chat": 1,
        "embedding": 2,
        "asr": 4,
        "speech2text": 4,
        "vision": 8,
        "image2text": 8,
        "rerank": 16,
        "tts": 32,
        "ocr": 64,
    }

    scan_batch_size = 500

    @classmethod
    def canonical_model_type(cls, model_type: str) -> str:
        aliases = {"speech2text": "asr", "image2text": "vision"}
        normalized = (model_type or "").strip().lower()
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.MODEL_TYPE_TO_INT:
            raise ValueError(f"unsupported tenant model type: {model_type}")
        return normalized

    @classmethod
    def resolve_legacy_reference(
        cls,
        exact_mapping: dict,
        *,
        tenant_id: str,
        legacy_reference,
        model_type: str,
    ) -> str:
        key = (
            str(tenant_id),
            str(legacy_reference),
            cls.canonical_model_type(model_type),
        )
        resolved = exact_mapping.get(key)
        if not resolved:
            raise RuntimeError(
                "legacy tenant model reference is unresolved: "
                f"tenant={tenant_id} reference={legacy_reference} type={model_type}"
            )
        return resolved

    @classmethod
    def resolve_legacy_model_name(
        cls,
        exact_mapping: dict,
        *,
        tenant_id: str,
        legacy_model_name: str,
        model_type: str,
    ) -> str:
        pure_name, _, provider_name = cls._split_model_name_value(legacy_model_name)
        key = (
            str(tenant_id),
            pure_name,
            provider_name,
            cls.canonical_model_type(model_type),
        )
        resolved = exact_mapping.get(key)
        if not resolved:
            raise RuntimeError(
                "legacy tenant model name is unresolved: "
                f"tenant={tenant_id} model={legacy_model_name} type={model_type}"
            )
        return resolved

    def _build_exact_source_mapping(self) -> tuple[dict, dict]:
        cursor = self.db.execute_sql(
            "SELECT id, tenant_id, llm_factory, llm_name, model_type, api_key, api_base "
            "FROM tenant_llm WHERE status = '1' ORDER BY id"
        )
        source_models = cursor.fetchall()

        cursor = self.db.execute_sql(
            "SELECT id, tenant_id, provider_name FROM tenant_model_provider"
        )
        providers = {}
        for provider_id, tenant_id, provider_name in cursor.fetchall():
            key = (str(tenant_id), provider_name)
            if key in providers and providers[key] != provider_id:
                raise RuntimeError("tenant_model_provider mapping is ambiguous")
            providers[key] = provider_id

        cursor = self.db.execute_sql(
            "SELECT id, provider_id, api_key, extra FROM tenant_model_instance WHERE status = 'active'"
        )
        instances = {}
        for instance_id, provider_id, api_key, extra in cursor.fetchall():
            identity = TenantModelInstanceStage.instance_identity(
                provider_id,
                api_key,
                TenantModelInstanceStage._base_url_from_extra(extra),
            )
            if identity in instances and instances[identity] != instance_id:
                raise RuntimeError("tenant_model_instance mapping is ambiguous")
            instances[identity] = instance_id

        cursor = self.db.execute_sql(
            "SELECT id, provider_id, instance_id, model_name, model_type "
            "FROM tenant_model WHERE status = 'active'"
        )
        models = {}
        valid_new_references = {}
        provider_tenants = {provider_id: tenant_id for (tenant_id, _), provider_id in providers.items()}
        for model_id, provider_id, instance_id, model_name, model_type in cursor.fetchall():
            key = (provider_id, instance_id, model_name)
            if key in models and models[key][0] != model_id:
                raise RuntimeError("tenant_model mapping is ambiguous")
            models[key] = (model_id, int(model_type))
            tenant_id = provider_tenants.get(provider_id)
            if not tenant_id:
                raise RuntimeError("tenant_model provider ownership is unresolved")
            for type_name, type_bit in self.MODEL_TYPE_TO_INT.items():
                canonical_type = self.canonical_model_type(type_name)
                if int(model_type) & type_bit:
                    valid_new_references[(str(tenant_id), str(model_id), canonical_type)] = str(model_id)

        exact_mapping = dict(valid_new_references)
        exact_name_mapping = {}
        for source_id, tenant_id, factory, model_name, model_type, api_key, api_base in source_models:
            provider_id = providers.get((str(tenant_id), factory))
            if not provider_id:
                raise RuntimeError(f"tenant_model_provider mapping is incomplete for tenant_llm id={source_id}")
            identity = TenantModelInstanceStage.instance_identity(provider_id, api_key, api_base)
            instance_id = instances.get(identity)
            if not instance_id:
                raise RuntimeError(f"tenant_model_instance mapping is incomplete for tenant_llm id={source_id}")
            model = models.get((provider_id, instance_id, model_name))
            canonical_type = self.canonical_model_type(model_type)
            type_bit = self.MODEL_TYPE_TO_INT[canonical_type]
            if not model or not (model[1] & type_bit):
                raise RuntimeError(f"tenant_model mapping is incomplete for tenant_llm id={source_id}")
            key = (str(tenant_id), str(source_id), canonical_type)
            if key in exact_mapping and exact_mapping[key] != str(model[0]):
                raise RuntimeError("legacy tenant model reference mapping is ambiguous")
            exact_mapping[key] = str(model[0])
            name_key = (str(tenant_id), model_name, factory, canonical_type)
            if name_key in exact_name_mapping and exact_name_mapping[name_key] != str(model[0]):
                raise RuntimeError("legacy tenant model name mapping is ambiguous")
            exact_name_mapping[name_key] = str(model[0])
        return exact_mapping, exact_name_mapping

    def _collect_reference_updates(self, exact_mapping: dict, exact_name_mapping: dict) -> tuple[list, list]:
        updates = []
        columns_to_add = []
        for table_name, fields in self.TENANT_ID_FIELDS.items():
            if not self.db.table_exists(table_name):
                continue
            for tenant_id_col, legacy_col, model_type_str in fields:
                target_exists = self.db.column_exists(table_name, tenant_id_col)
                legacy_exists = self.db.column_exists(table_name, legacy_col)
                if not target_exists:
                    columns_to_add.append((table_name, tenant_id_col))
                if not legacy_exists and not target_exists:
                    continue

                target_expr = f"`{tenant_id_col}`" if target_exists else "NULL"
                legacy_expr = f"`{legacy_col}`" if legacy_exists else "NULL"
                tenant_expr = "id" if table_name == "tenant" else "tenant_id"
                cursor = self.db.execute_sql(
                    f"SELECT id, `{tenant_expr}`, {target_expr}, {legacy_expr} FROM `{table_name}`"
                )
                for row_id, tenant_id, current_ref, model_name in cursor.fetchall():
                    if current_ref is None or str(current_ref).strip() == "":
                        if not model_name or not str(model_name).strip():
                            continue
                        resolved_id = self.resolve_legacy_model_name(
                            exact_name_mapping,
                            tenant_id=str(tenant_id),
                            legacy_model_name=str(model_name),
                            model_type=model_type_str,
                        )
                    else:
                        resolved_id = self.resolve_legacy_reference(
                            exact_mapping,
                            tenant_id=str(tenant_id),
                            legacy_reference=current_ref,
                            model_type=model_type_str,
                        )
                    if str(current_ref or "") != resolved_id:
                        updates.append((table_name, tenant_id_col, row_id, resolved_id))
        return columns_to_add, updates

    def check(self) -> bool:
        """Check if migration is needed - any tenant_*_id column is still INT or empty."""
        if not self.db.table_exists("tenant_model"):
            logger.warning("Dependency table 'tenant_model' does not exist")
            return False

        if not self.db.table_exists("tenant_model_provider"):
            logger.warning("Dependency table 'tenant_model_provider' does not exist")
            return False

        if not self.db.table_exists("tenant_model_instance"):
            logger.warning("Dependency table 'tenant_model_instance' does not exist")
            return False

        has_work = False
        for table_name, fields in self.TENANT_ID_FIELDS.items():
            if not self.db.table_exists(table_name):
                continue
            for tenant_id_col, _, _ in fields:
                if not self.db.column_exists(table_name, tenant_id_col):
                    # Column does not exist yet — needs to be added
                    has_work = True
                    break
                # Check if column is still INT type -> needs conversion
                col_type = self.db.get_column_type(table_name, tenant_id_col)
                if col_type and col_type.lower() in ("int", "bigint", "integer"):
                    has_work = True
                    break
                _, legacy_col, _ = next(field for field in fields if field[0] == tenant_id_col)
                if not self.db.column_exists(table_name, legacy_col):
                    continue
                cursor = self.db.execute_sql(
                    f"SELECT COUNT(*) FROM `{table_name}` WHERE "
                    f"(`{tenant_id_col}` IS NOT NULL AND `{tenant_id_col}` != '' AND LENGTH(`{tenant_id_col}`) <> 32) "
                    f"OR ((`{tenant_id_col}` IS NULL OR `{tenant_id_col}` = '') "
                    f"AND `{legacy_col}` IS NOT NULL AND `{legacy_col}` != '')"
                )
                null_count = cursor.fetchone()[0]
                if null_count > 0:
                    has_work = True
                    break
            if has_work:
                break

        if not has_work:
            self.mark_noop_completes_migration()
            logger.info("All tenant_*_id columns are already VARCHAR(32) and populated, no migration needed")
            return False

        return True

    def execute(self) -> tuple[int, list]:
        """Validate every reference first, then alter columns and write exact IDs."""
        rows_updated = 0
        tables_operated = set()

        if self.create_table_only:
            logger.info("[CREATE TABLE ONLY] No tables are created for this data migration")
            return 0, []

        exact_mapping, exact_name_mapping = self._build_exact_source_mapping()
        columns_to_add, updates = self._collect_reference_updates(exact_mapping, exact_name_mapping)

        if self.dry_run:
            logger.info(
                "[DRY RUN] Validated full tenant model chain; would add %s columns and update %s rows",
                len(columns_to_add),
                len(updates),
            )
            return len(updates), sorted({item[0] for item in columns_to_add + updates})

        # All source and target references are proven before the first DDL or DML statement.
        for table_name, fields in self.TENANT_ID_FIELDS.items():
            if not self.db.table_exists(table_name):
                continue
            for tenant_id_col, _, _ in fields:
                if not self.db.column_exists(table_name, tenant_id_col):
                    # Column does not exist yet — add it as VARCHAR(32)
                    logger.info(f"Adding column {table_name}.{tenant_id_col} as VARCHAR(32) NULL")
                    self.db.execute_sql(f"ALTER TABLE `{table_name}` ADD COLUMN `{tenant_id_col}` VARCHAR(32) NULL")
                    tables_operated.add(table_name)
                    continue
                col_type = self.db.get_column_type(table_name, tenant_id_col)
                if col_type and col_type.lower() in ("int", "bigint", "integer"):
                    logger.info(f"Converting {table_name}.{tenant_id_col} from {col_type} to VARCHAR(32)")
                    self.db.execute_sql(f"ALTER TABLE `{table_name}` MODIFY COLUMN `{tenant_id_col}` VARCHAR(32) NULL")
                    tables_operated.add(table_name)

        for table_name, tenant_id_col, row_id, resolved_id in updates:
            self.db.execute_sql(
                f"UPDATE `{table_name}` SET `{tenant_id_col}` = %s WHERE id = %s",
                (resolved_id, row_id),
            )
            rows_updated += 1
            tables_operated.add(table_name)

        logger.info(f"Updated {rows_updated} rows with tenant_model.id values")

        return rows_updated, sorted(tables_operated)

    @staticmethod
    def _split_model_name_value(model_name: str) -> tuple[str, str, str]:
        """Parse model_name: {model_name}@{factory_name} or {model_name}@{instance_name}@{factory_name}"""
        if not model_name:
            return "", "", ""
        parts = model_name.rsplit("@", 2)
        if len(parts) == 1:
            return parts[0], "default", ""
        elif len(parts) == 2:
            return parts[0], "default", parts[1]
        else:
            return parts[0], parts[1], parts[2]

    def _split_model_name(self, model_name: str) -> tuple[str, str, str]:
        return self._split_model_name_value(model_name)

    def _build_model_lookup(self) -> dict:
        """Build a lookup dict: (tenant_id, model_name_pure, provider_name, model_type_int) -> tenant_model.id"""
        cursor = self.db.execute_sql(
            "SELECT tm.id, tm.model_name, tm.model_type, tmp.tenant_id, tmp.provider_name "
            "FROM tenant_model tm "
            "INNER JOIN tenant_model_provider tmp ON tm.provider_id = tmp.id "
            "WHERE tm.status = 'active'"
        )
        cursor = self.db.execute_sql(
            "SELECT tm.id, tm.model_name, tm.model_type, tmp.tenant_id, tmp.provider_name, tmi.instance_name "
            "FROM tenant_model tm "
            "INNER JOIN tenant_model_provider tmp ON tm.provider_id = tmp.id "
            "INNER JOIN tenant_model_instance tmi ON tm.instance_id = tmi.id "
            "WHERE tm.status = 'active' AND tmi.status = 'active'"
        )
        lookup = {}
        for model_id, model_name, model_type, tenant_id, provider_name, instance_name in cursor.fetchall():
            # model_type is a binary integer; we check each bit
            for type_str, type_bit in self.MODEL_TYPE_TO_INT.items():
                if model_type & type_bit:
                    key = (str(tenant_id), model_name, instance_name, provider_name, self.canonical_model_type(type_str))
                    if key in lookup and lookup[key] != model_id:
                        raise RuntimeError("tenant model name mapping is ambiguous")
                    lookup[key] = model_id
        return lookup

    def _resolve_model_id(self, model_lookup: dict, tenant_id: str, model_name: str, model_type_str: str) -> str | None:
        """Look up tenant_model.id from the model_lookup cache."""
        pure_name, instance_name, provider_name = self._split_model_name(model_name)
        if not pure_name or not provider_name:
            logger.warning(f"Cannot resolve model id for {model_name}: missing model_name or provider")
            return None

        # Try exact lookup with the provider name from the model_name
        key = (str(tenant_id), pure_name, instance_name, provider_name, self.canonical_model_type(model_type_str))
        result = model_lookup.get(key)
        if result:
            return result

        logger.warning(f"No tenant_model.id found for tenant={tenant_id}, model={model_name}, type={model_type_str}")
        return None


class TabularStructureDiscoveryIndexStage(MigrationStage):
    """Create the MySQL 8 ngram index used only for bounded table recall."""

    INDEX_SCHEMA_VERSION = "tabular-structure-index/v2"
    TABLE_REF_COLUMN_CONTRACT = ("varchar", 512, "ascii", "ascii_bin")
    name = "tabular_structure_discovery_index"
    description = "Create the versioned tabular structure discovery index"
    source_tables = ["tabular_structure_generation", "document", "knowledgebase"]
    target_tables = [
        "tabular_structure_dataset_index_state",
        "tabular_structure_table_index",
    ]

    def _require_supported_backend(self):
        version_row = self.db.execute_sql("SELECT VERSION()").fetchone()
        version = str(version_row[0] if version_row else "")
        if not version.startswith("8.0."):
            raise RuntimeError("discovery_unsupported_backend: MySQL 8.0 is required")
        plugin = self.db.execute_sql(
            "SELECT PLUGIN_STATUS FROM INFORMATION_SCHEMA.PLUGINS "
            "WHERE PLUGIN_NAME = 'ngram' AND PLUGIN_TYPE = 'FTPARSER'"
        ).fetchone()
        if not plugin or str(plugin[0]).upper() != "ACTIVE":
            raise RuntimeError("discovery_unsupported_backend: MySQL ngram parser is unavailable")

    def _fulltext_index_exists(self) -> bool:
        if not self.db.table_exists("tabular_structure_table_index"):
            return False
        cursor = self.db.execute_sql(
            "SELECT COUNT(*) FROM information_schema.statistics "
            "WHERE table_schema = %s AND table_name = %s "
            "AND index_name = %s AND index_type = 'FULLTEXT'",
            (
                self.db.config.database,
                "tabular_structure_table_index",
                "ft_tabular_structure_search_text",
            ),
        )
        return cursor.fetchone()[0] > 0

    def _table_ref_column_contract(self) -> tuple[str, int, str, str] | None:
        if not self.db.table_exists("tabular_structure_table_index"):
            return None
        row = self.db.execute_sql(
            "SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, CHARACTER_SET_NAME, COLLATION_NAME "
            "FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s AND column_name = %s",
            (
                self.db.config.database,
                "tabular_structure_table_index",
                "table_ref",
            ),
        ).fetchone()
        if not row:
            return None
        return (
            str(row[0]).lower(),
            int(row[1]),
            str(row[2]).lower(),
            str(row[3]).lower(),
        )

    def _has_stale_index_schema_state(self) -> bool:
        if not self.db.table_exists("tabular_structure_dataset_index_state"):
            return False
        return (
            self.db.execute_sql(
                "SELECT 1 FROM tabular_structure_dataset_index_state "
                "WHERE index_schema_version <> %s LIMIT 1",
                (self.INDEX_SCHEMA_VERSION,),
            ).fetchone()
            is not None
        )

    def _active_generation_lacks_index_state(self) -> bool:
        if not self.db.table_exists("tabular_structure_dataset_index_state"):
            return True
        return (
            self.db.execute_sql(
                "SELECT 1 FROM tabular_structure_generation g "
                "LEFT JOIN tabular_structure_dataset_index_state s "
                "ON s.tenant_id = g.tenant_id AND s.kb_id = g.kb_id "
                "WHERE g.status = 'active' AND s.tenant_id IS NULL LIMIT 1"
            ).fetchone()
            is not None
        )

    def check(self) -> bool:
        self._require_supported_backend()
        missing_schema = (
            any(not self.db.table_exists(table) for table in self.target_tables)
            or not self._fulltext_index_exists()
        )
        if missing_schema:
            return True
        if self._table_ref_column_contract() != self.TABLE_REF_COLUMN_CONTRACT:
            return True
        if self._has_stale_index_schema_state():
            return True
        return self._active_generation_lacks_index_state()

    def execute(self) -> tuple[int, list]:
        self._require_supported_backend()
        if self.dry_run:
            return 0, self.target_tables
        index_table_existed = self.db.table_exists("tabular_structure_table_index")
        state_table_existed = self.db.table_exists(
            "tabular_structure_dataset_index_state"
        )
        table_ref_contract = self._table_ref_column_contract()
        stale_index_state = self._has_stale_index_schema_state()
        requires_reprojection = (index_table_existed or state_table_existed) and (
            not index_table_existed
            or not state_table_existed
            or table_ref_contract != self.TABLE_REF_COLUMN_CONTRACT
            or stale_index_state
        )
        self.create_target_table()
        if (
            index_table_existed
            and table_ref_contract != self.TABLE_REF_COLUMN_CONTRACT
        ):
            self.db.execute_sql(
                "ALTER TABLE tabular_structure_table_index MODIFY table_ref VARCHAR(512) "
                "CHARACTER SET ascii COLLATE ascii_bin NOT NULL"
            )
        if requires_reprojection:
            # A truncated opaque identity cannot be reconstructed from the index.
            # Mark every affected dataset pending so startup backfill reads Active manifests.
            with self.db.atomic():
                self.db.execute_sql("DELETE FROM tabular_structure_table_index")
                self.db.execute_sql(
                    "UPDATE tabular_structure_dataset_index_state "
                    "SET index_revision = index_revision + 1, "
                    "backfill_status = 'pending', backfill_cursor = NULL, "
                    "index_schema_version = 'tabular-structure-index/v2'"
                )
        if not self._fulltext_index_exists():
            self.db.execute_sql(
                "ALTER TABLE tabular_structure_table_index "
                "ADD FULLTEXT INDEX ft_tabular_structure_search_text (search_text) WITH PARSER ngram"
            )
        self.db.execute_sql(
            "INSERT INTO tabular_structure_dataset_index_state "
            "(tenant_id, kb_id, index_revision, backfill_status, backfill_cursor, index_schema_version) "
            "SELECT DISTINCT tenant_id, kb_id, 0, 'pending', NULL, 'tabular-structure-index/v2' "
            "FROM tabular_structure_generation WHERE status = 'active' "
            "ON DUPLICATE KEY UPDATE index_schema_version = VALUES(index_schema_version)"
        )
        return 0, self.target_tables

    def create_target_table(self):
        self.db.execute_sql(
            """
            CREATE TABLE IF NOT EXISTS tabular_structure_dataset_index_state (
                tenant_id VARCHAR(32) NOT NULL,
                kb_id VARCHAR(256) NOT NULL,
                index_revision BIGINT UNSIGNED NOT NULL DEFAULT 0,
                backfill_status VARCHAR(16) NOT NULL DEFAULT 'pending',
                backfill_cursor VARCHAR(32) NULL,
                index_schema_version VARCHAR(64) NOT NULL,
                create_time BIGINT NULL,
                create_date DATETIME NULL,
                update_time BIGINT NULL,
                update_date DATETIME NULL,
                PRIMARY KEY (tenant_id, kb_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        self.db.execute_sql(
            """
            CREATE TABLE IF NOT EXISTS tabular_structure_table_index (
                tenant_id VARCHAR(32) NOT NULL,
                kb_id VARCHAR(256) NOT NULL,
                document_id VARCHAR(32) NOT NULL,
                producer_generation_ref VARCHAR(36) NOT NULL,
                table_ref VARCHAR(512) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
                table_ordinal INT UNSIGNED NOT NULL,
                search_text TEXT NOT NULL,
                identity_hash CHAR(64) NOT NULL,
                index_revision BIGINT UNSIGNED NOT NULL,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                projection_status VARCHAR(16) NOT NULL DEFAULT 'safe',
                unsafe_reason VARCHAR(64) NULL,
                create_time BIGINT NULL,
                create_date DATETIME NULL,
                update_time BIGINT NULL,
                update_date DATETIME NULL,
                PRIMARY KEY (tenant_id, kb_id, document_id, producer_generation_ref, table_ref),
                INDEX idx_tabular_structure_dataset_revision (tenant_id, kb_id, active, index_revision),
                INDEX idx_tabular_structure_document (document_id),
                INDEX idx_tabular_structure_identity (identity_hash)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )


# Registry of available migration stages
MIGRATION_STAGES = {
    "tabular_structure_discovery_index": TabularStructureDiscoveryIndexStage,
    "tenant_model_contract_preflight": TenantModelContractPreflightStage,
    "tenant_model_provider": TenantModelProviderStage,
    "tenant_model_instance": TenantModelInstanceStage,
    "tenant_model": TenantModelStage,
    "model_id_config": ModelIdConfigStage,
    "tenant_model_seeding": TenantModelSeedingStage,
    "model_type_merge": ModelTypeMergeStage,
    "tenant_model_id_migration": TenantModelIdMigrationStage,
}


def list_available_stages():
    """List all available migration stages"""
    logger.info("Available migration stages:")
    for name, stage_cls in MIGRATION_STAGES.items():
        logger.info(f"  - {name}: {stage_cls.description}")
        logger.info(f"    Source tables: {stage_cls.source_tables}")
        logger.info(f"    Target tables: {stage_cls.target_tables}")


def run_migration(
    config: MigrationConfig,
    stages: list,
    dry_run: bool = True,
    create_table_only: bool = False,
    database_version: str | None = None,
    mark_database_version_on_success: bool = False,
):
    """Run migration with specified stages"""
    stats = MigrationStats()
    stats.start()

    db = MigrationDatabase(config)

    try:
        db.connect()

        if database_version:
            current_db_version = db.get_database_version()
            if should_skip_migration(current_db_version, database_version):
                logger.info(
                    "Database migration version is %s, target version is %s, skipping all stages",
                    current_db_version,
                    database_version,
                )
                return

            if current_db_version:
                logger.info(
                    "Current database migration version is %s, target version is %s",
                    current_db_version,
                    database_version,
                )
            else:
                logger.info(
                    "Database migration version marker is not set, target version is %s",
                    database_version,
                )

        total_stages = len(stages)
        all_stages_completed = True

        for idx, stage_name in enumerate(stages, 1):
            logger.info(f"{'=' * 60}")
            logger.info(f"Stage [{idx}/{total_stages}]: {stage_name}")
            logger.info(f"{'=' * 60}")

            if stage_name not in MIGRATION_STAGES:
                logger.error(f"Unknown stage: {stage_name}")
                stats.add_stage_stats(stage_name, [], 0, 0)
                all_stages_completed = False
                continue

            stage_cls = MIGRATION_STAGES[stage_name]
            stage = stage_cls(db, dry_run=dry_run, create_table_only=create_table_only)

            stage_start = time.time()

            # For create_table_only mode, skip check and directly execute
            if create_table_only:
                logger.info("[CREATE TABLE ONLY] Skipping check, will create/verify target table")
                rows, tables = stage.execute()
            else:
                # Check if migration is needed
                if not stage.check():
                    logger.info(f"Stage '{stage_name}' check: no migration needed")
                    stats.add_stage_stats(stage_name, [], 0, time.time() - stage_start)
                    continue

                # Execute migration
                rows, tables = stage.execute()

            stage_duration = time.time() - stage_start

            stats.add_stage_stats(stage_name, tables, rows, stage_duration)
            logger.info(f"Stage '{stage_name}' completed: {rows} rows in {stage_duration:.2f}s")

        if mark_database_version_on_success and not dry_run and not create_table_only and database_version and all_stages_completed:
            db.set_database_version(database_version)
            logger.info("Marked database migration version as %s", database_version)

    finally:
        db.close()
        stats.end()
        stats.print_summary()


def check_database_version(config: MigrationConfig, target_version: str) -> int:
    db = MigrationDatabase(config)
    try:
        db.connect()
        current_db_version = db.get_database_version()
        if should_skip_migration(current_db_version, target_version):
            logger.info(
                "Database migration version is %s, target version is %s, migration is not needed",
                current_db_version,
                target_version,
            )
            return 0

        if current_db_version:
            logger.info(
                "Database migration version is %s, target version is %s, migration is needed",
                current_db_version,
                target_version,
            )
        else:
            logger.info(
                "Database migration version marker is not set, target version is %s, migration is needed",
                target_version,
            )
        return 1
    finally:
        db.close()


def mark_database_version(config: MigrationConfig, version: str) -> None:
    db = MigrationDatabase(config)
    try:
        db.connect()
        db.set_database_version(version)
        logger.info("Marked database migration version as %s", version)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="MySQL Data Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available stages
  python mysql_migration.py --list-stages

  # Check whether migration is needed for a target version
  python mysql_migration.py --check-database-version --database-version v0.26.4 --config /path/to/config.yaml

  # Mark database version separately
  python mysql_migration.py --mark-database-version --database-version v0.26.4 --config /path/to/config.yaml

  # Dry run (default - check only, no write) with config file
  python mysql_migration.py --stages tenant_model_provider --config /path/to/config.yaml

  # Dry run with command line MySQL connection
  python mysql_migration.py --stages tenant_model_provider --host localhost --port 3306 --user root --password secret

  # Create target tables only (no data migration)
  python mysql_migration.py --stages tenant_model_provider --config /path/to/config.yaml --create-table-only

  # Execute full migration (create tables and migrate data)
  python mysql_migration.py --stages tenant_model_provider --config /path/to/config.yaml --execute

  # Execute migration only when database version is lower than v0.26.4
  python mysql_migration.py --stages tenant_model_provider --config /path/to/config.yaml --execute --database-version v0.26.4

  # Execute migration and mark the database version when all stages succeed
  python mysql_migration.py --stages tenant_model_provider,tenant_model_instance,tenant_model,model_id_config --config /path/to/config.yaml --execute --database-version v0.26.4 --mark-database-version-on-success

  # Normalize legacy model IDs in stored configs
  python mysql_migration.py --stages model_id_config --config /path/to/config.yaml --execute

  # Run multiple stages
  python mysql_migration.py --stages stage1,stage2,stage3 --config /path/to/config.yaml --execute
""",
    )

    # MySQL connection options
    parser.add_argument("--host", type=str, default="localhost", help="MySQL host (default: localhost)")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port (default: 3306)")
    parser.add_argument("--user", type=str, default="root", help="MySQL user (default: root)")
    parser.add_argument("--password", type=str, default="", help="MySQL password (default: empty)")
    parser.add_argument("--database", type=str, default="rag_flow", help="MySQL database name (default: rag_flow)")

    # Configuration options
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")

    # Migration options
    parser.add_argument("--stages", "-s", type=str, help="Comma-separated list of stages to run")
    parser.add_argument("--list-stages", "-l", action="store_true", help="List available stages")
    parser.add_argument("--check-database-version", action="store_true", help="Check whether migration is needed for the target database version")
    parser.add_argument("--mark-database-version", action="store_true", help="Write the database migration version marker and exit")
    parser.add_argument("--database-version", type=str, metavar="VERSION", help="Database migration version used by check/mark commands and as the migration threshold for --stages")
    parser.add_argument("--mark-database-version-on-success", action="store_true", help="When used with --stages and --execute, write --database-version after all stages succeed")
    parser.add_argument("--execute", "-e", action="store_true", default=False, help="Execute full migration: create tables and migrate data")
    parser.add_argument("--create-table-only", action="store_true", default=False, help="Only create target tables, skip data migration")
    parser.add_argument(
        "--backfill-tabular-structure-index",
        action="store_true",
        help="Backfill the active tabular structure discovery index from immutable manifests",
    )

    args = parser.parse_args()

    # List stages and exit
    if args.list_stages:
        list_available_stages()
        return

    if args.backfill_tabular_structure_index:
        if args.execute or args.create_table_only or args.stages:
            logger.error(
                "--backfill-tabular-structure-index cannot be combined with migration stages or table modes"
            )
            sys.exit(1)
        from common import config_utils

        if args.config:
            loaded = config_utils.load_yaml_conf(args.config)
            if not isinstance(loaded, dict):
                raise RuntimeError("tabular discovery backfill config is invalid")
            config_utils.CONFIGS = loaded
        from common import settings

        settings.init_settings()
        from api.db.db_models import DB
        from api.db.services.tabular_structure_service import TabularStructureService

        with DB.lock("tabular_structure_discovery_index_backfill", -1):
            stats = TabularStructureService.backfill_active_generation_indexes(
                settings.STORAGE_IMPL,
            )
        logger.info("Tabular structure discovery index backfill completed: %s", stats)
        return

    # Load configuration: command line args take precedence over config file
    if args.config:
        config = MigrationConfig.from_config_file(args.config)
        # Override with command line args if provided
        if args.host != "localhost":
            config.host = args.host
        if args.port != 3306:
            config.port = args.port
        if args.user != "root":
            config.user = args.user
        if args.password != "":
            config.password = args.password
        if args.database != "rag_flow":
            config.database = args.database
    else:
        # Use command line args directly
        config = MigrationConfig(host=args.host, port=args.port, user=args.user, password=args.password, database=args.database)

    logger.info(f"MySQL Configuration: host={config.host}, port={config.port}, user={config.user}, database={config.database}")

    if args.check_database_version and args.mark_database_version:
        logger.error("--check-database-version and --mark-database-version are mutually exclusive")
        sys.exit(1)

    if args.check_database_version:
        if not args.database_version:
            logger.error("--check-database-version requires --database-version")
            sys.exit(1)
        sys.exit(check_database_version(config, args.database_version))

    if args.mark_database_version:
        if not args.database_version:
            logger.error("--mark-database-version requires --database-version")
            sys.exit(1)
        mark_database_version(config, args.database_version)
        return

    if args.mark_database_version_on_success and not args.database_version:
        logger.error("--mark-database-version-on-success requires --database-version")
        sys.exit(1)

    # Three mutually exclusive modes: dry-run (default), create-table-only, execute
    if args.execute and args.create_table_only:
        logger.error("--execute and --create-table-only are mutually exclusive")
        sys.exit(1)

    if not args.stages:
        logger.error("No stages specified. Use --stages to specify stages or --list-stages to see available stages.")
        sys.exit(1)

    stages = [s.strip() for s in args.stages.split(",")]
    dry_run = True
    create_table_only = False

    if args.create_table_only:
        logger.info("Running in CREATE TABLE ONLY mode (create tables, no data migration)")
        dry_run = False
        create_table_only = True
    elif args.execute:
        logger.info("Running in EXECUTE mode (create tables and migrate data)")
        dry_run = False
    else:
        logger.info("Running in DRY-RUN mode (check only, no write). Use --create-table-only to create tables, or --execute for full migration.")

    run_migration(
        config=config,
        stages=stages,
        dry_run=dry_run,
        create_table_only=create_table_only,
        database_version=args.database_version,
        mark_database_version_on_success=args.mark_database_version_on_success,
    )


if __name__ == "__main__":
    main()
