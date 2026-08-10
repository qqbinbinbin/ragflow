#!/bin/bash
# -----------------------------------------------------------------------------
# Shared migration script for model provider tables.
#
# Called by docker/entrypoint.sh and docker/launch_backend_service.sh.
# Keeps migration stages and versions in one place to avoid divergence.
#
# Usage:
#   PY=python3 tools/scripts/run_migrations.sh [--config CONFIG_PATH]
#
# Environment variables:
#   PY  - Python interpreter path (default: python3)
# -----------------------------------------------------------------------------

set -e

PY="${PY:-python3}"
CONFIG="${1:-conf/service_conf.yaml}"

echo "Running model provider table migrations..."

# Structure discovery is a MySQL 8-only capability. The stage performs the
# backend/ngram preflight before creating any discovery state.
"$PY" tools/scripts/mysql_migration.py \
    --stages tabular_structure_discovery_index \
    --config "$CONFIG" \
    --execute

# Read immutable Active manifests outside the DDL transaction, then commit each
# index batch only after rechecking the generation CAS in the service layer.
"$PY" tools/scripts/mysql_migration.py \
    --backfill-tabular-structure-index \
    --config "$CONFIG"

# The stages are idempotent and validate the complete legacy-to-current model
# mapping on every startup. A version marker must never hide partial schema/data.
"$PY" tools/scripts/mysql_migration.py \
    --stages tenant_model_contract_preflight,tenant_model_provider,tenant_model_instance,tenant_model,tenant_model_id_migration \
    --config "$CONFIG" \
    --execute

# Record completion only after all strict stages succeed.
"$PY" tools/scripts/mysql_migration.py \
    --config "$CONFIG" \
    --mark-database-version \
    --database-version "v0.27.0.dev2"

echo "Model provider table migrations completed."
