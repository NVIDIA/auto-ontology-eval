#!/bin/bash
# Provision (seed + ingest + semantically compile) a named dataset's GSF
# catalog. One generic script for every benchmark -- what varies per
# benchmark lives in configs/datasets/<name>.env, not in this script. To add
# a new benchmark, add a config file there; do not add a new script.
#
# Usage:
#   scripts/provision_catalog.sh <dataset-name> [--skip-seed]
#   scripts/provision_catalog.sh --list
#
# Each configs/datasets/<name>.env sets:
#   POSTGRES_DATABASE       -- target catalog DB name (must already exist and
#                              have GSF's schema migrated onto it -- this
#                              script does not create or migrate it; see
#                              GSF/docker-compose.yml's gsf-migrate service).
#   CONNECTION_STRINGS      -- comma-separated source DB URLs (static list),
#     or CONNECTION_STRINGS_CMD  -- a shell command that prints that same
#                              comma-separated list, for datasets whose DB set
#                              is discovered from disk rather than fixed.
#   SEED_CMD                -- (optional) command to fetch/prepare the source
#                              data before ingesting.
#
# Everything else (LLM/embedding keys, POSTGRES_HOST/PORT/USER/PASSWORD) comes
# from the repo's own .env, unchanged -- only the two dataset-varying values
# above are overlaid on top, and only for this script's own subprocesses. The
# real .env file on disk is never edited, so there is nothing to remember to
# switch back.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/configs/datasets"

list_datasets() {
    echo "available datasets:"
    local found=false
    for f in "$CONFIG_DIR"/*.env; do
        [ -e "$f" ] || continue
        found=true
        echo "  $(basename "$f" .env)"
    done
    [ "$found" = true ] || echo "  (none found in $CONFIG_DIR)"
}

usage() {
    echo "Usage: $(basename "$0") <dataset-name> [--skip-seed]"
    echo "       $(basename "$0") --list"
    echo
    list_datasets
}

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi
if [ "$1" = "--list" ]; then
    list_datasets
    exit 0
fi

DATASET="$1"
shift
SKIP_SEED=false
for arg in "$@"; do
    case "$arg" in
        --skip-seed) SKIP_SEED=true ;;
        *)
            echo "error: unknown argument '$arg'" >&2
            usage
            exit 1
            ;;
    esac
done

DATASET_CONFIG="$CONFIG_DIR/${DATASET}.env"
if [ ! -f "$DATASET_CONFIG" ]; then
    echo "error: no such dataset '$DATASET' (looked for $DATASET_CONFIG)" >&2
    echo >&2
    list_datasets >&2
    exit 1
fi

cd "$REPO_ROOT"

# Base .env first: shared LLM/embedding/POSTGRES_HOST-PORT-USER-PASSWORD
# creds. Optional -- some environments may set these directly.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Dataset overlay second, so its POSTGRES_DATABASE/CONNECTION_STRINGS win over
# anything the base .env happens to also set.
set -a
# shellcheck disable=SC1090
source "$DATASET_CONFIG"
set +a

if [ -n "${CONNECTION_STRINGS_CMD:-}" ]; then
    echo "[provision_catalog] resolving CONNECTION_STRINGS via: $CONNECTION_STRINGS_CMD"
    CONNECTION_STRINGS="$(eval "$CONNECTION_STRINGS_CMD")"
    export CONNECTION_STRINGS
fi

if [ -z "${CONNECTION_STRINGS:-}" ]; then
    echo "error: dataset '$DATASET' config set neither CONNECTION_STRINGS nor CONNECTION_STRINGS_CMD (or the latter produced nothing)" >&2
    exit 1
fi
if [ -z "${POSTGRES_DATABASE:-}" ]; then
    echo "error: dataset '$DATASET' config did not set POSTGRES_DATABASE" >&2
    exit 1
fi

echo "[provision_catalog] dataset=$DATASET postgres_database=$POSTGRES_DATABASE"
echo "[provision_catalog] $(echo "$CONNECTION_STRINGS" | tr ',' '\n' | wc -l | tr -d ' ') source DB(s)"

if [ "$SKIP_SEED" = false ] && [ -n "${SEED_CMD:-}" ]; then
    echo "[provision_catalog] seeding: $SEED_CMD"
    eval "$SEED_CMD"
else
    echo "[provision_catalog] skipping seed step"
fi

export PYTHONPATH="${PYTHONPATH:-../GSF}"

echo "[provision_catalog] ingesting into $POSTGRES_DATABASE ..."
uv run python -m ontology_sql_eval.ingestion.ingest

echo "[provision_catalog] compiling semantic layer in $POSTGRES_DATABASE ..."
uv run python -m ontology_sql_eval.ingestion.semantic

echo "[provision_catalog] done: $DATASET -> $POSTGRES_DATABASE"
