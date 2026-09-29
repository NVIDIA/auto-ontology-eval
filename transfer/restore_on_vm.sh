#!/usr/bin/env bash
# Restore the GSF eval stores on a fresh machine from transfer/gsf_eval.dump.
#
# The dump carries the finished state of a full ingest + semantic compile of the
# 11 BIRD Mini-Dev databases: 873 data vectors, 746 semantic vectors, 75 catalog
# tables, 68 Terms, 671 ColumnAttributes. Restoring it skips re-running the
# semantic compile, which is the expensive step (LLM term extraction across all
# 75 tables) and the only one whose output would differ run to run.
#
# Usage:  ./transfer/restore_on_vm.sh [container_name] [host_port]
set -euo pipefail

CONTAINER="${1:-eval-postgres}"
PORT="${2:-55433}"
DUMP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gsf_eval.dump"

[ -f "$DUMP" ] || { echo "Dump not found: $DUMP" >&2; exit 1; }

echo "==> Starting $CONTAINER on :$PORT"
docker run -d --name "$CONTAINER" --restart unless-stopped \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=postgres -e POSTGRES_DB=gsf \
  -p "${PORT}:5432" pgvector/pgvector:pg17 >/dev/null

echo "==> Waiting for Postgres"
until docker exec "$CONTAINER" pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done

# pgvector must exist before restore: the dump's tables declare vector(2048)
# columns, and pg_restore cannot create a column of a type the database does
# not yet know about.
echo "==> Enabling pgvector"
docker exec "$CONTAINER" psql -U postgres -d gsf -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null

echo "==> Restoring dump"
docker cp "$DUMP" "$CONTAINER:/tmp/gsf_eval.dump"
docker exec "$CONTAINER" pg_restore -U postgres -d gsf --no-owner /tmp/gsf_eval.dump 2>&1 \
  | grep -vi "already exists" || true

echo "==> Verifying"
docker exec "$CONTAINER" psql -U postgres -d gsf -c "
  select (select count(*) from vdb.data_objects_layer) as data_vectors,
         (select count(*) from vdb.semantic_layer)     as semantic_vectors,
         (select count(*) from catalog_table)          as tables,
         (select count(*) from catalog_database)       as databases,
         (select count(*) from term)                   as terms,
         (select count(*) from column_attribute)       as column_attributes;"

cat <<'EOF'

Expected: 873 / 746 / 75 / 11 / 68 / 671

Remaining steps on the VM:
  1. Clone GSF at latest main next to this repo, so it resolves as ../GSF:
       git clone https://github.com/NVIDIA/GSF.git ../GSF && (cd ../GSF && git pull)
  2. Apply migrations. The dump was taken at revision 57abbbf6ff90; main has
     since added 8c3d5b17a204 (trigram indexes for global search). Without this
     the catalog is missing indexes that newer GSF expects:
       (cd ../GSF && uv run alembic upgrade head)
  3. Copy .env across (it holds the API keys and CONNECTION_STRINGS) and repoint
     the sqlite:// paths in CONNECTION_STRINGS at the VM's checkout location --
     they are absolute paths to the 11 BIRD .sqlite files.
  4. Fetch the BIRD Mini-Dev databases:  uv run python scripts/seed_bird.py
     The dump holds the *catalog* built from them, not the source DBs themselves;
     the eval executes generated SQL against the real SQLite files.
  5. Sanity-check before the long sweep:
       PYTHONPATH=../GSF uv run python -m ontology_sql_eval.retrieval.eval_chatbot \
         --input datasets/bird60/evaluation.json --output /tmp/smoke.csv \
         --workers 2 --limit 2 --run-id vm-smoke
  6. Run the sweep:
       PYTHONPATH=../GSF uv run python scripts/concurrency_sweep.py \
         --input datasets/bird60/evaluation.json --levels 1 2 3 4 5 6 7 8 \
         --sweep-id sweep-vm --resume
  7. Analyse:
       uv run python scripts/analyze_sweep.py --sweep-dir logs/sweep-vm

Neo4j is not required -- the eval was verified running with it stopped.

Verify the run picked up the new GSF behaviour: summary.json should show
  "evidence_as_parameter": true   (evidence sent as its own field, not glued
                                   onto the question text)
  "node_timing_mode": "phase"     (per-node timings measured start-to-end)
If either reads false/"gap", ../GSF is not on latest main.
EOF
