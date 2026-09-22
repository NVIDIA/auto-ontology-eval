#!/usr/bin/env bash
# Resume the BEAVER benchmark after an interruption.
#
# The semantic compile is checkpointed in the GSF catalog: it only visits tables
# that have no Term yet, so re-running picks up where the previous run stopped.
# Ingest is skipped because the dw schema is already in the catalog.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/beaver_resume_$(date +%Y%m%d_%H%M%S).log"

echo "Waiting for network…"
until curl -sf -m 5 -o /dev/null https://inference-api.nvidia.com/v1/models \
    -H "Authorization: Bearer ${DEFAULT_MODELS_API_KEY:-}" 2>/dev/null \
    || curl -sf -m 5 -o /dev/null https://github.com; do
  sleep 5
done
echo "Network up."

for c in postgres beaver-mysql; do
  docker start "$c" >/dev/null 2>&1 || true
done
sleep 5

# Hold sleep off only for as long as this script runs.
caffeinate -imsw $$ &

echo "Resuming — remaining tables only, then eval + judge. Log: $LOG"
PYTHONPATH=../GSF uv run python main.py \
  --database-name beaverbench \
  --skip-ingest \
  2>&1 | tee "$LOG"

echo
echo "Done. Scores are at the end of $LOG"
