#!/usr/bin/env bash
# Run only the eval + judge stages. The semantic layer for BEAVER `dw` is
# already compiled (97/97 tables), so ingest and semantic compile are skipped.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/beaver_evalonly_$(date +%Y%m%d_%H%M%S).log"

for c in neo4j postgres beaver-mysql; do
  docker start "$c" >/dev/null 2>&1 || true
done
sleep 4

caffeinate -imsw $$ &

echo "Eval + judge starting. Log: $LOG"
PYTHONPATH=../GSF uv run python main.py \
  --database-name beaverbench \
  --skip-ingest \
  --skip-semantic \
  2>&1 | tee "$LOG"

echo
echo "Done. Scores are at the end of $LOG"
