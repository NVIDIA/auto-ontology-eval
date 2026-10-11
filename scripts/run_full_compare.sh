#!/usr/bin/env bash
# Full BIRD Dev comparison: schema-only control, then the ontology arm with the
# question-keyed prompt inputs, then the delta.
#
# The ontology arm receives exemplars and anchors and the control arm cannot,
# so the reported delta covers those inputs as well as the ontology grounding.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d-%H%M%S)"
CTRL="runs/full_${STAMP}/control/bird.jsonl"
TRT="runs/full_${STAMP}/treatment/bird.jsonl"
LOG_DIR="runs/full_${STAMP}/logs"
mkdir -p "$(dirname "$CTRL")" "$(dirname "$TRT")" "$LOG_DIR"

CFG="+auto_ontology_sql_resources_server.resources_servers.auto_ontology_sql"
CONC="${CONC:-10}"

# Gym caches each task file's aggregate metrics beside it and aborts with
# "Found conflicting aggregate metrics" when the task file has since changed --
# which it has whenever the question count differs from the last run.
rm -f resources_servers/*/data/tasks_metrics.json \
      resources_servers/*/data/tasks_metrics_conflict.json \
      resources_servers/*/data/tasks_prepare.jsonl

echo "=== control arm started $(date +%H:%M:%S) ==="
GYM_CONCURRENCY="$CONC" scripts/run_gym_arm.sh \
  schema_only_sql "$CTRL" \
  > "$LOG_DIR/control.log" 2>&1
echo "control exit=$? $(date +%H:%M:%S)"

echo "=== ontology arm started $(date +%H:%M:%S) ==="
GYM_CONCURRENCY="$CONC" scripts/run_gym_arm.sh \
  auto_ontology_sql "$TRT" --max-output-tokens 16 \
  "${CFG}.max_concurrency=${CONC}" \
  "${CFG}.sql_examples_csv=prompt_inputs/bird/predicted_structural_exemplars.csv" \
  "${CFG}.sql_examples_k=5" \
  "${CFG}.value_anchors_csv=prompt_inputs/bird/value_anchors.csv" \
  > "$LOG_DIR/treatment.log" 2>&1
echo "ontology exit=$? $(date +%H:%M:%S)"

echo "=== comparison ==="
.venv/bin/python scripts/compare_arms.py --control "$CTRL" --treatment "$TRT" \
  2>&1 | tee "$LOG_DIR/compare.txt"

echo "=== done $(date +%H:%M:%S) ==="
echo "artifacts under runs/full_${STAMP}/"
