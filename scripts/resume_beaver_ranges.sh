#!/usr/bin/env bash
# One-off resume for the 2026-09-09 gpt-5.5 run that was OOM-killed at 64/100.
# Runs the missing ranges with bounded concurrency, then merges the preserved
# partial shards with the resumed ranges and runs judge + official ex_acc.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs input output

STAMP=$(date +%Y%m%d_%H%M%S)
VENV_PY="$PWD/.venv/bin/python"
MERGED="input/beaverbench_gpt-5.5.csv"
CONCURRENCY=2

# start:end(exclusive) pairs still missing
RANGES=("18:25" "42:50" "63:75" "91:100")

run_range() {
  local s=$1 e=$2
  local out="input/shards/resume_${s}_${e}.csv"
  local log="logs/beaver_resume_${s}_${e}_${STAMP}.log"
  echo "  range ${s}..$((e - 1)) -> $out"
  "$VENV_PY" scripts/run_eval_shard.py \
    --database-name beaverbench \
    --start-index "$s" \
    --end-index "$e" \
    --output "$out" \
    >"$log" 2>&1
}

echo "Resuming ${#RANGES[@]} ranges, ${CONCURRENCY} at a time"
pids=()
for r in "${RANGES[@]}"; do
  s=${r%%:*}; e=${r##*:}
  run_range "$s" "$e" &
  pids+=($!)
  if [ "${#pids[@]}" -ge "$CONCURRENCY" ]; then
    wait "${pids[0]}" || { echo "range starting at $s (or earlier) failed" >&2; exit 1; }
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "a resume range failed" >&2; exit 1; }
done

echo "Merging preserved partials + resumed ranges…"
"$VENV_PY" scripts/merge_eval_shards.py \
  --output "$MERGED" \
  input/shards/attempt1/beaverbench_shard0.csv \
  input/shards/attempt1/beaverbench_shard1.csv \
  input/shards/attempt1/beaverbench_shard2.csv \
  input/shards/attempt1/beaverbench_shard3.csv \
  input/shards/resume_*.csv

echo "Verifying ground truth on merged rows…"
"$VENV_PY" scripts/backfill_gold_answers.py \
  --csv "$MERGED" \
  --dataset datasets/beaverbench/evaluation.json \
  --skip-dataset

echo "Running LLM judge…"
"$VENV_PY" main.py \
  --database-name beaverbench \
  --skip-ingest --skip-semantic --skip-eval \
  --workers 4 2>&1 | tee "logs/beaver_judge_${STAMP}.log"

echo "Running official BEAVER ex_acc…"
"$VENV_PY" -m ontology_sql_eval.judge.beaver \
  --input "$MERGED" 2>&1 | tee "logs/beaver_exacc_${STAMP}.log"

echo "Done."
