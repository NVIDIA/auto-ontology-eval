#!/usr/bin/env bash
# Run the remaining BEAVER eval questions as parallel shards, then merge.
#
# Questions 0-6 already completed in the sequential run and are preserved in
# input/beaverbench_nemotron_first7.csv, so sharding starts at index 7.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs input

START=${START:-7}
TOTAL=${TOTAL:-100}
SHARDS=${SHARDS:-4}
STAMP=$(date +%Y%m%d_%H%M%S)
VENV_PY="$PWD/.venv/bin/python"

# Sync once up front so the parallel shards never mutate the venv themselves.
uv sync --quiet
[ -x "$VENV_PY" ] || { echo "missing interpreter: $VENV_PY" >&2; exit 1; }

for c in neo4j postgres beaver-mysql; do
  docker start "$c" >/dev/null 2>&1 || true
done
sleep 4

caffeinate -imsw $$ &

REMAINING=$((TOTAL - START))
PER=$(( (REMAINING + SHARDS - 1) / SHARDS ))

pids=()
outputs=()
echo "Sharding questions ${START}..$((TOTAL - 1)) across ${SHARDS} processes (~${PER} each)"
for ((i = 0; i < SHARDS; i++)); do
  s=$((START + i * PER))
  e=$((s + PER))
  [ "$e" -gt "$TOTAL" ] && e=$TOTAL
  [ "$s" -ge "$TOTAL" ] && break

  out="input/beaverbench_shard${i}.csv"
  log="logs/beaver_shard${i}_${STAMP}.log"
  outputs+=("$out")

  echo "  shard $i: questions ${s}..$((e - 1))  -> $out"
  # Call the interpreter directly: concurrent `uv run` invocations each
  # reinstall the local project into the shared venv, which races and breaks
  # imports mid-run.
  PYTHONPATH=../GSF "$VENV_PY" scripts/run_eval_shard.py \
    --database-name beaverbench \
    --start-index "$s" \
    --end-index "$e" \
    --output "$out" \
    >"$log" 2>&1 &
  pids+=($!)
done

echo
echo "Waiting for ${#pids[@]} shards…"
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "shard pid $pid exited non-zero"; failed=1; }
done

produced=0
for out in "${outputs[@]}"; do
  [ -s "$out" ] && produced=$((produced + 1))
done
if [ "$produced" -eq 0 ]; then
  echo "ERROR: no shard produced output; refusing to merge/judge a partial set." >&2
  echo "Check logs/beaver_shard*_${STAMP}.log" >&2
  exit 1
fi

echo
echo "Merging ${produced} shard file(s) with the 7 sequential rows…"
"$VENV_PY" scripts/merge_eval_shards.py \
  --output input/beaverbench_nemotron.csv \
  input/beaverbench_nemotron_first7.csv "${outputs[@]}"

echo
echo "Verifying ground truth on merged rows…"
# evaluation.json already carries materialized answers, so shards write
# expected_answer_raw natively; this only repairs rows that predate that.
PYTHONPATH=../GSF "$VENV_PY" scripts/backfill_gold_answers.py \
  --csv input/beaverbench_nemotron.csv \
  --dataset datasets/beaverbench/evaluation.json \
  --skip-dataset

echo
echo "Running LLM judge…"
PYTHONPATH=../GSF "$VENV_PY" main.py \
  --database-name beaverbench \
  --skip-ingest --skip-semantic --skip-eval \
  --workers 4 2>&1 | tee "logs/beaver_judge_${STAMP}.log"

[ "$failed" -eq 0 ] || echo "NOTE: at least one shard reported an error; check logs/beaver_shard*.log"
echo "Done."
