#!/usr/bin/env bash
# Run the BEAVER eval as parallel shards, merge, judge, and score with the
# official BEAVER ex_acc. Output filenames derive from MODEL_NAME in .env
# (same slug rule as eval_chatbot), so the judge stage scores the same file
# the shards produced.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs input output

START=${START:-0}
SHARDS=${SHARDS:-4}
STAMP=$(date +%Y%m%d_%H%M%S)
VENV_PY="$PWD/.venv/bin/python"

# Sync once up front so the parallel shards never mutate the venv themselves.
uv sync --quiet
[ -x "$VENV_PY" ] || { echo "missing interpreter: $VENV_PY" >&2; exit 1; }

TOTAL=${TOTAL:-$("$VENV_PY" -c "import json; print(len(json.load(open('datasets/beaverbench/evaluation.json'))))")}
# Same slug rule as eval_chatbot: last '/' segment of MODEL_NAME, default nemotron.
MODEL_SLUG=$("$VENV_PY" -c "
from dotenv import dotenv_values
name = (dotenv_values('.env').get('MODEL_NAME') or 'nemotron')
print(name.rsplit('/', 1)[-1])
")
MERGED="input/beaverbench_${MODEL_SLUG}.csv"

for c in postgres beaver-mysql; do
  docker start "$c" >/dev/null 2>&1 || true
done
sleep 4

command -v caffeinate >/dev/null && caffeinate -imsw $$ &

REMAINING=$((TOTAL - START))
PER=$(( (REMAINING + SHARDS - 1) / SHARDS ))

pids=()
outputs=()
echo "Model slug: ${MODEL_SLUG}  ->  ${MERGED}"
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
  "$VENV_PY" scripts/run_eval_shard.py \
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
if [ "$produced" -ne "${#outputs[@]}" ]; then
  echo "ERROR: only ${produced}/${#outputs[@]} shards produced output; refusing to merge/judge a partial set." >&2
  echo "Check logs/beaver_shard*_${STAMP}.log" >&2
  exit 1
fi

echo
echo "Merging ${produced} shard file(s)…"
"$VENV_PY" scripts/merge_eval_shards.py \
  --output "$MERGED" \
  "${outputs[@]}"

echo
echo "Verifying ground truth on merged rows…"
# evaluation.json already carries materialized answers, so shards write
# expected_answer_raw natively; this only repairs rows that predate that.
"$VENV_PY" scripts/backfill_gold_answers.py \
  --csv "$MERGED" \
  --dataset datasets/beaverbench/evaluation.json \
  --skip-dataset

echo
echo "Running LLM judge…"
"$VENV_PY" main.py \
  --database-name beaverbench \
  --skip-ingest --skip-semantic --skip-eval \
  --workers 4 2>&1 | tee "logs/beaver_judge_${STAMP}.log"

echo
echo "Running official BEAVER ex_acc…"
"$VENV_PY" -m ontology_sql_eval.judge.beaver \
  --input "$MERGED" 2>&1 | tee "logs/beaver_exacc_${STAMP}.log"

[ "$failed" -eq 0 ] || echo "NOTE: at least one shard reported an error; check logs/beaver_shard*.log"
echo "Done."
