# SQL judge (standalone)

[← Back to main README](../../README.md)

Re-scores evaluation CSVs with an LLM that rates each row's SQL on logic,
semantics, and similarity to the ground truth. This workflow is fully
self-contained — no GSF/NeMo install, database, or vector stores required, only
a judge LLM API key.

The judge scores every CSV in `input/`. The retrieval eval writes its results
straight into `input/` (as `input/<db>_<model>.csv`), so after an eval run you
can judge with no manual copy. You can also drop in any CSV by hand. Each file
must contain `question`, `expected_sql`, and `returned_sql` columns (and
optionally `returned_answer`, used as a result preview). Rows with an empty
`returned_sql` are skipped. Scored CSVs are written to
`output/<name>_scores.csv`.

## Run

```bash
uv run ontology-sql-eval     # or: uv run python -m ontology_sql_eval.judge.main
```

Options: `--input-dir` (default `input`), `--output-dir` (default `output`),
`--workers` (default `1`).

## Output

The judge preserves all original columns and appends:

```
llm_logic_match, llm_semantic_match, llm_final_weighted_score,
llm_sql_vs_ground_truth, llm_is_valid_sql, llm_is_sql_returns_data,
llm_logic_issues
```

The scoring model is configured via `JUDGE_MODEL_NAME` / `JUDGE_BASE_URL` /
`JUDGE_API_KEY`, each falling back to the shared `MODEL_NAME` / `BASE_URL` /
`NVIDIA_API_KEY` when unset.

## BIRD official scoring (EX + VES)

For the BIRD dataset, `bird.py` in this package additionally ports BIRD's own
deterministic Execution Accuracy / Valid Efficiency Score metrics — a separate
step from the generic LLM scoring above. See
[datasets/bird/README.md](../../datasets/bird/README.md#official-scoring-ex--ves).
