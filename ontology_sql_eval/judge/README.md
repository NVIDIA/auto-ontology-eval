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

### Via script

```bash
uv run ontology-sql-eval     # or: uv run python main.py
```

Options: `--input-dir` (default `input`), `--output-dir` (default `output`),
`--workers` (default `1`).

### Via launch.json

Use the **Judge (score CSVs)** configuration in
[.vscode/launch.json](../../.vscode/launch.json); it loads `.env` automatically
(no `PYTHONPATH` needed — the judge is standalone).

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
