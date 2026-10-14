# SQL judge (standalone)

[← Back to main README](../../README.md)

Re-scores evaluation CSVs with an LLM that rates each row's SQL on logic,
semantics, and similarity to the ground truth. This workflow is fully
self-contained — no Auto Ontology/NeMo install, database, or vector stores required, only
a judge LLM API key.

The judge scores every CSV you put in `input/`. Each file must contain `question`, `expected_sql`, and `returned_sql` columns (and
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
`JUDGE_API_KEY`. Each falls back to `DEFAULT_MODELS_MODEL` /
`DEFAULT_MODELS_ENDPOINT` / `DEFAULT_MODELS_API_KEY`, then the legacy shared
`MODEL_NAME` / `BASE_URL` / `NVIDIA_API_KEY`. When those are also unset, a
built-in endpoint/model is chosen from the API key prefix (`sk-` →
inference-api.nvidia.com, `nvapi-` → integrate.api.nvidia.com), matching Auto Ontology's
`model_config` behavior.
