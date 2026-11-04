# Extra prompts: SQL exemplars and value anchors

Two optional CSVs add question-specific context to the treatment arm's
SQL-generation prompt. Both are built from BIRD Dev and keyed by `question_id`.

| File                                                           | `main.py` flag        | What it carries                                                              |
| -------------------------------------------------------------- | --------------------- | ---------------------------------------------------------------------------- |
| `inputs/extra_prompts/bird/predicted_structural_exemplars.csv` | `--sql-examples-csv`  | Training questions whose SQL has the shape this question needs, ranked       |
| `inputs/extra_prompts/bird/value_anchors.csv`                  | `--value-anchors-csv` | The column that stores a phrase from the question, with its stored spelling  |

Both are off unless you pass their flags, and the control arm has no
equivalent. A difference measured with them covers these inputs as well as the
ontology, so say which inputs a reported number used.

## Using them

The repository includes generated copies of both files. Pass either or both:

```bash
uv run python main.py --database-name bird --skip-ingest --skip-semantic \
  --sql-examples-csv inputs/extra_prompts/bird/predicted_structural_exemplars.csv \
  --sql-examples-k 5 \
  --value-anchors-csv inputs/extra_prompts/bird/value_anchors.csv
```

`--sql-examples-k` sets the exemplars per question (default 5).
`--sql-examples-min-score` withholds exemplars from questions whose best match
scores below the given value.

## Rebuilding them

[`scripts/extra_prompts/build_eval_inputs.py`](../scripts/extra_prompts/build_eval_inputs.py)
runs every step and writes both files:

```bash
uv run python scripts/extra_prompts/build_eval_inputs.py
```

It needs the BIRD Dev split and its training corpus on disk
(`uv run python scripts/seed/seed_bird.py --splits dev`) and the `REASONING_*`
model settings.

- **Exemplars** are the slow, paid step: the model predicts a query for each Dev
  question, and the script matches it against training questions with the same
  SQL shape. Model responses are cached next to the CSV; a rebuild archives the
  old cache under a timestamped name, so answers are resampled each time.
- **Anchors** need no model. The script indexes values across the Dev SQLite
  databases, rebuilding the index on each run so anchors match the current data.

| Option                             | Default                      | Purpose                                                       |
| ---------------------------------- | ---------------------------- | ------------------------------------------------------------- |
| `--only anchors`                   | both                         | Rebuild only the anchors, with no model calls                 |
| `--only exemplars`                 | both                         | Rebuild only the exemplars                                    |
| `--k N`                            | `5`                          | Exemplars kept per question                                   |
| `--alternatives N`                 | `3`                          | Distinct readings predicted per question; `0` predicts one    |
| `--no-schema`                      | off                          | Predict without showing the question's schema                 |
| `--workers N`                      | `8`                          | Concurrent model requests in the exemplar step                |
| `--limit N`                        | all                          | First N Dev questions, for a smoke test                       |
| `--dry-run`                        | off                          | Print the commands without running them                       |
| `--anchors-out`, `--exemplars-out` | `inputs/extra_prompts/bird/` | Output locations                                              |
