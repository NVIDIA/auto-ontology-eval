# Retrieval eval

[← Back to main README](../../README.md)

Runs the text-to-SQL agent against `datasets/<database_name>/evaluation.json`
and scores each question deterministically: the expected and returned SQL are
both executed against the live source DB and their result sets compared, and the
returned answer is compared against `answer_raw`.

> Requires the sibling `../GSF` checkout on `PYTHONPATH`. See
> [Prerequisites](../../README.md#prerequisites) in the main README.

## Run

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.retrieval.eval_chatbot --database-name <database_name>
```

## CLI flags

| Flag                | Default    | Purpose                                                                                   |
| ------------------- | ---------- | ----------------------------------------------------------------------------------------- |
| `--database-name`   | —          | Derives input `datasets/<name>/evaluation.json` and output `input/<name>_<model>.csv`.    |
| `--input PATH`      | derived    | Override the input JSON path.                                                             |
| `--output PATH`     | derived    | Override the output CSV path.                                                             |
| `--single`          | off        | Run one example query (`SINGLE_QUERY` in the script) and print the result.                |
| `--start-index N`   | 0          | Index (0-based) of the first question to run. When > 0, appends to an existing `--output` CSV instead of overwriting it — use to resume a chunked run. |
| `--end-index N`     | run to end | Index (0-based, exclusive) of the last question to run.                                   |

`--start-index`/`--end-index` are mainly useful for chunking/resuming large
evaluation sets (e.g. the full BIRD Dev/Train splits — see
[datasets/bird/README.md](../../datasets/bird/README.md)) that are impractical
to run start-to-finish in one unattended shot.

## Output

The output CSV is written to the repo-root `input/` folder (the judge's input
directory) as `input/<database_name>_<model>.csv`, where `<model>` is the last
segment of `MODEL_NAME`. This means the judge can pick it up directly — no manual
copy needed. It has these columns:

```
row_index, question_id, difficulty, question, expected_sql, returned_sql,
sql_text_similarity, sql_exec_match, expected_sql_error, returned_sql_error,
expected_sql_result, expected_answer_raw, returned_answer,
answer_text_similarity, answer_numbers_match, runtime_seconds, error
```

These column names are compatible with the judge's input contract (`question`,
`expected_sql`, `returned_sql`, `returned_answer`), so the CSV can be re-scored
directly.
