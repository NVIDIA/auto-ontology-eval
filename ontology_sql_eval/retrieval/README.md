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

Run ten questions, two at a time:

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.retrieval.eval_chatbot \
    --database-name bird --limit 10 --workers 2
```

## CLI flags

| Flag                  | Default     | Purpose                                                                                |
| --------------------- | ----------- | -------------------------------------------------------------------------------------- |
| `--database-name`     | —           | Derives input `datasets/<name>/evaluation.json` and output `input/<name>_<model>.csv`. |
| `--input PATH`        | derived     | Override the input JSON path.                                                          |
| `--output PATH`       | derived     | Override the output CSV path.                                                          |
| `--workers N`         | `1`         | Run N questions concurrently (see [Parallelism](#parallelism)).                         |
| `--start-index N`     | `0`         | First question index to run. Non-zero **appends** to an existing output CSV.            |
| `--end-index N`       | end         | Stop before this question index.                                                       |
| `--limit N`           | —           | Run at most N questions from `--start-index`. Ignored when `--end-index` is given.      |
| `--log-dir PATH`      | `logs/`     | Root directory for run logs.                                                           |
| `--run-id NAME`       | timestamp   | Name of this run's log directory.                                                      |
| `--verbose`           | off         | Also print DEBUG to the console; the files always have it.                              |
| `--single`            | off         | Run one example query (`SINGLE_QUERY` in the script) and print the result.              |

## Parallelism

`--workers N` runs N questions at once on a thread pool. Questions are
independent, and the pieces they share are safe to use concurrently: the
compiled agent graph carries no cross-invocation state, connectors hand out one
connection per thread, and the retrievers are read-only. The only shared mutable
state in the driver is the CSV writer, which is lock-guarded.

Rows are appended as questions finish, so a killed run keeps everything it had
already scored; the CSV is re-sorted by `row_index` at the end so a parallel run
is diffable against a serial one.

Most of a question's wall time is spent waiting on the model, so throughput
scales with `--workers` until you hit the inference endpoint's rate limit — at
which point `effective_parallelism` in the run summary stops tracking `workers`,
which is the signal to back off.

## Run logs

Every run writes an instrumentation bundle to `logs/<run-id>/`:

| File                 | Contents                                                                            |
| -------------------- | ------------------------------------------------------------------------------------- |
| `run.log`            | Every record from every thread at DEBUG, tagged with worker and question.            |
| `console.log`        | The INFO subset — what the terminal showed.                                          |
| `questions/q<id>.log`| One file per question, holding only that question's records.                          |
| `events.jsonl`       | Phase timeline with monotonic offsets — reconstruct a Gantt view of the workers.      |
| `questions.jsonl`    | Per question: total / agent / scoring seconds, per-node timings, HTTP call counts.    |
| `summary.json`       | Wall clock, throughput, effective parallelism, per-node and per-phase statistics.     |
| `run_config.json`    | Dataset, flags, model/store env vars, and the machine — so timings stay comparable.   |

The bottleneck-hunting fields are:

- `summary.agent_nodes` — total, mean and max seconds per agent graph node
  across the run, sorted by total. The top row is where the run's time went.
- `summary.effective_parallelism` — busy seconds over wall-clock seconds. Well
  below `--workers` means questions are serialising on something.
- `questions.jsonl[].node_timings[].http_calls` — model round-trips attributed
  to the node that made them, which separates "the model is slow" from "we call
  it too many times".
- `summary.http_endpoints` — run-wide call totals split by endpoint
  (`chat_completions` vs `embeddings`), plus `http_calls_per_question`.

Only calls issued on a question's own thread can be attributed to it, so the
per-question and per-node counts are effectively the **chat completion** counts.
`nemo_retriever` dispatches embeddings to its own thread pool, and a pool thread
starts from an empty context; those show up in the run-wide `http_endpoints`
tally and in `http_calls_unattributed`, not against a question.

## Output

The output CSV is written to the repo-root `input/` folder (the judge's input
directory) as `input/<database_name>_<model>.csv`, where `<model>` is the last
segment of `MODEL_NAME`. This means the judge can pick it up directly — no manual
copy needed. It has these columns:

```
row_index, question_id, difficulty, question, expected_sql, returned_sql,
sql_text_similarity, sql_exec_match, expected_sql_error, returned_sql_error,
expected_sql_result, expected_answer_raw, returned_answer,
answer_text_similarity, answer_numbers_match, runtime_seconds,
agent_seconds, scoring_seconds, http_calls, agent_nodes, slowest_node,
slowest_node_seconds, worker, error
```

`agent_seconds` … `worker` are the headline timing columns, so sorting the
results in a spreadsheet already points at the slow rows; the full per-node
breakdown lives in `logs/<run-id>/questions.jsonl`.

These column names are compatible with the judge's input contract (`question`,
`expected_sql`, `returned_sql`, `returned_answer`), so the CSV can be re-scored
directly.
