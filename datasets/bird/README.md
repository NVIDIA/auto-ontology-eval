# BIRD

[← Back to main README](../../README.md)

[BIRD](https://bird-bench.github.io/) is a large-scale cross-domain Text-to-SQL
benchmark. This repo defaults to the **Mini-Dev** subset as a cheap worked
example (11 SQLite databases, 500 questions), but the seed script can also
pull the full official **Dev** and **Train** splits, plus the official
EX/VES scoring script ported from the BIRD repo. Unlike WideWorldImporters,
BIRD's source DBs are SQLite files (no Postgres seeding step) and each
question carries its own `db_id`, routing to the matching connector at eval
time.

## Download

```bash
uv run python scripts/seed_bird.py                    # Mini-Dev (default): 500 Qs, 11 DBs
uv run python scripts/seed_bird.py --splits dev        # full Dev: 1,534 Qs, same 11 DBs
uv run python scripts/seed_bird.py --splits dev train  # full Dev + Train: ~11k Qs, ~80 DBs
```

| Split       | Questions | Databases                | Notes                                                             |
| ----------- | --------: | ------------------------ | ------------------------------------------------------------------ |
| `mini-dev`  |       500 | 11                        | Default. Cheap/fast for day-to-day development.                    |
| `dev`       |     1,534 | 11 (same as `mini-dev`)   | The full official Dev split used for the BIRD leaderboard.         |
| `train`     |     9,428 | ~69 additional            | Fine-tuning split — rows have **no `difficulty` label**.           |

Pass more than one `--splits` value to combine them into one shared
`datasets/bird/` (e.g. `--splits dev train`). This writes:

```
datasets/bird/<db_id>/<db_id>.sqlite
datasets/bird/<db_id>/database_description/*.csv
datasets/bird/evaluation.json
```

Only the SQLite dialect is kept (the MySQL/PostgreSQL JSONs and `*_gold.sql`
files are ignored). Options: `--splits` (one or more of `mini-dev`/`dev`/`train`,
default `mini-dev`), `--force` (re-download and overwrite), `--keep-archive`
(keep the cached zip), `--url` (use a different/local zip — only valid with a
single `--splits` value), `--dest` (default `datasets/bird/`), `--log-level`.
See [scripts/seed_bird.py](../../scripts/seed_bird.py).

> **Combining splits is a much bigger undertaking than Mini-Dev.** `dev` alone
> is ~3x the questions of `mini-dev` over the same 11 DBs; adding `train`
> brings in ~69 more (large) databases and ~9,400 more questions. Running the
> agent + LLM judge (and especially VES timing) over that many questions
> takes proportionally longer and costs proportionally more in LLM calls —
> start with `--splits dev` alone before adding `train`. When combining
> splits, each split's `question_id` is offset internally (by 100,000 per
> split) so ids from different splits never collide in the merged
> `evaluation.json`.

## Configure

By default, the download script writes a ready-to-use `CONNECTION_STRINGS`
line (listing every installed database) directly into your `.env`, replacing
any existing `CONNECTION_STRINGS` entry:

```bash
CONNECTION_STRINGS=sqlite:///<abs>/datasets/bird/california_schools/california_schools.sqlite,sqlite:///<abs>/datasets/bird/card_games/card_games.sqlite,...
```

(every installed database, comma-separated — one entry per `db_id`; 11 for
`mini-dev`/`dev`, more if `train` is included). Pass `--no-write-env` to skip
this and just log the value for manual copy-paste instead.

## Run the full pipeline

```bash
PYTHONPATH=../GSF uv run python main.py --database-name bird
```

This ingests every database in `CONNECTION_STRINGS`, compiles the semantic
layer, runs the agent against `datasets/bird/evaluation.json`
(→ `input/bird_<model>.csv`), then LLM-judges the result
(→ `output/bird_<model>_scores.csv`).

For the full `dev`/`train` splits, a single unattended run over thousands of
questions is riskier to babysit — the retrieval eval's
[`ontology_sql_eval.retrieval.eval_chatbot`](../../ontology_sql_eval/retrieval/README.md)
CLI supports `--start-index`/`--end-index` to chunk the run and resume (by
appending to the existing output CSV) after an interruption, e.g.:

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.retrieval.eval_chatbot \
  --database-name bird --start-index 0 --end-index 500
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.retrieval.eval_chatbot \
  --database-name bird --start-index 500   # resumes, appends to the same CSV
```

## Official scoring (EX + VES)

In addition to the generic LLM judge from `main.py`, BIRD ships its own
deterministic **Execution Accuracy (EX)** and **Valid Efficiency Score (VES)**
metrics, ported from the official BIRD evaluation scripts:

```bash
uv run python -m ontology_sql_eval.judge.bird \
  --input input/bird_<model>.csv \
  --skip-ves   # omit to also compute VES (slower — runs timing on EX-passing rows)
```

This writes `output/<name>_bird_scores.csv`. See
[ontology_sql_eval/judge/bird.py](../../ontology_sql_eval/judge/bird.py) for the
full flag reference (`--evaluation-json`, `--db-root`, `--num-cpus`,
`--meta-time-out`, `--iterate-num`, `--no-output-csv`).

This is a separate, additional scoring step — it does not replace the LLM
judge stage in `main.py`.

> **Note on `train` rows:** the Train split has no official `difficulty`
> label (it's a fine-tuning split, not part of the BIRD leaderboard), so
> `train` rows don't land in the `simple`/`moderate`/`challenging` columns of
> the EX/VES tables — they're still scored correctly and counted in the
> `total` column, just without a difficulty breakdown.
