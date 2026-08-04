# BIRD (Mini-Dev)

[← Back to main README](../../README.md)

[BIRD](https://bird-bench.github.io/) is a large-scale cross-domain Text-to-SQL
benchmark. This repo uses the **Mini-Dev** subset as a worked example: 11
SQLite databases and 500 questions, plus the official EX/VES scoring script
ported from the BIRD repo. Unlike WideWorldImporters, BIRD's source DBs are
SQLite files (no Postgres seeding step) and each question carries its own
`db_id`, routing to the matching connector at eval time.

## Download

```bash
uv run python scripts/seed_bird.py
```

This fetches the official Mini-Dev zip and writes:

```
datasets/bird/<db_id>/<db_id>.sqlite
datasets/bird/<db_id>/database_description/*.csv
datasets/bird/evaluation.json
```

Only the SQLite dialect is kept (the MySQL/PostgreSQL JSONs and `*_gold.sql`
files are ignored). Options: `--force` (re-download and overwrite), `--keep-archive`
(keep the cached zip), `--url` (use a different/local zip), `--dest` (default
`datasets/bird/`), `--log-level`. See [scripts/seed_bird.py](../../scripts/seed_bird.py).

## Configure

By default, the download script writes a ready-to-use `CONNECTION_STRINGS`
line (listing every installed database) directly into your `.env`, replacing
any existing `CONNECTION_STRINGS` entry:

```bash
CONNECTION_STRINGS=sqlite:///<abs>/datasets/bird/california_schools/california_schools.sqlite,sqlite:///<abs>/datasets/bird/card_games/card_games.sqlite,...
```

(all 11 databases, comma-separated — one entry per `db_id`). Pass
`--no-write-env` to skip this and just log the value for manual copy-paste
instead.

## Run the full pipeline

```bash
PYTHONPATH=../GSF uv run python main.py --database-name bird
```

This ingests every database in `CONNECTION_STRINGS`, compiles the semantic
layer, runs the agent against `datasets/bird/evaluation.json`
(→ `input/bird_<model>.csv`), then LLM-judges the result
(→ `output/bird_<model>_scores.csv`).

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
