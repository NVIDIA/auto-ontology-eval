# FDABench-Lite

[← Back to main README](../../README.md)

[FDABench](https://github.com/fdabench/FDAbench) (KDD'26) evaluates data agents
on analytical queries over heterogeneous data. This repo uses the
**FDABench-Lite** subset as a text-to-SQL worked example: gold SQL is extracted
from each task's `gold_subtasks`, and only the SQLite-backed databases those
tasks need are installed (BIRD train, Spider1, Spider2-lite `local*`).

Of the 282 Lite tasks, **169 become questions across 15 databases**. The rest
are skipped: 76 whose gold SQL is the placeholder `"N/A"`, 32 with no gold SQL
subtask at all, and 5 dabstep tasks (their `merchant_data.db` isn't
redistributed).

## Download

```bash
uv run python scripts/seed_fdabench.py
```

This fetches the HuggingFace Lite JSONLs and the required SQLite archives, then
writes:

```
datasets/fdabench/<db_id>/<db_id>.sqlite
datasets/fdabench/<db_id>/database_description/*.csv   # BIRD DBs only
datasets/fdabench/<db_id>/metadata.json                # from those CSVs
datasets/fdabench/evaluation.json
```

For BIRD-sourced databases, column descriptions from upstream
`database_description/*.csv` are converted into our `metadata.json` shape so
ingest enrichment (`apply_metadata`) can stamp them onto Postgres. Spider1 /
Spider2-lite packs do not ship equivalent structured descriptions.

Re-running is cheap: databases already on disk are skipped (use `--force` to
reinstall), so an interrupted seed can simply be run again.

### Database sources

| Group | Source | Notes |
|---|---|---|
| BIRD train | `train.zip` (Aliyun OSS) | ~9 GB; databases live in a nested `train_databases.zip`, and only the needed ones are extracted |
| Spider2-lite | [`xlangai/spider2-localdb`](https://huggingface.co/datasets/xlangai/spider2-localdb) then Google Drive | The HuggingFace mirror is reliable but an older partial snapshot; Google Drive has the complete pack |
| Spider 1.0 | [`HAL-9001/spider-databases`](https://huggingface.co/datasets/HAL-9001/spider-databases) then Google Drive | HuggingFace mirror covers both databases needed here |

Google Drive frequently answers large downloads with a "Quota exceeded" HTML
page. The seed script detects this, warns, and continues with whatever the
other sources provided; at the end it reports any missing databases and how
many questions they strand. Re-run later, or supply a local copy:

```bash
uv run python scripts/seed_fdabench.py \
  --bird-root /path/to/BIRD_train/train_databases \
  --spider2-root /path/to/spider2-localdb \
  --spider1-root /path/to/spider_data/database
```

Other useful flags: `--tasks-only` (write `evaluation.json` only), `--force`
(re-download and reinstall), `--keep-archive` (keep zips under `.cache/`),
`--no-write-env`, `--dest`.
See [scripts/seed_fdabench.py](../../scripts/seed_fdabench.py).

## Configure

By default, the seed script writes a ready-to-use `CONNECTION_STRINGS` line
(listing every installed database) into your `.env`, replacing any existing
`CONNECTION_STRINGS` entry:

```bash
CONNECTION_STRINGS=sqlite:///<abs>/datasets/fdabench/app_store/app_store.sqlite,sqlite:///<abs>/datasets/fdabench/bank_sales_trading/bank_sales_trading.sqlite,...
```

(all installed databases, comma-separated — one entry per `db_id`). Pass
`--no-write-env` to skip this and just log the value for manual copy-paste
instead.

## Run the full pipeline

```bash
PYTHONPATH=../GSF uv run python main.py --database-name fdabench
```

This ingests every database in `CONNECTION_STRINGS`, compiles the semantic
layer, runs the agent against `datasets/fdabench/evaluation.json`
(→ `input/fdabench_<model>.csv`), then LLM-judges the result
(→ `output/fdabench_<model>_scores.csv`).

## Notes

- Question text is taken from each gold SQL subtask's
  `natural_language_query` (the SQL-focused prompt), not the full multi-source
  FDABench agent query.
- FDABench's MCQ / report rubric scoring is **not** ported here — scoring uses
  this repo's generic exec-match + LLM judge path.
- Per-database `metadata.json` lives under
  `datasets/fdabench/<db_id>/metadata.json`. Ingestion resolves that nested
  layout automatically (same as a top-level `datasets/<db_id>/metadata.json`).
- Source datasets are downloaded at runtime and are **not** vendored in git
  (see [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)).
