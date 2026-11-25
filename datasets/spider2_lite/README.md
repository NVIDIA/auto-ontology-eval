# Spider2-lite (local SQLite)

[← Back to main README](../../README.md)

> **Disclaimer:** Running this code will automatically download data from
> https://github.com/xlang-ai/Spider2,
> https://huggingface.co/datasets/xlangai/spider2-localdb, and the Spider2
> authors' Google Drive
> (https://drive.google.com/file/d/1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG/view).
> Before you run the code, please confirm the content of the dataset and
> licensing is appropriate for your intended use.

[Spider2-lite](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite) has
547 enterprise text-to-SQL questions over BigQuery (`bq*`), Snowflake (`sf*`)
and local SQLite (`local*`) databases.
[scripts/seed/seed_spider2_lite.py](../../scripts/seed/seed_spider2_lite.py)
installs only the **135 local questions over 30 SQLite databases**, the ones
that run without cloud credentials. The 205 BigQuery questions are a separate
benchmark, [`spider2_lite_bq`](../spider2_lite_bq/README.md).

> **Nothing you want to keep belongs in this folder.** Everything under
> `datasets/spider2_lite/` except this README and `.gitkeep` is gitignored and
> replaced by the next seed. Our own analyses and descriptions belong in
> `inputs/annotations/spider2_lite/`, as for BIRD.

## Download

```bash
uv run python scripts/seed/seed_spider2_lite.py
```

This clones (or updates) a sparse checkout of the upstream repository under
`third_party/Spider2/` for the questions, gold artifacts and per-table metadata,
then fetches the SQLite pack: the HuggingFace mirror first (reliable, but an
older partial snapshot), then Google Drive for whatever it lacks. Google Drive
often answers large downloads with a "Quota exceeded" page; the seeder detects
it, keeps what the mirror provided, and reports the databases and questions
still missing. Re-run later, or supply a local copy:

```bash
uv run python scripts/seed/seed_spider2_lite.py --sqlite-root /path/to/spider2-localdb
```

Databases already installed are skipped, so an interrupted seed can simply be
run again. Other options: `--upstream-root` (use an existing Spider2 checkout
as-is, without git), `--ref` (upstream branch, tag or commit; default `main`),
`--skip-databases` (questions, gold results and metadata only), `--force`
(re-download and overwrite), `--keep-archive`, `--no-write-env`, `--dest`,
`--log-level`.

## Folder layout

```
datasets/spider2_lite/
  evaluation.json                   # the 135 local* questions
  <db_id>/<db_id>.sqlite            # one database per db_id
  <db_id>/metadata.json             # sample values per column (ingestion enrichment)
  gold/<instance_id>[_<x>].csv      # gold execution results
  manifest.json                     # upstream commit, per-database counts and install status
```

`db_id` is the upstream `db` name slugified (`Db-IMDB` → `db_imdb`, `IPL` →
`ipl`). It is also the SQLite file stem, so it is the database name ingestion
reports. `question_id` is the upstream `instance_id` (`local001`).

Upstream leaves every local column description empty, so `metadata.json`
carries only up to five distinct sample values per column, taken from the
upstream `sample_rows`.

## Gold answers

Spider2 scores a prediction against gold **result** CSVs, not gold SQL.
Upstream publishes gold SQL for only 24 of the 135 local questions, so `SQL` is
empty for the rest. Every question has gold results, and each
`evaluation.json` row carries what upstream's
[`evaluate.py`](https://github.com/xlang-ai/Spider2/blob/main/spider2-lite/evaluation_suite/evaluate.py)
needs to score it:

| Field            | Meaning                                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| `gold_results`   | Gold CSV paths relative to this folder. When there are several, matching any one of them scores.          |
| `condition_cols` | Gold column indexes that must appear in the prediction. Either one list for every CSV, or one per CSV.    |
| `ignore_order`   | Whether row order is ignored.                                                                             |

## Configure

By default the seed script writes a `CONNECTION_STRINGS` line for every
installed database into your `.env`, replacing any existing entry. Pass
`--no-write-env` to only log it. Then ingest:

```bash
uv run python -m ontology_sql_eval.ingestion.ingest --benchmark-name spider2_lite
uv run python -m ontology_sql_eval.ingestion.semantic --benchmark-name spider2_lite
```

## Run the benchmark

Both NeMo Gym arms, then the comparison:

```bash
uv run python main.py --database-name spider2_lite --skip-ingest --skip-semantic
```

The verifier scores the way `evaluate.py` does, against the gold CSVs above;
see [resources_servers/README.md](../../resources_servers/README.md) for the
details and the quirks it keeps.
