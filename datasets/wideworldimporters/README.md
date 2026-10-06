# WideWorldImporters (WWI)

[← Back to main README](../../README.md)

WideWorldImporters is Microsoft's sample OLTP database, ported to Postgres and
seeded here from pre-generated DDL + CSVs. It's the repo's bundled public
worked example — no source (e.g. MSSQL) connection is required, since all
seed artifacts are already generated under this folder.

> **License.** The contents of this directory originate from
> [microsoft/sql-server-samples](https://github.com/microsoft/sql-server-samples)
> (Copyright (c) Microsoft Corporation) and are used under the **MIT License**.
> See [LICENSE](LICENSE) in this directory for the upstream notice, which
> covers the `ddl/` and `data/` files.

## Seed the local Postgres

```bash
uv run python scripts/seed_wwi.py --database-name wideworldimporters --drop
```

This creates the database (using `POSTGRES_*` from `.env`, with
`POSTGRES_DATABASE` as the admin connection), then applies DDL from `ddl/` and
loads CSVs from `data/` in this folder.

Flags (see [scripts/seed_wwi.py](../../scripts/seed_wwi.py)):

- `--database-name <name>` — selects `datasets/<name>/` and names the target DB
  (default `wideworldimporters`).
- `--drop` — drop and recreate the database first (uses `WITH (FORCE)` to
  terminate any lingering sessions).
- `--refresh-collation` — run `ALTER DATABASE ... REFRESH COLLATION VERSION` on
  `template1` and the admin DB before creating. Use this if Postgres reports a
  `collation version mismatch` (common after an OS `libc`/ICU change, e.g. when
  the pgvector container image is updated).
- `--log-level {DEBUG,INFO,WARNING,ERROR}` — logging verbosity (default `INFO`).

## Configure

Point `CONNECTION_STRINGS` at the seeded DB in your `.env`. Replace
`<password>` with your Postgres password — the same one Auto Ontology's `docker-compose`
uses (i.e. the value of `POSTGRES_PASSWORD` in your `.env`):

```bash
CONNECTION_STRINGS=postgresql://postgres:<password>@localhost:5432/wideworldimporters
```

## Run the full pipeline

### Via script

```bash
uv run python main.py --database-name wideworldimporters
```

Individual stages can be skipped with `--skip-ingest`, `--skip-semantic`,
`--skip-eval`, `--skip-judge` (e.g. to re-judge an existing eval CSV:
`--skip-ingest --skip-semantic --skip-eval`).

### Manual (per-stage) equivalent

```bash
# 1. Ingest the source DB schema into the catalog + pgvector, then compile semantics.
uv run python -m ontology_sql_eval.ingestion.ingest
uv run python -m ontology_sql_eval.ingestion.semantic --database-name wideworldimporters

# 2. Run the agent against the eval set -> input/wideworldimporters_<model>.csv
uv run python -m ontology_sql_eval.retrieval.eval_chatbot \
    --database-name wideworldimporters

# 3. Re-score every CSV in input/ with the LLM judge -> output/<name>_scores.csv
uv run ontology-sql-eval
```

See the per-workflow READMEs under [Workflows](../../README.md#workflows) for
the details of each stage.

## Folder layout

```
datasets/wideworldimporters/
  evaluation.json        # eval questions + expected SQL
  custom_analyses.json   # named example analyses (ingestion enrichment)
  ddl/                   # seed DDL: schemas, sequences, tables, indexes, fkeys, views
  data/                  # seed CSV data (COPYed into the tables)
```

There is no `metadata.json` for this dataset (it's optional — enrichment is
skipped when the file is missing).
