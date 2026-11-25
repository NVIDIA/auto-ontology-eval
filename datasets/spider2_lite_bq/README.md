# Spider2-lite (BigQuery)

[← Back to main README](../../README.md)

> **Disclaimer:** Running this code will automatically download data from
> https://github.com/xlang-ai/Spider2 and query Google BigQuery public datasets
> (`bigquery-public-data` and a few others). Before you run the code, please
> confirm the content of the dataset and licensing is appropriate for your
> intended use. **Queries are billed to your own GCP project.**

The 205 `bq*`/`ga*` questions of
[Spider2-lite](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite),
over 70 databases made of BigQuery public datasets. The 135 local SQLite
questions, which need no credentials, are in
[`../spider2_lite/`](../spider2_lite/README.md).

> **Nothing you want to keep belongs in this folder.** Everything under
> `datasets/spider2_lite_bq/` except this README and `.gitkeep` is gitignored
> and replaced by the next seed.

## Prerequisites

1. A GCP project you can run BigQuery jobs in (role **BigQuery Job User**,
   `roles/bigquery.jobUser`) with billing enabled. The public datasets
   themselves need no grant.
2. Application Default Credentials on the host, never in the repository:

   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project <PROJECT_ID>
   ```

   For CI, a service-account key via `GOOGLE_APPLICATION_CREDENTIALS` works
   too.

The Storage Read API is not used, so Job User is enough.

## Download

```bash
uv run python scripts/seed/seed_spider2_lite_bq.py
```

This adds `spider2-lite/resource/databases/bigquery` (about 450 MB of per-table
metadata) to the sparse checkout under `third_party/Spider2/`. It then writes
the questions, gold results, `metadata.json` and a schema dump per database,
and a `CONNECTION_STRINGS` line into `.env`. That line **replaces** any
existing entry, such as the SQLite one for `spider2_lite`. Pass
`--no-write-env` to only log it.

Options:

| Option | Meaning |
| --- | --- |
| `--billing-project` | Project that pays. Default: `GOOGLE_CLOUD_PROJECT`, else the ADC file's `quota_project_id` |
| `--maximum-bytes-billed` | Per-query cap written into every URL, e.g. `50GiB`. Default `BIGQUERY_MAXIMUM_BYTES_BILLED`, else 40 GiB |
| `--db ga4 --db ga360` | Seed only these databases (a cheap pilot) |
| `--schema-char-budget` | Size of a schema-only dump. Default 200,000 characters |
| `--upstream-root`, `--ref`, `--dest`, `--log-level` | As for `seed_spider2_lite.py` |

## Folder layout

```
datasets/spider2_lite_bq/
  evaluation.json              # the 205 bq*/ga* questions
  <db_id>/schema.sql           # the schema-only arm's schema dump
  <db_id>/metadata.json        # descriptions and sample values (ingestion enrichment)
  gold/<instance_id>[_<x>].csv # gold execution results
  manifest.json                # upstream commit, per-database counts and dump sizes
```

`evaluation.json` rows carry the same fields as `spider2_lite`'s, plus
`bq_datasets`: the `project.dataset` names that make up the database.

## How BigQuery maps onto the benchmark

- **Connection strings** look like
  `bigquery://<billing_project>/<db_id>?dataset=<project>.<dataset>[&dataset=...][&maximum_bytes_billed=50GiB][&location=US]`.
  They accept the same optional `schema`, `credentials` and `credentials_file`
  parameters as Auto Ontology's connector; seeded benchmark URLs use ADC.
  Auto Ontology's catalog is the project, its schema is the dataset.
- **Date-sharded tables** (`ga_sessions_20170801`, `events_20210131`, `gsod1929`,
  ...) are collapsed into one wildcard table, `ga_sessions_*`. It has a
  `_TABLE_SUFFIX` pseudo-column, and the shard range is in its description.
  This takes GA360 from 366 tables to one, and GA4 from 92 to one. A wildcard
  query reads every shard with the newest shard's schema, so a shard with a
  column the newest lacks (or has with another type) stays a table of its own
  and is named in the wildcard's description: `tree_census_1995`, the
  `tlc_*_trips_*` years that have `airport_fee`, and some `cms_medicare`,
  `irs_990` and `nhtsa_traffic_fatalities` years.
- **Schema dumps are cut.** Some databases are enormous: gnomAD's full DDL is
  about 5.4M characters. Tables whose columns repeat an earlier table's are
  listed under it. Past the budget, tables are named without columns. 13
  databases are cut this way (see `manifest.json`, `tables_listed_by_name`). The
  Auto Ontology arm is unaffected: it retrieves what it needs.
- **Column descriptions** come from upstream, including those of nested
  `STRUCT` fields (`event_params.key`).
- **Scoring** is described in
  [resources_servers/README.md](../../resources_servers/README.md): the
  empty-result rule, the dry-run read-only check, the byte cap, and project
  qualification.

## Cost

Dry runs are free. Ingestion goes through Auto Ontology's BigQuery connector,
which reads metadata with `INFORMATION_SCHEMA` queries (at least 10 MB billed
each, several per dataset) and profiles tables with ordinary queries; on a
wildcard table those read only the newest shard. Each query is held to the
URL's byte cap. A full run of every gold query would scan about 0.44 TiB, roughly $3 at
on-demand pricing. Models' exploratory queries add to this. The per-query cap
bounds any single mistake. Queries billing over 1 GiB are logged.

## Run

```bash
uv run python -m ontology_sql_eval.ingestion.ingest --benchmark-name spider2_lite_bq
uv run python -m ontology_sql_eval.ingestion.semantic --benchmark-name spider2_lite_bq
GYM_REBUILD_VENV=1 uv run python main.py --database-name spider2_lite_bq --skip-ingest --skip-semantic
```

`GYM_REBUILD_VENV=1` is needed once: the schema-only server's venv gains
`google-cloud-bigquery`, `db-dtypes`, `pyarrow` and `sqlglot`.

## Known gaps

- `bq109` (open_targets_genetics_1) reads `open-targets-genetics`, which
  currently answers 403 Access Denied. Both arms miss it alike.
- Upstream's gold SQL does not always reproduce its own gold CSVs; for example,
  `bq002`'s weekly maximum differs. Scoring is against the CSVs, as upstream's.
- The Auto Ontology agent's catalog check reads a single shard
  (`events_20210131`) or a partial wildcard (`events_2021*`) as the wildcard
  table, so such queries run as written.
