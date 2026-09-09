# Ingestion

[← Back to main README](../../README.md)

Populates the Neo4j graph and pgvector stores from a source database so the
text-to-SQL agent has schema and semantic context to retrieve. Set
`CONNECTION_STRINGS` to point at the source DB (metadata and custom analyses are
read from `datasets/<database_name>/`, where `<database_name>` comes from the
connector), then run the pipeline.

> Requires the sibling `../GSF` checkout on `PYTHONPATH`. See
> [Prerequisites](../../README.md#prerequisites) in the main README.

## Provisioning a named dataset's catalog

For a benchmark dataset (BIRD-Interact Lite/Full, Spider2-lite, ...),
`scripts/provision_catalog.sh <dataset-name>` runs seed → ingest → semantic
compile in one step, targeting that dataset's own isolated catalog database
(see `configs/datasets/*.env`) rather than one shared catalog — otherwise
GSF's semantic layer (the `term` table and friends) would silently
cross-link business terms compiled from unrelated datasets, since it shares
those globally *within* one Postgres database.

```bash
scripts/provision_catalog.sh --list        # see what's configured
scripts/provision_catalog.sh bird_lite
scripts/provision_catalog.sh bird_full
scripts/provision_catalog.sh spider2_lite
```

Each dataset's `POSTGRES_DATABASE` (see its `configs/datasets/<name>.env`)
must already exist and have GSF's schema migrated onto it before running
this — the script does not create or migrate the database itself. Create it
once with `createdb <name>` (or `psql -c "CREATE DATABASE <name>"`) against
the same Postgres server `../GSF`'s `docker-compose.yml` uses, then run that
compose file's `gsf-migrate` one-shot job against it
(`POSTGRES_DATABASE=<name> docker compose run --rm gsf-migrate`).

Adding a new benchmark is a config change, not a code change: add
`configs/datasets/<name>.env` (see the existing files for the format —
`POSTGRES_DATABASE`, `CONNECTION_STRINGS` or `CONNECTION_STRINGS_CMD`, and an
optional `SEED_CMD`); no new script.

## Run (manual, single dataset already exported)

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.ingest                                    # extract + embed + enrich graph
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.semantic --database-name <database_name>  # compile semantic layer
```

## What `run_ingest()` does

`run_ingest()` performs, in order:

1. Create a connector from `CONNECTION_STRINGS` (the first entry) and derive the
   database name from it.
2. Extract the tabular schema (tables/columns) into the Neo4j graph.
3. `apply_metadata()` — stamp `metadata.json` descriptions and sample values
   onto the graph nodes (skipped if the file is missing).
4. Embed the schema rows via the embedding endpoint and write them to the
   pgvector **data** store.
5. `add_custom_analyses()` — parse `custom_analyses.json`, create
   `CustomAnalysis` nodes, and embed them into the pgvector **semantic** store.

Data destinations:

| Destination                      | Content                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| Neo4j (`NEO4J_*`)                | Schema graph (Database → Schema → Table → Column), metadata properties, custom-analysis nodes. |
| Postgres pgvector (`POSTGRES_*`) | Schema embeddings (data store) + custom-analysis embeddings (semantic store).                  |

## Quick smoke test (DB-free)

For a quick, DB-free smoke test of the embed pipeline (uses an in-memory
4-table `mock_shop` schema; needs `DEFAULT_MODELS_API_KEY` for embeddings;
legacy `NVIDIA_API_KEY` is supported as a fallback):

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.mock_ingest
```
