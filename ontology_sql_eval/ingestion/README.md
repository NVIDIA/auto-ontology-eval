# Ingestion

[← Back to main README](../../README.md)

Populates the Neo4j graph and pgvector stores from a source database so the
text-to-SQL agent has schema and semantic context to retrieve. Set
`CONNECTION_STRINGS` to point at the source DB (metadata and custom analyses are
read from `datasets/<database_name>/`, where `<database_name>` comes from the
connector), then run the pipeline.

> Requires the sibling `../GSF` checkout on `PYTHONPATH`. See
> [Prerequisites](../../README.md#prerequisites) in the main README.

## Run

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.ingest
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.ingest --dataset-name bird
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.semantic --database-name <database_name>
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
After all source DBs finish, if ``--dataset-name <name>`` was passed, embed
that dataset's Train few-shot corpus from ``datasets/<name>/train/train.json``
into the pgvector **semantic** store (incremental; existing questions skipped).
Omit ``--dataset-name`` to skip few-shot enrichment.

Data destinations:

| Destination                      | Content                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| Neo4j (`NEO4J_*`)                | Schema graph (Database → Schema → Table → Column), metadata properties, custom-analysis nodes. |
| Postgres pgvector (`POSTGRES_*`) | Schema embeddings (data store) + custom-analysis and Train Q→SQL embeddings (semantic store). |

## Quick smoke test (DB-free)

For a quick, DB-free smoke test of the embed pipeline (uses an in-memory
4-table `mock_shop` schema; needs `NVIDIA_API_KEY` for embeddings):

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.mock_ingest
```
