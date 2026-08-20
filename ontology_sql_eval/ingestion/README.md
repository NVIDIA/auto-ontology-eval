# Ingestion

[← Back to main README](../../README.md)

Populates the Neo4j graph and pgvector stores from a source database so the
text-to-SQL agent has schema and semantic context to retrieve. Set
`CONNECTION_STRINGS` to point at the source DB, then run the pipeline.

`<database_name>` comes from the connector, and metadata and custom analyses are
read from beside that database: `datasets/<dataset>/dev/<database_name>/` when
`--dataset-name` names a multi-database dataset such as BIRD, otherwise
`datasets/<database_name>/`.

> Requires the sibling `../GSF` checkout on `PYTHONPATH`. See
> [Prerequisites](../../README.md#prerequisites) in the main README.

## Run

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.ingest
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.ingest --dataset-name bird
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.semantic --database-name <database_name>
```

## Publishing our descriptions and analyses

BIRD is downloaded with its own column annotations and no analyses at all, and a
plain ingest plus compile reproduces exactly that. Ours live outside
`datasets/bird/`, which is gitignored and replaced wholesale by the next download,
so two variables in `.env` say where to read them from — first match wins:

| Artifact | Variable | Fallback | Applied by |
| --- | --- | --- | --- |
| analyses | `SAVED_CUSTOM_ANALYSES_DIR` — a directory of `<db>.json` | `datasets/<dataset>/dev/<db>/custom_analyses.json` | `ingest` |
| descriptions | `SAVED_DESCRIPTIONS_CSV` — one export covering every database | `datasets/<dataset>/dev/<db>/semantic_descriptions.csv` | `semantic --override-descriptions` |

The baseline image is already in the right shape for both:

```bash
SAVED_CUSTOM_ANALYSES_DIR=experiments/LOCAL_BL/BL_V2/image_local_baseline_v2/custom_analyses
SAVED_DESCRIPTIONS_CSV=experiments/LOCAL_BL/BL_V2/image_local_baseline_v2/semantic_descriptions.csv
```

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.ingest --dataset-name bird
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.semantic --override-descriptions
```

Analyses need nothing but the schema graph, so ingest takes them as it always has
and the variable only moves the file it reads. Descriptions wait for the compile:
half of them belong to the `ColumnAttribute` nodes that hold the string retrieval
embeds, and those do not exist until it has run. Drop the flag and the compile's own
text stays; re-run the command after editing the export to push the edit through,
which is cheap because the compile only visits tables that have no `Term` yet.

Descriptions are written to the graph *and* to both vector collections, since a
description the graph holds and the index does not is invisible to retrieval:
columns go through the server's own edit path, which refreshes the column and its
parent table in the data-objects collection, and each attribute's semantic row is
deleted and re-embedded. Attributes are matched through the column they hang off
rather than by name, because the semantic layer names them itself and a rebuild may
name the same column differently.

Analyses merge by name in the graph but their vector rows append, so they are
loaded once, by the ingest that builds the database — edit the spec and re-ingest
rather than adding to a store that already has them.

## What `run_ingest()` does

`run_ingest()` performs, in order:

1. Create a connector from `CONNECTION_STRINGS` (the first entry) and derive the
   database name from it.
2. Extract the tabular schema (tables/columns) into the Neo4j graph.
3. `apply_metadata()` — stamp `metadata.json` descriptions and sample values
   onto the graph nodes (skipped if the file is missing).
4. Embed the schema rows via the embedding endpoint and write them to the
   pgvector **data** store.
5. `add_custom_analyses()` — parse the database's analyses (see the table above
   for where they are read from), create `CustomAnalysis` nodes, and embed them
   into the pgvector **semantic** store.
After all source DBs finish, if ``--dataset-name <name>`` was passed, embed
that dataset's Train few-shot corpus from ``datasets/<name>/train/train.json``
into the pgvector **train_qa** store (incremental; existing questions skipped).
Omit ``--dataset-name`` to skip few-shot enrichment.

Data destinations:

| Destination                      | Content                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| Neo4j (`NEO4J_*`)                | Schema graph (Database → Schema → Table → Column), metadata properties, custom-analysis nodes. |
| Postgres pgvector (`POSTGRES_*`) | Schema embeddings (data store) + custom-analysis embeddings (semantic store) + Train Q→SQL embeddings (``train_qa`` store). |

## Quick smoke test (DB-free)

For a quick, DB-free smoke test of the embed pipeline (uses an in-memory
4-table `mock_shop` schema; needs `DEFAULT_MODELS_API_KEY` for embeddings;
legacy `NVIDIA_API_KEY` is supported as a fallback):

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.mock_ingest
```
