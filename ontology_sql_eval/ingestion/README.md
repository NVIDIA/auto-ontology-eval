# Ingestion

[← Back to main README](../../README.md)

Populates the Postgres catalog and pgvector stores from a source database so the
text-to-SQL agent has schema and semantic context to retrieve. Set
`CONNECTION_STRINGS` to point at the source DB, then run the pipeline.

`<database_name>` comes from the connector, and metadata and custom analyses are
read from beside that database: `datasets/<benchmark>/dev/<database_name>/` when
`--benchmark-name` names a multi-database benchmark such as BIRD, otherwise
`datasets/<database_name>/`.

> Requires the public sibling `../auto-ontology` checkout on `PYTHONPATH`. See
> [Prerequisites](../../README.md#prerequisites) in the main README.

## Run

```bash
PYTHONPATH=../auto-ontology uv run python -m ontology_sql_eval.ingestion.ingest
PYTHONPATH=../auto-ontology uv run python -m ontology_sql_eval.ingestion.ingest --benchmark-name bird
PYTHONPATH=../auto-ontology uv run python -m ontology_sql_eval.ingestion.semantic --database-name <database_name>
```

## Publishing our descriptions and analyses

BIRD is downloaded with its own column annotations and no analyses at all, and a
plain ingest plus compile reproduces exactly that. Ours are tracked in
`annotations/<dataset>/`, since `datasets/bird/` is gitignored and replaced
wholesale by the next download. Nothing has to be configured: each file is found
by its path and preferred over the dataset's own copy, which is the fallback.

| Artifact | Ours (preferred) | Fallback | Applied by |
| --- | --- | --- | --- |
| analyses | `annotations/<dataset>/custom_analyses/<db>.json` — one per database | `datasets/<dataset>/dev/<db>/custom_analyses.json` | `ingest` |
| descriptions | `annotations/<dataset>/semantic_descriptions.csv` — one export covering every database | `datasets/<dataset>/dev/<db>/semantic_descriptions.csv` | `semantic` |

The set this repo ships for BIRD covers all 11 Dev databases, so the two commands
below are all it takes:

```bash
PYTHONPATH=../auto-ontology uv run python -m ontology_sql_eval.ingestion.ingest --benchmark-name bird
PYTHONPATH=../auto-ontology uv run python -m ontology_sql_eval.ingestion.semantic --benchmark-name bird
```

`--benchmark-name` is what names the `annotations/` subfolder. Omit it and every
benchmark's folder is searched instead, so the file is still found rather than
silently skipped. `--dataset-name` remains available as a compatibility alias.

Analyses need nothing but the schema catalog, so ingest takes them as it always
has. Descriptions wait for the compile: half of them belong to the
`ColumnAttribute` rows that hold the string retrieval embeds, and those do not
exist until it has run. When a saved export exists it is applied automatically.
Re-run the command after editing the export to push the edit through; this is
cheap because the compile only visits tables that have no `Term` yet.

`apply_saved_descriptions()` also takes an explicit path, for scoring one
off-tree export without moving it into `annotations/`.

Descriptions are written to the Postgres catalog *and* to both vector
collections, since a description the catalog holds and the index does not is
invisible to retrieval: columns go through the server's own edit path, which
refreshes the column and its parent table in the data-objects collection, and
each attribute's semantic row is deleted and re-embedded. Attributes are
matched through the column they hang off rather than by name, because the
semantic layer names them itself and a rebuild may name the same column
differently.

Analyses merge by name in the catalog but their vector rows append, so they are
loaded once, by the ingest that builds the database — edit the spec and re-ingest
rather than adding to a store that already has them.

## What `run_ingest()` does

`run_ingest()` performs, in order:

1. Create a connector from `CONNECTION_STRINGS` (the first entry) and derive the
   database name from it.
2. Extract the tabular schema (tables/columns) into the Postgres catalog.
3. `apply_metadata()` — stamp `metadata.json` descriptions and sample values
   onto the catalog rows (skipped if the file is missing).
4. Embed the schema rows via the embedding endpoint and write them to the
   pgvector **data** store.
5. `add_custom_analyses()` — parse the database's analyses (see the table above
   for where they are read from), create `CustomAnalysis` catalog rows, and
   embed them into the pgvector **semantic** store.

Data destinations:

| Destination                      | Content                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| Postgres (`POSTGRES_*`) | Catalog and semantic rows plus schema and custom-analysis embeddings. |

## Quick smoke test (DB-free)

For a quick, DB-free smoke test of the embed pipeline (uses an in-memory
4-table `mock_shop` schema; needs `DEFAULT_MODELS_API_KEY` for embeddings;
legacy `NVIDIA_API_KEY` is supported as a fallback):

```bash
PYTHONPATH=../auto-ontology uv run python -m ontology_sql_eval.ingestion.mock_ingest
```
