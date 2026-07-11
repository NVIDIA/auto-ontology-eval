# STaRK-Amazon

This dataset folder contains the reproducible schema and enrichment files for
ingesting the [STaRK-Amazon](https://stark.stanford.edu/dataset_amazon.html)
semi-structured knowledge base as a local Postgres database.

The seed CSVs under `datasets/amazon/data/` are not committed because the full
export is about 4.8 GB, including a multi-GB reviews file that exceeds GitHub's
regular file-size limit. Regenerate them locally with the converter script.

## Generate CSVs

`stark-qa` requires Python 3.8-3.11, while this project runs on Python 3.12+.
Run the converter in an isolated Python 3.11 environment:

```bash
uv run --no-project --python 3.11 --with stark-qa python scripts/convert_stark_amazon.py
```

This downloads the processed STaRK-Amazon SKB from Hugging Face and writes:

```text
datasets/amazon/data/amazon.<table>.csv
```

The full export is large and may take several minutes. To regenerate only the
schema/edge/entity CSVs while reusing existing review and Q&A CSVs:

```bash
uv run --no-project --python 3.11 --with stark-qa python scripts/convert_stark_amazon.py --skip-reviews --skip-qa
```

## Seed Postgres

After generating the CSVs, seed a local Postgres database named `amazon`:

```bash
uv run python scripts/seed_postgres.py --database-name amazon --drop
```

Then point `.env` at the seeded database:

```bash
CONNECTION_STRINGS=postgresql://postgres:<password>@localhost:5432/amazon
```

## Ingest And Compile

Start the GSF Neo4j/Postgres services if they are not already running:

```bash
cd ../GSF && docker compose up -d
```

Run schema ingestion and semantic-layer compilation:

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.ingest
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.semantic --database-name amazon
```

This prepares the schema graph and semantic layer only. STaRK QA evaluation
requires a separate conversion from STaRK gold product IDs to an evaluation
format/scorer and is not part of this dataset folder yet.
