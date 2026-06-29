# csv-sql-judge

A Text-to-SQL evaluation toolkit. It bundles three workflows in one project:

1. **sql judge** — a standalone, LLM-powered re-scorer for Text-to-SQL evaluation CSVs.
2. **ingestion** — extract a source DB's schema into the pgvector + Neo4j stores, compile the semantic layer, and enrich the graph with metadata / custom analyses.
3. **retrieval eval** — run the GSF text-to-SQL agent against an evaluation set and score its SQL + answers.

The ingestion and retrieval-eval workflows reuse the `gsf` and `nemo_retriever`
code from the sibling repos rather than re-implementing it.

## Layout

```
csv_sql_judge/        standalone LLM scorer (no GSF dependency)
evaluation/           GSF-backed ingestion + retrieval eval
  enrich_graph.py     graph metadata + custom-analysis enrichment
  local_ingest.py     source DB -> pgvector ingest (calls enrich_graph)
  mock_ingest.py      in-memory 4-table demo ingest (mock_shop)
  eval_chatbot.py     retrieval eval driver
  <database_name>/    evaluation.json, metadata.json, custom_analyses.json
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and **Python 3.12** (3.12–3.13).
- The sibling repos checked out next to this one:
  - `../GSF` (provides the `gsf` package and `dev_tools`)
  - `../NeMo-Retriever` (provides `nemo_retriever`, installed as an editable path dependency)
- For **ingestion** and **retrieval eval** only: live **Neo4j** and **Postgres (pgvector)** services, plus a reachable source DB. The easiest way to start the stores is GSF's `docker-compose.yml`:

```bash
cd ../GSF && docker compose up -d
```

## Setup

```bash
uv sync                # creates the unified .venv (Python 3.12)
cp .env.example .env   # then fill in your values
```

GSF is `package = false`, so `gsf` / `dev_tools` are not installed into the venv
— they import only when `../GSF` is on `PYTHONPATH`. The VS Code launch configs
set this automatically. For command-line runs of the ingestion / eval scripts,
prefix the command with `PYTHONPATH=../GSF` (the judge does not need it).

### Configuration

All settings are read from `.env` (see `.env.example` for the full list):

| Variable                                | Used by                | Description                                   |
| --------------------------------------- | ---------------------- | --------------------------------------------- |
| `NVIDIA_API_KEY`, `BASE_URL`, `MODEL_NAME` | agent (eval)        | LLM that the text-to-SQL agent generates with. |
| `JUDGE_API_KEY`, `JUDGE_BASE_URL`, `JUDGE_MODEL_NAME` | judge   | LLM that csv-sql-judge scores with (falls back to the shared vars above). |
| `EMBED_API_KEY`, `EMBED_ENDPOINT`, `EMBED_MODEL` | ingest + eval | Embedding endpoint (must match ingest/query). |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | ingest + eval  | Graph store connection.                       |
| `POSTGRES_*`                            | ingest + eval          | pgvector store connection.                    |
| `CONNECTION_STRINGS`                    | ingest + eval          | Source DB to extract / execute against.       |

> Note: GSF's own `gsf.semantic` entrypoint loads `../GSF/.env` (not this repo's
> `.env`). Keep the two files in sync — for example, symlink them:
> `ln -sf "$PWD/.env" ../GSF/.env`.

## Usage

### 1. sql judge (standalone)

Drop one or more CSV files into `input/`. Each file must contain `question`,
`expected_sql`, and `returned_sql` columns (and optionally `returned_answer`).
Scored CSVs are written to `output/<name>_scores.csv`.

```bash
uv run csv-sql-judge            # or: uv run python main.py
```

Options: `--input-dir` (default `input`), `--output-dir` (default `output`),
`--workers` (default `1`).

Output columns: `llm_logic_match`, `llm_semantic_match`,
`llm_final_weighted_score`, `llm_sql_vs_ground_truth`, `llm_is_valid_sql`,
`llm_is_sql_returns_data`, `llm_logic_issues`.

### 2. ingestion

Set `CONNECTION_STRINGS` to the source DB, then run the pipeline. Metadata and
custom analyses are read from `evaluation/<database_name>/`.

```bash
PYTHONPATH=../GSF uv run python -m evaluation.local_ingest                       # extract + embed + enrich graph
PYTHONPATH=../GSF uv run python -m gsf.semantic --database-name <database_name>  # compile semantic layer
```

For a quick, DB-free smoke test of the embed pipeline:

```bash
PYTHONPATH=../GSF uv run python -m evaluation.mock_ingest
```

### 3. retrieval eval

Provide `evaluation/<database_name>/evaluation.json` (an array of
`{question_id, question, SQL, answer_raw}` entries), then:

```bash
PYTHONPATH=../GSF uv run python -m evaluation.eval_chatbot --database-name <database_name>
```

Useful flags: `--single` (one hardcoded query), `--consistency --runs N`
(repeat-and-compare). Results are written to
`evaluation/<database_name>/<model>.csv`, which you can then re-score with the
sql judge above.

VS Code launch configurations for all of the above are provided in
`.vscode/launch.json`.
