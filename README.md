# Ontology SQL Eeval

A Text-to-SQL evaluation toolkit for benchmarking a text-to-SQL agent against a
dataset and scoring the results. It bundles three workflows in one project:

1. **ingestion** — extract a source database's schema into the pgvector + Neo4j
   stores, compile the semantic layer, and enrich the graph with metadata and
   custom analyses.
2. **retrieval eval** — run the text-to-SQL agent against an evaluation set and
   score its generated SQL and answers deterministically.
3. **sql judge** — a standalone, LLM-powered re-scorer for Text-to-SQL
   evaluation CSVs (works on its own, no database or GSF/NeMo install required).

The typical lifecycle is **ingest → eval → judge**: you ingest a database so the
agent can retrieve its schema, run an evaluation to produce a results CSV, then
optionally re-score that CSV with the LLM judge.

> [!IMPORTANT]
> **You need [GSF](https://github.com/NVIDIA/GSF).** The ingestion and
> retrieval-eval workflows import the `gsf` package from a **sibling `../GSF`
> checkout** ([NVIDIA/GSF](https://github.com/NVIDIA/GSF))
>
> It must live right next to this
> repo on the same machine (i.e. `../GSF` relative to this repo's root) and be on
> `PYTHONPATH`.
>
> It is **not** installed into the venv. Only the **judge** works
> without it.
>
> See [Prerequisites](#prerequisites) below.

## Prerequisites

| Requirement                                                                  | Ingestion  | Retrieval eval | Judge |
| ---------------------------------------------------------------------------- | :--------: | :------------: | :---: |
| [uv](https://docs.astral.sh/uv/) + Python 3.12 (3.12–3.13)                   |    yes     |      yes       |  yes  |
| Sibling repo `../GSF` checkout ([NVIDIA/GSF](https://github.com/NVIDIA/GSF)) |    yes     |      yes       |  no   |
| GitHub access to `NVIDIA/NeMo-Retriever`                                     |    yes     |      yes       |  no   |
| Neo4j + Postgres (pgvector) services                                         |    yes     |      yes       |  no   |
| Reachable source database                                                    |    yes     |      yes       |  no   |
| LLM API key                                                                  | embed only |      yes       |  yes  |

The two external dependencies are provided differently:

- `../GSF` (`https://github.com/NVIDIA/GSF`) — provides the `gsf` package. **Check it out next to this repo** so it
  resolves as `../GSF`; it is imported via `PYTHONPATH` (not installed — see
  below).
- `nemo-retriever` (`https://github.com/NVIDIA/NeMo-Retriever.git`) — provides
  `nemo_retriever`, installed from GitHub by `uv sync` (see `[tool.uv.sources]`
  in [pyproject.toml](pyproject.toml)).

For ingestion and retrieval eval you also need live **Neo4j** and **Postgres
(pgvector)** services, plus a reachable source DB. The easiest way to start the
stores is GSF's `docker-compose.yml`:

```bash
cd ../GSF && docker compose up -d
```

## Setup

```bash
uv sync                # creates the unified .venv (Python 3.12), pulling nemo_retriever from GitHub
cp .env.example .env   # then fill in your values
```

GSF is `package = false`, so `gsf` is not installed into the venv — it imports
only when `../GSF` is on `PYTHONPATH`. The VS Code launch configs set this
automatically. For command-line runs of the ingestion / eval scripts, prefix the
command with `PYTHONPATH=../GSF` (the judge does not need it).

## Configuration

> This project was tested with LLMs served from
> [https://integrate.api.nvidia.com](https://integrate.api.nvidia.com) (judge,
> embeddings) and [https://inference-api.nvidia.com](https://inference-api.nvidia.com)
> (agent); see [.env.example](.env.example) for the exact models used for each
> role.
>
> **Use only models that support structured output.**

All settings are read from `.env` (see [.env.example](.env.example) for the full
list). Variables are grouped by the workflow that uses them:

| Variable                                              | Used by       | Description                                                           |
| ----------------------------------------------------- | ------------- | --------------------------------------------------------------------- |
| `NVIDIA_API_KEY`, `BASE_URL`, `MODEL_NAME`            | agent (eval)  | LLM that the text-to-SQL agent generates with.                        |
| `JUDGE_API_KEY`, `JUDGE_BASE_URL`, `JUDGE_MODEL_NAME` | judge         | LLM that the judge scores with (falls back to the shared vars above). |
| `EMBED_API_KEY`, `EMBED_ENDPOINT`, `EMBED_MODEL`      | ingest + eval | Embedding endpoint (must be identical for ingest and query).          |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`       | ingest + eval | Graph store connection.                                               |
| `POSTGRES_*`                                          | ingest + eval | pgvector store connection.                                            |
| `CONNECTION_STRINGS`                                  | ingest + eval | Source DB to extract schema from / execute SQL against.               |

## Seed the local source DB (required)

> **Required to run this project.** Ingestion and retrieval eval need a reachable
> source database. Only the standalone judge can run without a source DB.

Each bundled dataset's README documents how to stand up its source DB and set
`CONNECTION_STRINGS` (see [Configuration](#configuration)):

- **[WideWorldImporters](datasets/wideworldimporters/README.md)** — seed a
  local Postgres from pre-generated DDL + CSVs.
- **[BIRD Mini-Dev](datasets/bird/README.md)** — download per-database SQLite
  files.

## Full pipeline (end-to-end example)

`main.py` runs the entire lifecycle for a dataset in one shot — ingest →
semantic compile → eval → judge — writing
`output/<database_name>_<model>_scores.csv` (assumes the stores are up and
`.env` is filled in):

```bash
PYTHONPATH=../GSF uv run python main.py --database-name <database_name>
```

Individual stages can be skipped with `--skip-ingest`, `--skip-semantic`,
`--skip-eval`, `--skip-judge` (e.g. to re-judge an existing eval CSV:
`--skip-ingest --skip-semantic --skip-eval`). Or use the **Run full pipeline**
configuration in [.vscode/launch.json](.vscode/launch.json), which sets
`PYTHONPATH=../GSF` and loads `.env` automatically.

For concrete, copy-pasteable walkthroughs (including the manual per-stage
commands), see the per-dataset READMEs:
[WideWorldImporters](datasets/wideworldimporters/README.md),
[BIRD Mini-Dev](datasets/bird/README.md). See the per-workflow READMEs under
[Workflows](#workflows) for the details of each stage.

## Datasets

Each dataset lives in its own folder under `datasets/<database_name>/`. The
folder name is the database name and is used to derive default file paths.

```
datasets/<database_name>/
  evaluation.json        # eval questions + expected SQL (retrieval-eval input)
  metadata.json          # table/column descriptions (ingestion enrichment; optional)
  custom_analyses.json   # named example analyses (ingestion enrichment; optional)
```

The retrieval eval writes its results CSV to the repo-root `input/` folder
(`input/<database_name>_<model>.csv`), not into the dataset folder, so the judge
can score it directly.

The repo ships with two public worked examples:

### BIRD (Mini-Dev)

The [BIRD](https://bird-bench.github.io/) Mini-Dev subset — 11 SQLite
databases, 500 questions, plus the official EX/VES scoring script. See
**[datasets/bird/README.md](datasets/bird/README.md)** for how to download
the dataset and run the full pipeline.

### WideWorldImporters (WWI)

Microsoft's WideWorldImporters sample OLTP database, ported to Postgres and
seeded from pre-generated DDL + CSVs. See
**[datasets/wideworldimporters/README.md](datasets/wideworldimporters/README.md)**
for how to seed the database and run the full pipeline.

The schemas below (`evaluation.json`, `metadata.json`, `custom_analyses.json`)
are generic across datasets; the examples use `wideworldimporters` throughout.

### `evaluation.json`

A JSON **array** of question objects. Fields consumed by the retrieval eval:

| Field         | Required | Purpose                                                                |
| ------------- | :------: | ---------------------------------------------------------------------- |
| `question_id` |    no    | Identifier for logging and the CSV row. Falls back to the array index. |
| `question`    |   yes    | Natural-language question sent to the agent.                           |
| `SQL`         |   yes    | Expected (ground-truth) SQL.                                           |
| `answer_raw`  |    no    | Expected user-facing answer, compared against the agent's DB result.   |
| `difficulty`  |    no    | Passed through to the output CSV.                                      |

```json
[
  {
    "question_id": 3,
    "db_id": "WIDEWORLDIMPORTERS",
    "question": "calculate the customer count by state province name",
    "evidence": "",
    "SQL": "select sp.stateprovincename as state_province_name, count(c.customerid) as customer_count from sales.customers c join application.cities ct on c.deliverycityid = ct.cityid join application.stateprovinces sp on ct.stateprovinceid = sp.stateprovinceid group by sp.stateprovincename order by state_province_name desc;",
    "difficulty": "",
    "answer_raw": "",
    "answer": ""
  }
]
```

### `metadata.json`

An object keyed by table name, used only during ingestion to stamp descriptions
and sample values onto the Neo4j graph. Optional — enrichment is skipped if the
file is missing.

```json
{
  "sales.customers": {
    "description": "One row per customer account.",
    "columns": [
      {
        "name": "customername",
        "description": "Display name of the customer.",
        "value_examples": ["Tailspin Toys", "Wingtip Toys"]
      }
    ]
  }
}
```

### `custom_analyses.json`

An array of named example analyses. During ingestion each analysis is parsed
against the ingested schema, added to the graph as a `CustomAnalysis` node, and
embedded into the semantic vector store. Optional.

```json
[
  {
    "name": "Americas Region Countries",
    "description": "Countries that belong to the Americas region.",
    "sql": "select region from application.countries where region like '%Americas%'"
  }
]
```

## Workflows

Each workflow has its own README with purpose, run instructions (script and
launch.json), and outputs:

- **[Ingestion](ontology_sql_eval/ingestion/README.md)** — extract + embed a
  source DB's schema into Neo4j + pgvector and enrich the graph (requires GSF).
- **[Retrieval eval](ontology_sql_eval/retrieval/README.md)** — run the
  text-to-SQL agent against an eval set and score deterministically (requires
  GSF).
- **[SQL judge](ontology_sql_eval/judge/README.md)** — standalone LLM re-scorer
  for eval CSVs (no GSF required).

## Layout

```
ontology_sql_eval/          single namespace package
  judge/                    standalone LLM scorer (no GSF dependency)
    README.md               judge workflow guide
    config.py               resolves JUDGE_* settings from .env
    scorer.py               per-row LLM scoring call
    runner.py               batch/directory orchestration
    models.py               Pydantic score model + scoring prompt
    main.py                 CLI entry point (ontology-sql-eval)
    bird.py                 BIRD official EX + VES scoring (separate from the LLM judge)
  ingestion/                GSF-backed ingestion pipeline
    README.md               ingestion workflow guide
    ingest.py               source DB -> pgvector ingest (calls enrich_graph)
    semantic.py             compile the semantic layer over the ingested graph
    enrich_graph.py         graph metadata + custom-analysis enrichment
    mock_ingest.py          in-memory 4-table demo ingest (mock_shop)
  retrieval/                GSF-backed retrieval eval
    README.md               retrieval-eval workflow guide
    eval_chatbot.py         retrieval eval driver
    scoring.py              SQL/answer scoring helpers
main.py                     end-to-end pipeline entry point (ingest -> judge)
scripts/
  seed_wwi.py               seed a local Postgres from datasets/<db>/{ddl,data}
  seed_bird.py              download the BIRD Mini-Dev dataset into datasets/bird/
datasets/
  <database_name>/          evaluation.json, metadata.json, custom_analyses.json
    ddl/                    seed DDL (schemas, sequences, tables, indexes, fkeys, views)
    data/                   seed CSV data (COPYed into the tables)
  bird/                     README.md, evaluation.json, <db_id>/<db_id>.sqlite (11 DBs)
  wideworldimporters/       README.md, evaluation.json, custom_analyses.json, ddl/, data/
input/                      judge input CSVs to score (contents gitignored)
output/                     judge scored CSVs (<name>_scores.csv; contents gitignored)
```

## Development

VS Code launch configurations for all of the above (Run full pipeline, Judge,
Ingest, Semantic compile, Eval, Eval single-query, Seed local WWI,
Seed local BIRD) are provided in [.vscode/launch.json](.vscode/launch.json); they set
`PYTHONPATH=../GSF` where needed and load `.env` automatically. The type-checker
path for `../GSF` is configured in [pyrightconfig.json](pyrightconfig.json).

## License and security

This project is licensed under **Apache-2.0** (see the SPDX headers in the
source files and the `license` field in [pyproject.toml](pyproject.toml)).

To report a security vulnerability, follow the process in [SECURITY.md](SECURITY.md).
