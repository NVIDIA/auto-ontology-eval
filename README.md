# Ontology SQL Eeval

A Text-to-SQL evaluation toolkit for benchmarking a text-to-SQL agent against a
dataset and scoring the results. It bundles three workflows in one project:

1. **ingestion** — extract a source database's schema into GSF's Postgres +
   pgvector stores, compile the semantic layer, and enrich the graph with metadata and
   custom analyses.
2. **retrieval eval** — run the text-to-SQL agent against an evaluation set and
   score its generated SQL and answers deterministically.
3. **sql judge** — a standalone, LLM-powered re-scorer for Text-to-SQL
   evaluation CSVs (works on its own, no database or GSF/NeMo install required).

The typical lifecycle is **ingest → eval → judge**: you ingest a database so the
agent can retrieve its schema, run an evaluation to produce a results CSV, then
optionally re-score that CSV with the LLM judge.

> [!IMPORTANT]
> **You need [GSF](https://github.com/NVIDIA/GSF).**
>
> The ingestion and
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
| Postgres (catalog + pgvector) service                                        |    yes     |      yes       |  no   |
| Reachable source database                                                    |    yes     |      yes       |  no   |
| LLM API key                                                                  | embed only |      yes       |  yes  |

The two external dependencies are provided differently:

- `../GSF` (`https://github.com/NVIDIA/GSF`) — provides the `gsf` package. **Check it out next to this repo** so it
  resolves as `../GSF`; it is imported via `PYTHONPATH` (not installed — see
  below).
- `nemo-retriever` (`https://github.com/NVIDIA/NeMo-Retriever.git`) — provides
  `nemo_retriever`, installed from GitHub by `uv sync` (see `[tool.uv.sources]`
  in [pyproject.toml](pyproject.toml)).

For ingestion and retrieval eval you also need a live **Postgres** service
(it holds both GSF's catalog and the pgvector collections), plus a reachable
source DB. GSF's catalog schema must be migrated first:

```bash
cd ../GSF && uv run alembic upgrade head
```
 The easiest way to start the
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

> **Disclaimer**
>
> This project was tested
> with **GPT 5.5** (agent, via
> [https://inference-api.nvidia.com](https://inference-api.nvidia.com))
>
> and
> **Nemotron** (judge / embeddings, via
> [https://integrate.api.nvidia.com](https://integrate.api.nvidia.com)). See
> [.env.example](.env.example) for the exact model IDs used for each role.
>
> **Use only models that support structured output.**
>
> Other models may fail at
> runtime or produce invalid scores. In particular,
>
> All settings are read from `.env` (see [.env.example](.env.example) for the full
> list). Variables are grouped by the workflow that uses them:

| Variable                                                                    | Used by        | Description                                                                                                                             |
| --------------------------------------------------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `DEFAULT_MODELS_API_KEY`, `DEFAULT_MODELS_ENDPOINT`, `DEFAULT_MODELS_MODEL` | shared default | Default credentials/model for model-backed workflows; legacy `NVIDIA_API_KEY` / `BASE_URL` / `MODEL_NAME` remain fallbacks.             |
| `REASONING_API_KEY`, `REASONING_ENDPOINT`, `REASONING_MODEL`                | agent (eval)   | LLM that the text-to-SQL agent generates with (falls back to `DEFAULT_MODELS_*`).                                                       |
| `JUDGE_API_KEY`, `JUDGE_BASE_URL`, `JUDGE_MODEL_NAME`                       | judge          | LLM that the judge scores with (falls back to `DEFAULT_MODELS_*`, then legacy shared vars, then a built-in default for the key prefix). |
| `EMBED_API_KEY`, `EMBED_ENDPOINT`, `EMBED_MODEL`                            | ingest + eval  | Embedding endpoint (must be identical for ingest and query).                                                                            |
| `POSTGRES_*`                                                                | ingest + eval  | Catalog, semantic layer, and pgvector store connection.                                                                                  |
| `CONNECTION_STRINGS`                                                        | ingest + eval  | Source DB to extract schema from / execute SQL against.                                                                                 |

## Seed the local source DB (required)

> **Required to run this project.** Ingestion and retrieval eval need a reachable
> source database. Only the standalone judge can run without a source DB.

Each bundled dataset's README documents how to stand up its source DB and set
`CONNECTION_STRINGS` (see [Configuration](#configuration)):

- **[WideWorldImporters](datasets/wideworldimporters/README.md)** — seed a
  local Postgres from pre-generated DDL + CSVs.
- **[BIRD](datasets/bird/README.md)** — download per-database SQLite files
  (Dev or the smaller Mini-Dev subset).
- **[FDABench-Lite](datasets/fdabench/README.md)** — download Lite tasks +
  SQLite databases (BIRD train / Spider1 / Spider2-lite local).

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
`--skip-ingest --skip-semantic --skip-eval`).

The eval stage runs `--eval-workers` questions concurrently and can be limited
to a slice of the eval set with `--start-index` / `--end-index` / `--limit`
(`--workers` stays the judge's concurrency). A ten-question smoke run against
an already-ingested dataset, two questions at a time:

```bash
PYTHONPATH=../GSF uv run python main.py --database-name bird \
    --skip-ingest --skip-semantic --limit 10 --eval-workers 2
```

Each eval run writes a full instrumentation bundle to `logs/<run-id>/` —
per-question log files, per-agent-node timings, a phase timeline, and an
aggregated `summary.json`. See
[Retrieval eval → Run logs](ontology_sql_eval/retrieval/README.md#run-logs)
for what to read when hunting a bottleneck.

For concrete, copy-pasteable walkthroughs (including the manual per-stage
commands), see the per-dataset READMEs:
[WideWorldImporters](datasets/wideworldimporters/README.md),
[BIRD](datasets/bird/README.md). See the per-workflow READMEs under
[Workflows](#workflows) for the details of each stage.

## Semantic compile with our own descriptions

A plain compile keeps whatever column annotations the dataset shipped with.
`--override-descriptions` finishes the compile by writing our own saved
descriptions over them, in the graph and in both vector collections. They are
read from `annotations/<dataset>/semantic_descriptions.csv`, which the repo
tracks — nothing to configure. Without that file (or a
`semantic_descriptions.csv` beside the database) the flag writes nothing.

Run the semantic stage on its own — one database, or every database in
`CONNECTION_STRINGS` when `--database-name` is omitted:

```bash
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.semantic --override-descriptions
PYTHONPATH=../GSF uv run python -m ontology_sql_eval.ingestion.semantic --database-name <database_name> --override-descriptions
```

Add `--dataset-name bird` for a multi-database dataset, so both our export and
the fallback beside the database are looked for under that dataset's name. The
same flag works on the full pipeline, where it applies to the semantic stage:

```bash
PYTHONPATH=../GSF uv run python main.py --database-name <database_name> --override-descriptions
```

Re-running is cheap — the compile only visits tables that have no `Term` yet —
so edit the export and run the command again to push the edit through. See
[Publishing our descriptions and analyses](ontology_sql_eval/ingestion/README.md#publishing-our-descriptions-and-analyses)
for why descriptions wait for the compile while custom analyses go in during
ingest.

## Datasets

Each dataset lives in its own folder under `datasets/`. The folder name is used
to derive default file paths.

A standalone dataset is one database, so its files sit at the dataset root:

```
datasets/<database_name>/
  evaluation.json        # eval questions + expected SQL (retrieval-eval input)
  metadata.json          # table/column descriptions (ingestion enrichment; optional)
  custom_analyses.json   # named example analyses (ingestion enrichment; optional)
```

A multi-database dataset such as BIRD keeps one `evaluation.json` for the whole
dataset and gives each database its own folder, so the per-database files sit
beside the database itself:

```
datasets/<dataset>/
  evaluation.json        # questions across every database in the dataset
  dev/<database_name>/
    <database_name>.sqlite
    metadata.json        # optional
    custom_analyses.json # optional
```

A downloaded dataset's folder is gitignored and replaced wholesale by the next
download, so our own analyses and our corrections to its column annotations are
tracked outside it, under the dataset's name. They are preferred over the
optional files above, which stay the fallback:

```
annotations/<dataset>/
  custom_analyses/<database_name>.json   # ours, one per database
  semantic_descriptions.csv              # one export covering every database
```

Nothing has to be configured to use them — see
[Publishing our descriptions and analyses](ontology_sql_eval/ingestion/README.md#publishing-our-descriptions-and-analyses).

The retrieval eval writes its results CSV to the repo-root `input/` folder
(`input/<database_name>_<model>.csv`), not into the dataset folder, so the judge
can score it directly.

The repo ships with three public worked examples:

### BIRD

The [BIRD](https://bird-bench.github.io/) benchmark, plus the official
EX/VES scoring script. Defaults to the full **Dev** split (11 SQLite
databases, 1,534 questions); the smaller **Mini-Dev** subset (500 questions)
and the **Train** split's few-shot question corpus are also available. See
**[datasets/bird/README.md](datasets/bird/README.md)** for how to download
the dataset(s) and run the full pipeline.

### FDABench-Lite

The [FDABench](https://github.com/fdabench/FDAbench) Lite subset, mapped to
text-to-SQL: gold SQL from each task's subtasks, plus the SQLite databases
those tasks need (169 questions, 15 databases). See
**[datasets/fdabench/README.md](datasets/fdabench/README.md)** for how to
download and run the full pipeline.

### WideWorldImporters (WWI)

Microsoft's WideWorldImporters sample OLTP database, ported to Postgres and
seeded from pre-generated DDL + CSVs. See
**[datasets/wideworldimporters/README.md](datasets/wideworldimporters/README.md)**
for how to seed the database and run the full pipeline.

### BIRD-Interact

Runs against Lite (~300 tasks / 18 DBs) and Full (600 tasks / 22 DBs)
BIRD-Interact, c-Interact mode only, query-category tasks only. See
**[BIRD_INTERACT_README.md](BIRD_INTERACT_README.md)** for seeding, GT,
running, and results.

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
and sample values onto GSF's catalog. Optional — enrichment is skipped if the
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

Each workflow has its own README with purpose, run instructions, and outputs:

- **[Ingestion](ontology_sql_eval/ingestion/README.md)** — extract + embed a
  source DB's schema into GSF's Postgres catalog + pgvector (requires GSF).
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
    eval_chatbot.py         retrieval eval driver (--workers runs N in parallel)
    run_logging.py          per-run log bundle, phase timeline, timing summary
    scoring.py              SQL/answer scoring helpers
main.py                     end-to-end pipeline entry point (ingest -> judge)
scripts/
  seed_wwi.py               seed a local Postgres from datasets/<db>/{ddl,data}
  seed_bird.py              download BIRD split(s) (mini-dev/dev/train) into datasets/bird/
  seed_fdabench.py          download FDABench-Lite tasks + SQLite DBs into datasets/fdabench/
datasets/
  <database_name>/          evaluation.json, metadata.json, custom_analyses.json
    ddl/                    seed DDL (schemas, sequences, tables, indexes, fkeys, views)
    data/                   seed CSV data (COPYed into the tables)
  bird/                     README.md, evaluation.json,
                            dev/<db_id>/ -> <db_id>.sqlite + metadata.json (11 DBs),
                            train/train.json (few-shot corpus; contents gitignored)
  fdabench/                 README.md, evaluation.json, <db_id>/<db_id>.sqlite (15 DBs)
  wideworldimporters/       README.md, evaluation.json, custom_analyses.json, ddl/, data/
input/                      judge input CSVs to score (contents gitignored)
output/                     judge scored CSVs (<name>_scores.csv; contents gitignored)
logs/<run-id>/              per-eval-run logs + timings (contents gitignored)
```

## Development

VS Code launch configurations for all of the above (Run full pipeline, Judge,
Ingest, Semantic compile, Eval, Eval single-query, Seed local WWI,
Seed local BIRD, Seed local FDABench) are provided in [.vscode/launch.json](.vscode/launch.json); they set
`PYTHONPATH=../GSF` where needed and load `.env` automatically. The type-checker
path for `../GSF` is configured in [pyrightconfig.json](pyrightconfig.json).

## Contributing

This project is currently not accepting contributions.

## License and security

This project is licensed under **Apache-2.0** (see [LICENSE](LICENSE), the SPDX
headers in each source file, and the `license` field in
[pyproject.toml](pyproject.toml)). Third-party dependency licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Two items of third-party content are included in this repository, both MIT
licensed and documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md#included-third-party-content):

- [`ontology_sql_eval/judge/bird.py`](ontology_sql_eval/judge/bird.py) — ports
  evaluation logic from the BIRD benchmark
  ([AlibabaResearch/DAMO-ConvAI](https://github.com/AlibabaResearch/DAMO-ConvAI),
  Copyright (c) 2022 Alibaba Research).
- [`datasets/wideworldimporters/`](datasets/wideworldimporters/) — a Postgres
  port of Microsoft's WideWorldImporters sample database
  ([microsoft/sql-server-samples](https://github.com/microsoft/sql-server-samples),
  Copyright (c) Microsoft Corporation). Its upstream notice is reproduced in
  [datasets/wideworldimporters/LICENSE](datasets/wideworldimporters/LICENSE).

To report a security vulnerability, follow the process in [SECURITY.md](SECURITY.md).
