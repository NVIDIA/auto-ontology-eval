# Ontology SQL Eeval

A Text-to-SQL evaluation toolkit for benchmarking a text-to-SQL agent against a
dataset and scoring the results. It bundles three workflows in one project:

1. **ingestion** — extract a source database's schema into Auto Ontology's
   Postgres + pgvector stores, compile the semantic layer, and enrich the graph
   with metadata and custom analyses.
2. **NeMo Gym benchmark** — run an evaluation set through two arms, a
   schema-only LLM control and the ontology-grounded agent, scored by one
   deterministic verifier and reported as a single delta.
3. **sql judge** — a standalone, LLM-powered re-scorer for Text-to-SQL
   evaluation CSVs (works on its own, no database or Auto Ontology install
   required).

The typical lifecycle is **ingest → semantic compile → benchmark**: you ingest a
database so the agent can retrieve its schema, then run the benchmark to see how
much that grounding is worth against a schema-only control.

The judge is a separate tool, not a stage of that flow. It scores eval CSVs
produced by the sharded runners (`scripts/run_eval_shard.py`), which is how the
BEAVER pipeline uses it. The benchmark itself uses no LLM judge — scoring is
deterministic execution match.

> [!IMPORTANT]
> **You need [Auto Ontology](https://github.com/NVIDIA/auto-ontology).**
>
> The ingestion and
> retrieval-eval workflows import the `auto_ontology` package from a **sibling
> `../GSF` checkout** ([NVIDIA/auto-ontology](https://github.com/NVIDIA/auto-ontology)).
> The repo was renamed; the directory is still `../GSF` because
> `.vscode/launch.json`, `pyrightconfig.json` and several scripts hardcode that
> path, so clone it as `GSF` (or symlink it)
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
| Sibling repo `../GSF` checkout ([NVIDIA/auto-ontology](https://github.com/NVIDIA/auto-ontology)) |    yes     |      yes       |  no   |
| GitHub access to `NVIDIA/NeMo-Retriever`                                     |    yes     |      yes       |  no   |
| Postgres (catalog + pgvector) service                                        |    yes     |      yes       |  no   |
| Reachable source database                                                    |    yes     |      yes       |  no   |
| LLM API key                                                                  | embed only |      yes       |  yes  |

The two external dependencies are provided differently:

- `../GSF` (`https://github.com/NVIDIA/auto-ontology`) — provides the `auto_ontology` package. **Check it out next to this repo** so it
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

Auto Ontology is `package = false`, so `auto_ontology` is not installed into the venv — it imports
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

| Variable                                              | Used by       | Description                                                           |
| ----------------------------------------------------- | ------------- | --------------------------------------------------------------------- |
| `DEFAULT_MODELS_API_KEY`, `DEFAULT_MODELS_ENDPOINT`, `DEFAULT_MODELS_MODEL` | shared default | Default credentials/model for model-backed workflows; legacy `NVIDIA_API_KEY` / `BASE_URL` / `MODEL_NAME` remain fallbacks. |
| `REASONING_API_KEY`, `REASONING_ENDPOINT`, `REASONING_MODEL` | agent (eval) | LLM that the text-to-SQL agent generates with (falls back to `DEFAULT_MODELS_*`). |
| `JUDGE_API_KEY`, `JUDGE_BASE_URL`, `JUDGE_MODEL_NAME` | judge         | LLM that the judge scores with (falls back to `DEFAULT_MODELS_*`, then legacy shared vars, then a built-in default for the key prefix). |
| `EMBED_API_KEY`, `EMBED_ENDPOINT`, `EMBED_MODEL`      | ingest + eval | Embedding endpoint (must be identical for ingest and query).          |
| `POSTGRES_*`                                          | ingest + eval | Store connection — GSF's catalog and the pgvector collections.        |
| `CONNECTION_STRINGS`                                  | ingest + eval | Source DB to extract schema from / execute SQL against.               |

## Seed the local source DB (required)

> **Required to run this project.** Ingestion and retrieval eval need a reachable
> source database. Only the standalone judge can run without a source DB.

Each bundled dataset's README documents how to stand up its source DB and set
`CONNECTION_STRINGS` (see [Configuration](#configuration)):

- **[WideWorldImporters](datasets/wideworldimporters/README.md)** — seed a
  local Postgres from pre-generated DDL + CSVs.
- **[BIRD](datasets/bird/README.md)** — download Dev or Mini-Dev SQLite files;
  Dev also includes the Train question corpus.
- **[FDABench-Lite](datasets/fdabench/README.md)** — download Lite tasks +
  SQLite databases (BIRD train / Spider1 / Spider2-lite local).
- **[BEAVER](datasets/beaverbench/README.md)** — download BeaverBench questions +
  MySQL dumps (`dw` / `nova` / `neutron`).

## Full pipeline (end-to-end example)

`main.py` runs the lifecycle for a dataset in one shot — ingest → semantic
compile → **NeMo Gym benchmark** — writing rollouts to `runs/control/` and
`runs/treatment/` and printing the delta between them (assumes the stores are up
and `.env` is filled in):

```bash
PYTHONPATH=../GSF uv run python main.py --database-name <database_name>
```

The benchmark stage builds both arms' task files, runs the schema-only control
and the ontology-grounded arm, and compares them. It replaced the old
single-arm retrieval eval: that measured how well the agent does, this measures
how much the ontology is worth.

Stages can be skipped with `--skip-ingest`, `--skip-semantic`, `--skip-eval`
(the benchmark) and `--skip-judge`. A ten-question smoke run against an
already-ingested dataset, three questions at a time:

```bash
PYTHONPATH=../GSF uv run python main.py --database-name bird \
    --skip-ingest --skip-semantic --limit 10 --eval-workers 3
```

`--eval-workers` is per-arm concurrency; size it against the model endpoint's
**rate limit**, not CPU.

> [!NOTE]
> The **LLM judge** scores an eval CSV, and the benchmark writes Gym rollouts
> rather than a CSV, so the judge is skipped whenever the benchmark runs. It
> still runs over a CSV produced elsewhere — which is how
> `run_beaver_shards.sh` and `resume_beaver_ranges.sh` use this entry point:
> `--skip-eval` to reach the judge. The standalone `ontology-sql-eval` console
> script does the same job directly.

Each arm writes Gym's own artifacts next to its rollouts —
`*_aggregate_metrics.json` (pass@1 overall and per difficulty, plus `health/*`
counters) and `*_failures.jsonl`. Read the health counters before quoting any
accuracy: a rate-limited or broken run scores near zero and is indistinguishable
from a weak model.

To drive the stages by hand — build task files, run one arm, compare — see
[NeMo Gym benchmark](#nemo-gym-benchmark) below and the full guide in
[resources_servers/README.md](resources_servers/README.md).

For concrete, copy-pasteable walkthroughs, see the per-dataset READMEs:
[WideWorldImporters](datasets/wideworldimporters/README.md),
[BIRD Mini-Dev](datasets/bird/README.md),
[FDABench-Lite](datasets/fdabench/README.md),
[BEAVER](datasets/beaverbench/README.md). See the per-workflow READMEs under
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

The repo ships with four public worked examples:

### BIRD

The [BIRD](https://bird-bench.github.io/) benchmark, plus the official
EX/VES scoring script. Defaults to full **Dev** (11 SQLite databases,
1,534 questions) and stores the **Train** question corpus without its
databases; the cheaper **Mini-Dev** subset (500 questions) is also available. See
**[datasets/bird/README.md](datasets/bird/README.md)** for how to download
the dataset(s) and run the full pipeline.

### FDABench-Lite

The [FDABench](https://github.com/fdabench/FDAbench) Lite subset, mapped to
text-to-SQL: gold SQL from each task's subtasks, plus the SQLite databases
those tasks need (169 questions, 15 databases). See
**[datasets/fdabench/README.md](datasets/fdabench/README.md)** for how to
download and run the full pipeline.

### BEAVER (BeaverBench)

The [BEAVER](https://beaverbench.github.io/) enterprise text-to-SQL benchmark
(gated HuggingFace questions + MySQL dumps). Defaults to a 100-question `dw`
sample. See **[datasets/beaverbench/README.md](datasets/beaverbench/README.md)**
for how to download, import MySQL, and run the full pipeline.

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
- **[NeMo Gym benchmark](resources_servers/README.md)** — schema-only control
  vs the ontology-grounded agent, one deterministic verifier, one delta
  (requires Auto Ontology for the ontology arm). This is `main.py`'s evaluation
  stage.
- **[Retrieval eval](ontology_sql_eval/retrieval/README.md)** — the single-arm
  agent runner behind `scripts/run_eval_shard.py`. No longer a `main.py` stage;
  it produces the CSVs the judge scores (requires Auto Ontology).
- **[SQL judge](ontology_sql_eval/judge/README.md)** — standalone LLM re-scorer
  for eval CSVs (no GSF required).

### NeMo Gym benchmark

`main.py` runs this end to end (see [Full pipeline](#full-pipeline-end-to-end-example));
the commands below drive the stages individually. Full guide:
[resources_servers/README.md](resources_servers/README.md).

```bash
# 1. build task files, once per dataset and arm
uv run python scripts/build_gym_tasks.py --dataset bird --arm schema_only
uv run python scripts/build_gym_tasks.py --dataset bird --arm auto_ontology

# 2. run each arm into its OWN output directory
scripts/run_gym_arm.sh schema_only_sql   runs/control/bird.jsonl
scripts/run_gym_arm.sh auto_ontology_sql runs/treatment/bird.jsonl --max-output-tokens 16

# 3. compare
uv run python scripts/compare_arms.py \
  --control runs/control/bird.jsonl --treatment runs/treatment/bird.jsonl
```

Datasets: `bird`, `fdabench`, `wideworldimporters` (Postgres), `beaverbench`
(MySQL). The ontology arm additionally needs the database ingested and its
semantic layer compiled (workflow 1) — it drives the agent in-process, importing
it from `AUTO_ONTOLOGY_PATH`.

Knobs: `GYM_CONCURRENCY` (default 3 — size against the model endpoint's **rate
limit**, not CPU), `GYM_API_KEY` (overrides the key for both the policy model and
the agent's own calls), `AUTO_ONTOLOGY_PATH`.

> [!WARNING]
> Read the `health/*` counters next to the accuracy. A rate-limited or otherwise
> broken run scores near zero and is indistinguishable from a weak model;
> `compare_arms.py` refuses to print a headline number when more than 5% of tasks
> failed that way.

> [!NOTE]
> BIRD databases live under `datasets/bird/dev/<db_id>/`. A seed made before that
> layout change reports `gold_execution_error` on every task — re-seed with
> `scripts/seed_bird.py`, or move the per-database directories under `dev/`.

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
  seed_bird.py              download BIRD eval data and the Dev Train corpus
  seed_fdabench.py          download FDABench-Lite tasks + SQLite DBs into datasets/fdabench/
  seed_beaverbench.py       download BEAVER questions + MySQL dumps into datasets/beaverbench/
datasets/
  <database_name>/          evaluation.json, metadata.json, custom_analyses.json
    ddl/                    seed DDL (schemas, sequences, tables, indexes, fkeys, views)
    data/                   seed CSV data (COPYed into the tables)
  bird/                     README.md, evaluation.json, <db_id>/<db_id>.sqlite (11 DBs)
  fdabench/                 README.md, evaluation.json, <db_id>/<db_id>.sqlite (15 DBs)
  beaverbench/              README.md, evaluation.json, dumps/*.sql, <db_id>/metadata.json
  wideworldimporters/       README.md, evaluation.json, custom_analyses.json, ddl/, data/
input/                      judge input CSVs to score (contents gitignored)
output/                     judge scored CSVs (<name>_scores.csv; contents gitignored)
logs/<run-id>/              per-eval-run logs + timings (contents gitignored)
```

## Development

VS Code launch configurations for all of the above (Run full pipeline, Judge,
Ingest, Semantic compile, Eval, Eval single-query, Seed local WWI,
Seed local BIRD, Seed local FDABench, Seed local BEAVER) are provided in [.vscode/launch.json](.vscode/launch.json); they set
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
