# Ontology SQL Eval

A Text-to-SQL evaluation toolkit for benchmarking a text-to-SQL agent against a
dataset and scoring the results. It bundles two workflows in one project:

1. **ingestion** — extract a source database's schema into Auto Ontology's
   Postgres + pgvector stores, compile the semantic layer, and enrich the graph
   with metadata and custom analyses.
2. **NeMo Gym benchmark** — run an evaluation set through two arms, a
   schema-only LLM control and the ontology-grounded agent, scored by one
   deterministic verifier and reported as a single delta.

The typical lifecycle is **ingest → semantic compile → benchmark**: you ingest a
database so the agent can retrieve its schema, then run the benchmark to see how
much that grounding is worth against a schema-only control. No LLM judge is
involved: scoring is deterministic execution match, with each benchmark's
official rule where it has one (BIRD, BEAVER).

> [!IMPORTANT]
> **Auto Ontology is installed automatically.**
>
> `uv sync` installs the pinned
> [NVIDIA/auto-ontology](https://github.com/NVIDIA/auto-ontology) Git revision.
> No sibling checkout or `PYTHONPATH` configuration is required.

## Prerequisites

| Requirement                                                                  | Ingestion  |   Benchmark    |
| ---------------------------------------------------------------------------- | :--------: | :------------: |
| [uv](https://docs.astral.sh/uv/) + Python 3.13 (≥3.13.14; uv installs it)   |    yes     |      yes       |
| Network access to GitHub (`NVIDIA/auto-ontology`) and PyPI                   |    yes     |      yes       |
| Postgres (catalog + pgvector) service                                        |    yes     |      yes       |
| Reachable source database                                                    |    yes     |      yes       |
| LLM API key                                                                  | embed only |      yes       |

`uv sync` installs Auto Ontology from its pinned GitHub revision and everything
else from PyPI; versions are pinned in
[pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

For ingestion and the benchmark you also need a live **Postgres** service
(it holds both Auto Ontology's catalog and the pgvector collections), plus a
reachable source DB. The catalog's schema has to be migrated before first use,
and again whenever the Auto Ontology pin changes; see [Setup](#setup).

## Setup

```bash
uv sync                # creates .venv and installs pinned Git dependencies
cp .env.example .env   # then fill in your values
```

`main.py` keeps the catalog database (`POSTGRES_*` in `.env`) on the schema
the installed Auto Ontology expects: before any stage, it checks the database's
migration revision and, if it's behind, downloads Auto Ontology's migrations for
the exact commit in `.venv` and applies them (the installed package doesn't
include them). To run that step on its own:

```bash
uv run python scripts/migrate_catalog.py
```

Migrating changes the database for every application using it, so give this
project its own `POSTGRES_DATABASE`. If you can't, pass `--skip-migrate`;
`main.py` also stops, rather than downgrading, when the database was migrated
by a newer Auto Ontology.

## Configuration

> **Tested configuration**
>
> This project is tested with a single `sk-` key against
> [inference-api.nvidia.com](https://inference-api.nvidia.com) and no model
> pinned in `.env`, so every role uses the built-in default for that key:
>
> | Role | Model |
> | --- | --- |
> | Agent (ontology arm) and control arm | `aws/anthropic/bedrock-claude-opus-4-8` |
> | Agent's non-reasoning calls | `aws/anthropic/bedrock-claude-sonnet-5-5` |
> | Embeddings | `nvidia/nvidia/nemotron-3-embed-1b` |
>
> The agent and embedding defaults come from the pinned Auto Ontology revision,
> so they can change when it is bumped. An `nvapi-` key selects Nemotron
> defaults on integrate.api.nvidia.com instead.
>
> **Use only models that support structured output.** Other models may fail at
> runtime.

All settings are read from `.env` (see [.env.example](.env.example) for the full
list). Variables are grouped by the workflow that uses them:

| Variable                                              | Used by       | Description                                                           |
| ----------------------------------------------------- | ------------- | --------------------------------------------------------------------- |
| `DEFAULT_MODELS_API_KEY`, `DEFAULT_MODELS_ENDPOINT`, `DEFAULT_MODELS_MODEL` | shared default | Default credentials/model for every model-backed workflow. The key is **required** by the benchmark: both arms authenticate with it. |
| `REASONING_API_KEY`, `REASONING_ENDPOINT`, `REASONING_MODEL` | ontology arm  | LLM the Auto Ontology agent generates SQL with (falls back to `DEFAULT_MODELS_*`). |
| `GYM_MODEL`, `GYM_MODEL_URL`                          | control arm   | Model the schema-only control answers with (default `aws/anthropic/bedrock-claude-opus-4-8` at `https://inference-api.nvidia.com/v1`). Optional. |
| `EMBED_API_KEY`, `EMBED_ENDPOINT`, `EMBED_MODEL`      | ingest + benchmark | Embedding endpoint (must be identical for ingest and query).     |
| `POSTGRES_*`                                          | ingest + benchmark | Store connection — Auto Ontology's catalog and pgvector collections. |
| `CONNECTION_STRINGS`                                  | ingest + benchmark | Source DB to extract schema from / execute SQL against.          |

To compare like with like, point `GYM_MODEL` and `REASONING_MODEL` at the same
model.

## Seed the local source DB (required)

> **Required to run this project.** Ingestion and the benchmark need a reachable
> source database.

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
uv run python main.py --database-name <database_name>
```

The benchmark stage builds both arms' task files, runs the schema-only control
and the ontology-grounded arm, and compares them. It measures how much the
ontology is worth, not just how well the agent does.

Stages can be skipped with `--skip-ingest`, `--skip-semantic` and `--skip-eval`
(the benchmark). A ten-question smoke run against an
already-ingested dataset, three questions at a time:

```bash
uv run python main.py --database-name bird \
    --skip-ingest --skip-semantic --limit 10 --eval-workers 3
```

`--eval-workers` is per-arm concurrency; size it against the model endpoint's
**rate limit**, not CPU.

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
[BIRD](datasets/bird/README.md),
[FDABench-Lite](datasets/fdabench/README.md),
[BEAVER](datasets/beaverbench/README.md). See the per-workflow READMEs under
[Workflows](#workflows) for the details of each stage.

## Extra prompts: SQL examples and value anchors

Two optional CSVs can be injected into the ontology arm's SQL-generation
prompt. Both are keyed by `question_id` and both are built from BIRD Dev:

| File                                                    | `main.py` flag        | What it carries                                                                                                |
| ------------------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------- |
| `inputs/extra_prompts/bird/predicted_structural_exemplars.csv` | `--sql-examples-csv`  | Train questions whose SQL has the same shape as the one this question needs, ranked — few-shot exemplars.      |
| `inputs/extra_prompts/bird/value_anchors.csv`                  | `--value-anchors-csv` | Which column stores a phrase from the question, and its exact stored spelling — so filters match real values. |

Generated copies of both are committed, so a run can reuse them without
regenerating. To rebuild them, [`scripts/extra_prompts/build_eval_inputs.py`](scripts/extra_prompts/build_eval_inputs.py)
runs every step in order and writes both files:

```bash
uv run python scripts/extra_prompts/build_eval_inputs.py
```

It needs the BIRD **Dev** split *and* its Train corpus on disk
(`uv run python scripts/seed/seed_bird.py --splits dev`, see
[datasets/bird/README.md](datasets/bird/README.md)) plus the `REASONING_*`
credentials from [Configuration](#configuration) — the exemplar step asks the
model to predict a query for every Dev question, so it is the slow and the only
paid half of the run.

Anchors are the cheap half: no model is involved, only a value index built by
walking the Dev SQLite databases. That index is rebuilt every time, because it
is derived wholly from those databases and a stale one produces stale anchors
without saying so.

| Option              | Default                | Purpose                                                        |
| -------------------- | ---------------------- | ---------------------------------------------------------------- |
| `--only anchors`     | both                   | Rebuild just the anchors — skips the model entirely.            |
| `--only exemplars`   | both                   | Rebuild just the exemplars.                                    |
| `--k N`              | `5`                    | Exemplars kept per question.                                   |
| `--alternatives N`   | `3`                    | Structurally different readings predicted per question; the `--k` slots are spread across them rather than all taken from one. `0` restores one prediction per question. |
| `--no-schema`        | off                    | Predict without showing the question's own schema, which is what the join count is counted from. |
| `--workers N`        | `8`                    | Concurrent model requests in the exemplar step.                |
| `--limit N`          | all                    | First N Dev questions, for a smoke test.                       |
| `--dry-run`          | off                    | Print the commands that would run, and run none of them.       |
| `--anchors-out`, `--exemplars-out` | `inputs/extra_prompts/bird/` | Write somewhere else.                              |

The exemplar step caches model responses next to its CSV. A rebuild archives
any existing cache under a timestamped name first, so the answers are always
resampled rather than replayed — caches are gitignored, the CSVs are not.

Pass the results to the benchmark's ontology arm (the flags are independent;
either can be omitted):

```bash
uv run python main.py --database-name bird --skip-ingest --skip-semantic \
    --sql-examples-csv inputs/extra_prompts/bird/predicted_structural_exemplars.csv \
    --sql-examples-k 5 \
    --value-anchors-csv inputs/extra_prompts/bird/value_anchors.csv
```

Neither is passed by default, so a run without them has no exemplars and no
anchors. The control arm gets no equivalent, so a delta measured with them
covers the inputs as well as the ontology grounding.

## Datasets

Each dataset lives in its own folder under `datasets/<database_name>/`. The
folder name is the database name and is used to derive default file paths.

```
datasets/<database_name>/
  evaluation.json        # eval questions + expected SQL (benchmark input)
  metadata.json          # table/column descriptions (ingestion enrichment; optional)
  custom_analyses.json   # named example analyses (ingestion enrichment; optional)
```

The repo ships with four public worked examples:

### BIRD

The [BIRD](https://bird-bench.github.io/) benchmark. Defaults to full **Dev** (11 SQLite databases,
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
**[scripts/bird_interact/README.md](scripts/bird_interact/README.md)** for seeding, GT,
running, and results.

The schemas below (`evaluation.json`, `metadata.json`, `custom_analyses.json`)
are generic across datasets; the examples use `wideworldimporters` throughout.

### `evaluation.json`

A JSON **array** of question objects. Fields consumed by the benchmark:

| Field         | Required | Purpose                                                                |
| ------------- | :------: | ---------------------------------------------------------------------- |
| `question_id` |    no    | Task id both arms are joined on. Falls back to a hash of the question. |
| `db_id`       |   yes    | Database the question runs against.                                    |
| `question`    |   yes    | Natural-language question sent to the agent.                           |
| `evidence`    |    no    | Hint text appended to the question.                                    |
| `SQL`         |   yes    | Expected (ground-truth) SQL.                                           |
| `difficulty`  |    no    | Reported as a per-difficulty accuracy breakdown.                       |

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
and sample values onto Auto Ontology's catalog. Optional — enrichment is
skipped if the file is missing.

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
  source DB's schema into Auto Ontology's Postgres catalog + pgvector.
- **[NeMo Gym benchmark](resources_servers/README.md)** — schema-only control
  vs the ontology-grounded agent, one deterministic verifier, one delta
  (requires Auto Ontology for the ontology arm). This is `main.py`'s evaluation
  stage.

### NeMo Gym benchmark

`main.py` runs this end to end (see [Full pipeline](#full-pipeline-end-to-end-example))
and is the way to produce a comparison — it rebuilds the task files, verifies
they hold the dataset you asked for, runs both arms, and stops on the first
failure:

```bash
uv run python main.py --database-name bird --skip-ingest --skip-semantic \
  --eval-workers 10 --run-dir runs/full_$(date +%Y%m%d-%H%M%S)
```

Add `--sql-examples-csv` / `--value-anchors-csv` to give the ontology arm the
question-keyed extra prompts — the full form, which is what the reported BIRD
delta was measured with:

```bash
uv run python main.py --database-name bird --skip-ingest --skip-semantic \
  --eval-workers 10 --run-dir runs/full_$(date +%Y%m%d-%H%M%S) \
  --sql-examples-csv inputs/extra_prompts/bird/predicted_structural_exemplars.csv \
  --sql-examples-k 5 \
  --value-anchors-csv inputs/extra_prompts/bird/value_anchors.csv
```

The control arm cannot be given an equivalent, so a delta measured this way
covers those inputs as well as the ontology grounding. That is why they are off
unless asked for.

The stages are also drivable individually, against whatever task files are
already on disk:

```bash
# 1. build task files, once per dataset and arm
uv run python scripts/build_gym_tasks.py --dataset bird --arm schema_only
uv run python scripts/build_gym_tasks.py --dataset bird --arm auto_ontology

# 2. run each arm into its OWN output directory
scripts/run_gym_arm.sh schema_only_sql   runs/control/bird.jsonl
scripts/run_gym_arm.sh auto_ontology_sql runs/treatment/bird.jsonl --max-output-tokens 16

# 3. compare
uv run python -m ontology_sql_eval.gym.compare \
  --control runs/control/bird.jsonl --treatment runs/treatment/bird.jsonl
```

`run_gym_arm.sh` reuses existing server venvs, so pass `GYM_REBUILD_VENV=1`
after changing a server's dependencies. See
[resources_servers/README.md](resources_servers/README.md#running-an-arm).

> **If you built a Gym server venv before these pins existed, rebuild it once:**
> `GYM_REBUILD_VENV=1 scripts/run_gym_arm.sh auto_ontology_sql /tmp/probe.jsonl`.
> The venv is reused by default. `run_gym_arm.sh` rebuilds on its own when a
> server venv's nemo-gym or Ray differs from `.venv`'s, but not for other
> packages: an existing venv keeps the Auto Ontology and openai it was built
> with, so it misses an Auto Ontology bump, and openai 2.45+ rejects the
> inference-api gateway's responses. A venv built from scratch is already
> correct.

Datasets: `bird`, `fdabench`, `wideworldimporters` (Postgres), `beaverbench`
(MySQL). The ontology arm additionally needs the database ingested and its
semantic layer compiled (workflow 1) — it drives the agent in-process, importing
the installed `auto_ontology` package.

Knobs: `--eval-workers` on `main.py` (size against the model endpoint's **rate
limit**, not CPU; 10 is measured-safe on the `sk-` key), or `GYM_CONCURRENCY`
when driving an arm directly. `GYM_API_KEY` overrides the key for both the
policy model and the agent's own calls; `GYM_MODEL` and `GYM_MODEL_URL` pick the
endpoint.

> [!NOTE]
> Driving the ontology arm directly, `GYM_CONCURRENCY` alone leaves it throttled:
> its server has a second ceiling (`max_concurrency`, default 3). `main.py`
> raises both from `--eval-workers`.

> [!WARNING]
> Read the `health/*` counters next to the accuracy. A rate-limited or otherwise
> broken run scores near zero and is indistinguishable from a weak model;
> `ontology_sql_eval/gym/compare.py` refuses to print a headline number when more than 5% of tasks
> failed that way.

> [!NOTE]
> BIRD databases live under `datasets/bird/dev/<db_id>/`. A seed made before that
> layout change reports `gold_execution_error` on every task — re-seed with
> `scripts/seed/seed_bird.py`, or move the per-database directories under `dev/`.

## Layout

```
main.py                     end-to-end pipeline entry point (ingest -> benchmark)
ontology_sql_eval/          single namespace package
  ingestion/                Auto Ontology-backed ingestion pipeline
    README.md               ingestion workflow guide
    ingest.py               source DB -> pgvector ingest (calls enrich_graph)
    semantic.py             compile the semantic layer over the ingested graph
    enrich_graph.py         graph metadata + custom-analysis enrichment
  gym/                      the benchmark logic shared by both arms and main.py
    tasks.py                build Gym task files from datasets/<name>/evaluation.json
    ddl.py                  schema dumps for the schema-only arm's prompt
    exec_match.py           execution-match verifier (BIRD EX, BEAVER ex_acc)
    model_routing.py        routes Auto Ontology's model calls through Gym's model servers
    compare.py              health-gated delta report between the two arms
  bird_interact/            BIRD-Interact adapter server (see scripts/bird_interact/README.md)
  env.py                    .env loading
  extra_prompts.py          loaders for the extra-prompt CSVs
resources_servers/          the two Gym arms: schema_only_sql (control), auto_ontology_sql
scripts/
  build_gym_tasks.py        build one arm's Gym task file
  run_gym_arm.sh            run one arm through NeMo Gym
  migrate_catalog.py        migrate the catalog DB to the installed Auto Ontology's schema
  seed/                     stand up a dataset's source DBs under datasets/
    seed_bird.py            BIRD eval data and the Dev Train corpus
    seed_fdabench.py        FDABench-Lite tasks + SQLite DBs
    seed_beaverbench.py     BEAVER questions + MySQL dumps
    seed_wwi.py             a local Postgres from datasets/<db>/{ddl,data}
  extra_prompts/            build the CSVs under inputs/extra_prompts/
    build_eval_inputs.py    both CSVs (anchors + exemplars) in one run
    value_index.py          value -> column index over the BIRD Dev DBs; value anchors
    llm_structural_exemplars.py  LLM-predicted SQL matched against train; exemplars
  bird_interact/            BIRD-Interact: README, seeding, runners, analysis, ADK patches
datasets/
  <database_name>/          evaluation.json, metadata.json, custom_analyses.json
    ddl/                    seed DDL (schemas, sequences, tables, indexes, fkeys, views)
    data/                   seed CSV data (COPYed into the tables)
  bird/                     README.md, evaluation.json, dev/<db_id>/<db_id>.sqlite (11 DBs)
  fdabench/                 README.md, evaluation.json, <db_id>/<db_id>.sqlite (15 DBs)
  beaverbench/              README.md, evaluation.json, dumps/*.sql, <db_id>/metadata.json
  wideworldimporters/       README.md, evaluation.json, custom_analyses.json, ddl/, data/
inputs/                     tracked inputs we add on top of the datasets
  annotations/<dataset>/    descriptions + custom analyses applied at ingest
  extra_prompts/<dataset>/  eval extra prompts (sql examples + value anchors)
tests/                      pytest suite, one folder per area (gym/, ingestion/, bird_interact/, scripts/)
runs/{control,treatment}/   benchmark rollouts + Gym metrics (gitignored)
output/                     scratch output of the extra-prompt scripts (gitignored)
```

## Development

VS Code launch configurations (Run full pipeline, Benchmark (BIRD, 1 question),
Ingest, Semantic compile, and Seed local WWI / BIRD / FDABench / BEAVER)
are provided in [.vscode/launch.json](.vscode/launch.json); they load `.env`
automatically. The type checker uses the project's `.venv`, as
configured in [pyrightconfig.json](pyrightconfig.json).

## Contributing

This project is currently not accepting contributions.

## License and security

This project is licensed under **Apache-2.0** (see [LICENSE](LICENSE), the SPDX
headers in each source file, and the `license` field in
[pyproject.toml](pyproject.toml)). Third-party dependency licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Third-party content included in this repository is MIT licensed and
documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md#included-third-party-content):

- [`datasets/wideworldimporters/`](datasets/wideworldimporters/) — a Postgres
  port of Microsoft's WideWorldImporters sample database
  ([microsoft/sql-server-samples](https://github.com/microsoft/sql-server-samples),
  Copyright (c) Microsoft Corporation). Its upstream notice is reproduced in
  [datasets/wideworldimporters/LICENSE](datasets/wideworldimporters/LICENSE).
- [`tests/gym/beaver_reference.py`](tests/gym/beaver_reference.py) —
  ports BEAVER's official execution-accuracy rule
  ([beaverbench/beaver](https://github.com/beaverbench/beaver),
  Copyright (c) 2024 MIT Data Systems Group).

To report a security vulnerability, follow the process in [SECURITY.md](SECURITY.md).
