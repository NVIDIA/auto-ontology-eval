# Auto-Ontology Eval

Measures how much an [Auto-Ontology](https://github.com/NVIDIA/auto-ontology)
ontology improves text-to-SQL accuracy, on public benchmarks and on your own
data.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

## Why it matters

Auto-Ontology exists to help agents give fewer wrong answers on enterprise
data. This repository measures whether it does.

A trustworthy accuracy number needs a control. The eval asks one question: how
much does Auto-Ontology's ontology add over the same model given only the
database schema? Each run asks the same questions of the same model twice and
scores the SQL from both arms with one deterministic verifier. The result is a
single accuracy difference, plus the list of questions the ontology fixed and
the ones it broke, so you can see where the difference comes from.

## How it works

A run has three stages, and `main.py` runs them in order:

1. **Ingest.** Extract the source database's schema into Auto-Ontology's catalog
   and pgvector store, adding table descriptions and example analyses when the
   dataset provides them.
2. **Compile.** Build the ontology over the ingested catalog.
3. **Benchmark.** Run every question through two arms on
   [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym), then compare them.

| Arm       | Server              | What the model gets                                                             |
| --------- | ------------------- | ------------------------------------------------------------------------------- |
| Control   | `schema_only_sql`   | The question and a schema dump, in one prompt with no tools                     |
| Treatment | `auto_ontology_sql` | The question; Auto-Ontology's agent retrieves its own context from the ontology |

The verifier executes each arm's SQL and the gold SQL against the same database
and compares the results. It applies each benchmark's official rule: BIRD's
execution accuracy for most datasets, and BEAVER's `ex_acc` for BEAVER. Scoring
is deterministic, with no LLM judge.

`uv sync` installs Auto-Ontology from the Git revision pinned in
[`pyproject.toml`](./pyproject.toml), so the treatment arm runs that exact
version in-process.

## Benchmarks and datasets

| Dataset                                               | Questions                   | Engine   | Setup                                                                  |
| ----------------------------------------------------- | --------------------------- | -------- | ---------------------------------------------------------------------- |
| [BIRD](https://bird-bench.github.io/)                 | 1,534 (Dev), 500 (Mini-Dev) | SQLite   | [datasets/bird](./datasets/bird/README.md)                             |
| [BEAVER](https://beaverbench.github.io/)              | 100-question `dw` sample    | MySQL    | [datasets/beaverbench](./datasets/beaverbench/README.md)               |
| [FDABench-Lite](https://github.com/fdabench/FDAbench) | 169                         | SQLite   | [datasets/fdabench](./datasets/fdabench/README.md)                     |
| Wide World Importers                                  | 41                          | Postgres | [datasets/wideworldimporters](./datasets/wideworldimporters/README.md) |
| BIRD-Interact                                         | ~300 (Lite), 600 (Full)     | Postgres | [scripts/bird_interact](./scripts/bird_interact/README.md)             |

BIRD-Interact is multi-turn: the agent asks clarifying questions, then answers a
follow-up. It runs Auto-Ontology alone on its own c-Interact and a-Interact
harnesses, on query tasks.

## Quickstart

This runs the full comparison on Wide World Importers, Microsoft's sample sales
database, which ships with this repository.

You need:

- [uv](https://docs.astral.sh/uv/), which installs the required Python 3.13
- A Postgres server with pgvector for Auto-Ontology's catalog. Auto-Ontology's
  Docker Compose provides one: in an
  [auto-ontology](https://github.com/NVIDIA/auto-ontology) checkout, run
  `cp .env.example .env && docker compose up -d postgres`.
- A model API key (see [Models](#models))

1. Install and configure:

   ```bash
   git clone https://github.com/NVIDIA/auto-ontology-eval.git && cd auto-ontology-eval
   uv sync
   cp .env.example .env
   ```

   In `.env`, set `DEFAULT_MODELS_API_KEY` and point `CONNECTION_STRINGS` at the
   sample database. The `POSTGRES_*` defaults match Auto-Ontology's Docker
   Compose.

   ```dotenv
   DEFAULT_MODELS_API_KEY=<your key>
   CONNECTION_STRINGS=postgresql://postgres:change-me@localhost:5432/wideworldimporters
   ```

   With an `nvapi-` key, also set the control arm's model as shown in
   [Models](#models).

2. Load the sample database into Postgres:

   ```bash
   uv run python scripts/seed/seed_wwi.py --database-name wideworldimporters --drop
   ```

3. Ingest, compile, and run both arms:

   ```bash
   uv run python main.py --database-name wideworldimporters
   ```

   The run prints accuracy for each arm, the difference between them, and the
   questions that flipped.

## Models

Point both arms at the same model, so the difference measures what Auto-Ontology
adds. The treatment arm reads Auto-Ontology's model settings; the control arm
reads `GYM_MODEL` and `GYM_MODEL_URL`.

| Variables                                     | Used by           | Purpose                                         |
| --------------------------------------------- | ----------------- | ----------------------------------------------- |
| `DEFAULT_MODELS_API_KEY`                      | Both arms         | Shared API key                                  |
| `DEFAULT_MODELS_ENDPOINT`, `_MODEL`           | Treatment         | Default endpoint and model for the agent        |
| `REASONING_*`, `NON_REASONING_*`              | Treatment         | Per-role overrides for the agent                |
| `GYM_MODEL`, `GYM_MODEL_URL`                  | Control           | The schema-only model and its endpoint          |
| `EMBED_*`                                     | Ingest, treatment | Embedding model; keep it the same for both      |

With an `nvapi-` key from [build.nvidia.com](https://build.nvidia.com),
Auto-Ontology defaults to Nemotron models on `integrate.api.nvidia.com`. Set the
control arm to match:

```dotenv
GYM_MODEL=nvidia/nemotron-3-nano-30b-a3b
GYM_MODEL_URL=https://integrate.api.nvidia.com/v1
```

The maintainers test with an NVIDIA-internal `sk-` key on
`inference-api.nvidia.com`, where both arms default to
`azure/anthropic/claude-opus-5-5` and embeddings to
`nvidia/nvidia/nemotron-3.5-embed-1b-nvfp4`. Use models that support structured
output. [`.env.example`](./.env.example) documents every variable.

## Running benchmarks

`main.py --database-name <dataset>` runs the whole pipeline. Common flags:

| Flag                                        | Effect                                                                       |
| ------------------------------------------- | ---------------------------------------------------------------------------- |
| `--skip-ingest`, `--skip-semantic`          | Reuse an ingested and compiled database                                      |
| `--skip-eval`                               | Ingest and compile only                                                      |
| `--skip-migrate`                            | Leave the catalog schema alone, for a Postgres shared with a live deployment |
| `--limit N`, `--start-index`, `--end-index` | Run a subset of questions                                                    |
| `--eval-workers N`                          | Questions in flight per arm (default 1)                                      |
| `--run-dir DIR`                             | Keep a run's output; `runs/` is overwritten each time                        |
| `--reuse-control`                           | Reuse a matching control run and rerun only the treatment                    |
| `--no-evidence`                             | Drop BEAVER's domain hints, as its leaderboard's Setting 0 does              |

A ten-question smoke run on an already-compiled BIRD:

```bash
uv run python main.py --database-name bird \
  --skip-ingest --skip-semantic --limit 10 --eval-workers 3
```

Size `--eval-workers` against the model endpoint's rate limit. Ten workers ran
cleanly on the maintainers' `sk-` key.

### Reading the results

Each arm writes its rollouts under `runs/control/` and `runs/treatment/` (or
`--run-dir`), next to two files from NeMo Gym:

- `*_aggregate_metrics.json`: accuracy overall and by difficulty, plus `health/*`
  counters
- `*_failures.jsonl`: questions that failed to run

Check the health counters before quoting a number. A rate-limited run scores
near zero and looks the same as a weak model, so the comparison withholds its
headline number when more than 5% of questions got no model output, or while
any question is missing a result.

To drive the stages by hand (build task files, run one arm, compare), see
[resources_servers/README.md](./resources_servers/README.md).

### Extra prompts

Two optional inputs for BIRD give the treatment arm question-specific help:
SQL exemplars from similar training questions, and value anchors that say which
column holds a phrase from the question. The control arm has no equivalent, so
a difference measured with them covers those inputs as well as the ontology.
They're off unless you pass `--sql-examples-csv` or `--value-anchors-csv`; see
[docs/extra-prompts.md](./docs/extra-prompts.md).

## Evaluate your own data

1. Create `datasets/<name>/evaluation.json`: a list of questions, each with its
   database ID and gold SQL. Add `metadata.json` and `custom_analyses.json` to
   enrich ingestion. The formats are in [docs/datasets.md](./docs/datasets.md).
2. Register the dataset in `DATASETS` in
   [`ontology_sql_eval/gym/tasks.py`](./ontology_sql_eval/gym/tasks.py) with its
   SQL dialect and the folder that holds its databases or DDL. The control arm
   builds its schema dump from there.
3. Point `CONNECTION_STRINGS` at the source database and run
   `uv run python main.py --database-name <name>`.

## Troubleshooting

- **Every task reports `gold_execution_error` on BIRD.** The databases belong
  under `datasets/bird/dev/<db_id>/`. Re-seed with
  `uv run python scripts/seed/seed_bird.py`.
- **The treatment arm misses an Auto-Ontology update.** Server environments are
  reused between runs. Rebuild them with
  `GYM_REBUILD_VENV=1 scripts/run_gym_arm.sh auto_ontology_sql /tmp/probe.jsonl`.
- **A run stopped partway.** Rerun `scripts/run_gym_arm.sh` for that arm with
  `--resume` to finish the missing questions; see
  [resources_servers/README.md](./resources_servers/README.md#running-an-arm).

## Reference

| Topic                                     | Doc                                                                              |
| ----------------------------------------- | -------------------------------------------------------------------------------- |
| Benchmark arms, scoring rules, and stages | [resources_servers/README.md](./resources_servers/README.md)                     |
| Ingestion and enrichment                  | [ontology_sql_eval/ingestion/README.md](./ontology_sql_eval/ingestion/README.md) |
| BIRD-Interact                             | [scripts/bird_interact/README.md](./scripts/bird_interact/README.md)             |
| Dataset file formats                      | [docs/datasets.md](./docs/datasets.md)                                           |
| Extra prompts                             | [docs/extra-prompts.md](./docs/extra-prompts.md)                                 |
| Repository layout and development         | [docs/development.md](./docs/development.md)                                     |

## Contributing

The Auto-Ontology team maintains this repository and doesn't accept external
pull requests. Bug reports are welcome as issues.

## License and security

This project is licensed under the [Apache License 2.0](./LICENSE). Third-party
dependency licenses are listed in
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

Third-party content included in this repository is MIT licensed and documented
in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md#included-third-party-content):

- [`datasets/wideworldimporters/`](./datasets/wideworldimporters/): a Postgres
  port of Microsoft's WideWorldImporters sample database
  ([microsoft/sql-server-samples](https://github.com/microsoft/sql-server-samples),
  Copyright (c) Microsoft Corporation). Its upstream notice is in
  [datasets/wideworldimporters/LICENSE](./datasets/wideworldimporters/LICENSE).
- [`tests/gym/beaver_reference.py`](./tests/gym/beaver_reference.py): a port of
  BEAVER's official execution-accuracy rule
  ([beaverbench/beaver](https://github.com/beaverbench/beaver), Copyright (c)
  2024 MIT Data Systems Group).

To report a security vulnerability, follow [SECURITY.md](./SECURITY.md).
