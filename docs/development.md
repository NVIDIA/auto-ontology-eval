# Development

## Checks

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The type checker uses the project's `.venv`, as configured in
[pyrightconfig.json](../pyrightconfig.json). VS Code launch configurations in
[.vscode/launch.json](../.vscode/launch.json) cover the full pipeline, a
one-question BIRD benchmark, ingestion, semantic compilation, and seeding each
dataset; they load `.env` automatically.

## Repository layout

```
main.py                     end-to-end pipeline entry point (ingest -> compile -> benchmark)
ontology_sql_eval/          single namespace package
  ingestion/                Auto-Ontology-backed ingestion pipeline
    README.md               ingestion workflow guide
    ingest.py               source DB -> pgvector ingest (calls enrich_graph)
    semantic.py             compile the semantic layer over the ingested graph
    enrich_graph.py         graph metadata + custom-analysis enrichment
  gym/                      the benchmark logic shared by both arms and main.py
    tasks.py                build Gym task files from datasets/<name>/evaluation.json
    ddl.py                  schema dumps for the schema-only arm's prompt
    exec_match.py           execution-match verifier (BIRD EX, BEAVER ex_acc)
    model_routing.py        routes Auto-Ontology's model calls through Gym's model servers
    compare.py              health-gated delta report between the two arms
  bird_interact/            BIRD-Interact adapter server (see scripts/bird_interact/README.md)
  env.py                    .env loading
  extra_prompts.py          loaders for the extra-prompt CSVs
resources_servers/          the two Gym arms: schema_only_sql (control), auto_ontology_sql
scripts/
  build_gym_tasks.py        build one arm's Gym task file
  run_gym_arm.sh            run one arm through NeMo Gym
  migrate_catalog.py        migrate the catalog DB to the installed Auto-Ontology's schema
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
