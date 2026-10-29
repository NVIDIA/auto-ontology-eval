# BEAVER (BeaverBench)

[← Back to main README](../../README.md)

> **Disclaimer:** Running this code will automatically download data from
> https://huggingface.co/datasets/beaverbench/beaver-query and
> https://huggingface.co/datasets/beaverbench/beaver-table. Before you run the
> code, please confirm the content of the dataset and licensing is appropriate
> for your intended use.

[BEAVER](https://beaverbench.github.io/) is an enterprise text-to-SQL benchmark
sourced from private data warehouses. This repo maps it onto the same
ingest → semantic compile → NeMo Gym benchmark pipeline used for BIRD and
FDABench.

Public HuggingFace data (gated — accept the terms, then `hf auth login`):

- Questions: [`beaverbench/beaver-query`](https://huggingface.co/datasets/beaverbench/beaver-query)
- Tables / MySQL dumps: [`beaverbench/beaver-table`](https://huggingface.co/datasets/beaverbench/beaver-table)

Splits: `dw`, `nova`, `neutron`, `dw_real` (`dw_real` questions run against the
`dw` database). Gold SQL is MySQL dialect; Auto Ontology's MySQL connector is on `main`
(`auto_ontology/connectors/mysql.py`, PR #195).

## Download / seed

```bash
# Default: dw split, 100-question sample (seed 77), download dumps, import MySQL
uv run python scripts/seed/seed_beaverbench.py --import-mysql
```

This writes:

```
datasets/beaverbench/evaluation.json
datasets/beaverbench/<db_id>/metadata.json
datasets/beaverbench/dumps/<db_id>.sql
```

and (with `--import-mysql`) starts a Docker MySQL container `beaver-mysql` on
port 3306, imports the dumps, and sets `CONNECTION_STRINGS` in `.env`.

Useful flags:

| Flag | Meaning |
|---|---|
| `--domains dw nova neutron dw_real` | Which query splits to include (default: `dw`) |
| `--sample N` | Sample size across selected domains (`0` = all; default `100`) |
| `--sample-seed 77` | RNG seed for sampling (matches official BEAVER download script) |
| `--tasks-only` | Write `evaluation.json` (+ metadata) only |
| `--no-metadata` | Skip `metadata.json` from beaver-table |
| `--import-mysql` | Docker MySQL import + write `CONNECTION_STRINGS` |
| `--force` | Re-download / re-extract dumps |
| `--no-write-env` | Print `CONNECTION_STRINGS` instead of writing `.env` |

`nova.sql` is ~2 GB; prefer starting with `--domains dw` (default).

## Configure

With `--import-mysql`, the seed script writes something like:

```bash
CONNECTION_STRINGS=mysql://root:beaver-benchmark@localhost:3306/dw
```

Also point pgvector at the Auto Ontology Docker Postgres host port (typically `5434`):

```bash
POSTGRES_PORT=5434
POSTGRES_USER=postgres     # match the deployed Auto Ontology service
POSTGRES_PASSWORD=...      # match the deployed Auto Ontology service
POSTGRES_DATABASE=auto_ontology
```

Models come from NVIDIA inference. The tested setup is a single `sk-` key with
no model pinned, so every role uses its built-in default (see
[Configuration](../../README.md#configuration) and
[`.env.example`](../../.env.example)):

```bash
DEFAULT_MODELS_API_KEY=sk-...
```

## Run the full pipeline

The Postgres store (catalog + pgvector) must be up; follow the
[Auto Ontology deployment guide](https://github.com/NVIDIA/auto-ontology).

Then from this repo:

```bash
uv run python main.py --database-name beaverbench
```

That ingests every database in `CONNECTION_STRINGS`, compiles the semantic
layer, then runs the NeMo Gym benchmark over `datasets/beaverbench/evaluation.json`:
the schema-only control and the ontology-grounded arm, with rollouts in
`runs/control/` and `runs/treatment/` and the delta printed at the end. Re-run
without re-ingesting, and smoke-test on a few questions first:

```bash
uv run python main.py --database-name beaverbench \
    --skip-ingest --skip-semantic --limit 10 --eval-workers 3
```

### Long runs

A full run can take a day or more. Ingest and compile with `main.py`, then build
the tasks and run each arm with `scripts/run_gym_arm.sh`, so an arm can be
resumed:

```bash
uv run python main.py --database-name beaverbench --skip-eval

# Gym caches metrics beside the task file; a rebuild with a different question
# count must clear them first.
rm -f resources_servers/{schema_only_sql,auto_ontology_sql}/data/tasks_{metrics,metrics_conflict}.json \
      resources_servers/{schema_only_sql,auto_ontology_sql}/data/tasks_prepare.jsonl
uv run python scripts/build_gym_tasks.py --dataset beaverbench --arm schema_only
uv run python scripts/build_gym_tasks.py --dataset beaverbench --arm auto_ontology

GYM_CONCURRENCY=5 scripts/run_gym_arm.sh schema_only_sql runs/control/beaverbench.jsonl
GYM_CONCURRENCY=5 scripts/run_gym_arm.sh auto_ontology_sql runs/treatment/beaverbench.jsonl \
    --max-output-tokens 16 \
    +auto_ontology_sql_resources_server.resources_servers.auto_ontology_sql.max_concurrency=5

uv run python -m ontology_sql_eval.gym.compare \
    --control runs/control/beaverbench.jsonl \
    --treatment runs/treatment/beaverbench.jsonl
```

The ontology arm ignores the policy model's answer, so `--max-output-tokens 16`
keeps that call from competing for the rate limit, and `max_concurrency` lifts
the arm's own limit, which is 3 by default.

- A question whose run fails, for example on a burst of rate limits, is
  recorded in `runs/<arm>/beaverbench_failures.jsonl` and the run goes on.
- To finish it, run the same command again with `--resume`. Questions already
  done are kept; failed and unfinished ones are retried, up to
  `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS` (default 3) attempts each. Without `--resume`,
  the command starts over and deletes the arm's earlier results.
- Until then Gym's metrics count each failed question as a miss, and the
  comparison refuses to report while any question has no result. A published
  number should come from a run with none left.

After the Auto Ontology pin changes (a pull and `uv sync`), set
`GYM_REBUILD_VENV=1` for the next run of the ontology arm. The script rebuilds a
server venv on its own only when nemo-gym, ray or openai differ, so it would
otherwise keep running the old Auto Ontology.

## Scoring

Both arms are scored with BEAVER's official execution accuracy (`ex_acc`), ported
into the Gym verifier so the numbers are comparable to the BEAVER leaderboard:
values stringified and stripped, deduplicated row sets, a 10-second query
timeout, and every query that yields no rows (an error, a timeout, a missing
prediction, a failed gold query) counted as an empty result. Accuracy is
reported both with and without empty-gold questions
(`nonempty_gold/pass@1[...]`). See
[`resources_servers/README.md`](../../resources_servers/README.md) for details.

No LLM judge is involved.

The seed script fills each question's `evidence` from BEAVER's domain
knowledge, and both arms see it by default. The leaderboard's Setting 0 gives no
hints, so pass `--no-evidence` to `main.py` or `scripts/build_gym_tasks.py` for
a number comparable with it. Only `beaverbench` takes the flag.

## Notes

- Only `ex_acc` is ported, not the official BEAVER coarse/fine-grained
  evaluators from [`beaverbench/beaver`](https://github.com/beaverbench/beaver).
- Source dumps are downloaded at runtime and are **not** vendored in git.
- Requires HuggingFace access to the gated beaverbench datasets.
