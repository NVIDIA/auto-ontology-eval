# BEAVER (BeaverBench)

[← Back to main README](../../README.md)

[BEAVER](https://beaverbench.github.io/) is an enterprise text-to-SQL benchmark
sourced from private data warehouses. This repo maps it onto the same
ingest → semantic compile → retrieval eval → LLM judge pipeline used for BIRD
and FDABench.

Public HuggingFace data (gated — accept the terms, then `hf auth login`):

- Questions: [`beaverbench/beaver-query`](https://huggingface.co/datasets/beaverbench/beaver-query)
- Tables / MySQL dumps: [`beaverbench/beaver-table`](https://huggingface.co/datasets/beaverbench/beaver-table)

Splits: `dw`, `nova`, `neutron`, `dw_real` (`dw_real` questions run against the
`dw` database). Gold SQL is MySQL dialect; GSF's MySQL connector is on `main`
(`gsf/connectors/mysql.py`, PR #195).

## Download / seed

```bash
# Default: dw split, 100-question sample (seed 77), download dumps, import MySQL
uv run python scripts/seed_beaverbench.py --import-mysql
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

Also point pgvector at the GSF docker Postgres host port (typically `5434`):

```bash
POSTGRES_PORT=5434
POSTGRES_USER=gsf          # match GSF/.env
POSTGRES_PASSWORD=...      # match GSF/.env
POSTGRES_DATABASE=gsf
```

Agent / judge / embed models should use NVIDIA inference (see
[`.env.example`](../../.env.example)). Example:

```bash
DEFAULT_MODELS_API_KEY=nvapi-...
DEFAULT_MODELS_ENDPOINT=https://integrate.api.nvidia.com/v1
DEFAULT_MODELS_MODEL=nvidia/nemotron-3-nano-30b-a3b
```

## Run the full pipeline

Stores (Neo4j + pgvector) must be up — easiest via GSF:

```bash
cd ../GSF && docker compose up -d
```

Then from this repo:

```bash
PYTHONPATH=../GSF uv run python main.py --database-name beaverbench
```

That ingests every database in `CONNECTION_STRINGS`, compiles the semantic
layer, runs the agent against `datasets/beaverbench/evaluation.json`
(→ `input/beaverbench_<model>.csv`), then LLM-judges the result
(→ `output/beaverbench_<model>_scores.csv`).

## Notes

- Scoring uses this repo's exec-match + LLM judge path, not the official BEAVER
  coarse/fine-grained evaluators from [`beaverbench/beaver`](https://github.com/beaverbench/beaver).
- Source dumps are downloaded at runtime and are **not** vendored in git.
- Requires HuggingFace access to the gated beaverbench datasets.
