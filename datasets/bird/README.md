# BIRD

[← Back to main README](../../README.md)

> **Disclaimer:** Running this code will automatically download data from
> https://bird-bench.oss-cn-beijing.aliyuncs.com (the `dev` split), the BIRD
> maintainers' direct
> [`train.json`](https://github.com/user-attachments/files/15820826/train.json)
> attachment, and, for the `mini-dev` split, from the BIRD authors' Google Drive
> (https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view).
> Before you run the code, please confirm the content of the dataset and
> licensing is appropriate for your intended use.

[BIRD](https://bird-bench.github.io/) is a large-scale cross-domain Text-to-SQL
benchmark. [scripts/seed_bird.py](../../scripts/seed_bird.py) installs the full
official **Dev** split by default (1,534 questions over 11 SQLite databases) and
its Train question corpus, or can install the cheaper **Mini-Dev** subset alone.
The repo additionally ships the official EX/VES scoring script ported from the
BIRD repo. Unlike WideWorldImporters, BIRD's source DBs are SQLite files (no
Postgres seeding step) and each question carries its own `db_id`, routing to the
matching connector at eval time.

> **Nothing you want to keep belongs in this folder.** Everything under
> `datasets/bird/` except this README and `.gitkeep` is gitignored and replaced
> wholesale by the next download. Our own column descriptions and custom
> analyses therefore live outside it, in `annotations/bird/`, and are picked up
> from there automatically — see
> [Publishing our descriptions and analyses](../../ontology_sql_eval/ingestion/README.md#publishing-our-descriptions-and-analyses).

## Download

```bash
uv run python scripts/seed_bird.py                     # Dev (default): 1,534 Qs, 11 DBs
uv run python scripts/seed_bird.py --splits mini-dev    # 500-Q subset of the same 11 DBs
```

| Split      | Questions | Databases installed | Role                                                                                       |
| ---------- | --------: | ------------------- | ------------------------------------------------------------------------------------------ |
| `dev`      |     1,534 | 11                  | Default evaluation split; also downloads the Train question corpus.                        |
| `mini-dev` |       500 | 11 (the same DBs)   | Cheap/fast subset for day-to-day development; also becomes `evaluation.json`.              |
| Train      |     9,428 | none                | Stored as `train/train.json` when Dev is selected; the pool the few-shot exemplars are drawn from. |

Train contributes no connection strings and installs no databases. Its database
set is disjoint from Dev, so the seeder downloads the official 4.3 MB question
JSON directly and retains only complete question, evidence, and SQL rows. It
does not download the nearly 1 GB Train database archive.

Only the SQLite dialect is kept (the MySQL/PostgreSQL question JSONs and the
`*_gold.sql` / `*_tables.json` files are ignored). Options: `--splits` (one or
more of `mini-dev`/`dev`, default `dev`), `--force` (re-download and
overwrite), `--keep-archive` (keep downloaded split archives and the Train JSON
cache), `--url` (use a different/local evaluation-split zip — only valid with a
single `--splits` value), `--dest`
(default `datasets/bird/`), `--no-write-env` (see [Configure](#configure)),
`--log-level`.

> Passing two *evaluation* splits (`mini-dev dev`) merges both into one
> `evaluation.json`, with each split's `question_id` offset by 100,000 per split
> so ids never collide. There is little reason to do it — Mini-Dev's questions
> are a subset of Dev's.

## Folder layout

```
datasets/bird/
  evaluation.json                          # Mini-Dev / Dev questions for the whole dataset
  dev/<db_id>/<db_id>.sqlite               # one evaluation database per db_id
  dev/<db_id>/database_description/*.csv   # BIRD's own column annotations
  dev/<db_id>/metadata.json                # derived from those CSVs (ingestion enrichment)
  train/train.json                         # Train corpus, included with Dev
  subsets/<name>.json                      # optional hand-picked question slices (see below)

annotations/bird/                          # tracked, outside the download
  custom_analyses/<db_id>.json             # our analyses, one file per db_id
  semantic_descriptions.csv                # our column descriptions, all 11 databases

prompt_inputs/bird/                         # precomputed inputs supplied during eval
  predicted_structural_exemplars.csv        # question-keyed Train SQL examples
  value_anchors.csv                         # question-keyed verified value anchors
```

`metadata.json` is generated per database from BIRD's `database_description`
CSVs, in the shape [`enrich_graph.apply_metadata`](../../ontology_sql_eval/ingestion/enrich_graph.py)
consumes, so column meanings and value descriptions reach the Postgres catalog
and from there the text-to-SQL prompt. A `value_description` cell that reads
exactly `not useful` is dropped (it's an annotator note about an opaque column)
but the column entry is kept, since some of those columns are join keys many
gold queries need. The semantic compile can describe columns BIRD leaves
undocumented after profiling their values.

## Configure

By default the download script writes a ready-to-use `CONNECTION_STRINGS` line
into your `.env`, replacing any existing `CONNECTION_STRINGS` entry:

```bash
CONNECTION_STRINGS=sqlite:///<abs>/datasets/bird/dev/california_schools/california_schools.sqlite,sqlite:///<abs>/datasets/bird/dev/card_games/card_games.sqlite,...
```

One comma-separated entry per evaluation `db_id` (11 for `mini-dev`/`dev`).
Pass `--no-write-env` to skip the rewrite and just log the value for manual
copy-paste.

## Run the full pipeline

```bash
uv run python main.py --database-name bird
```

This ingests every database in `CONNECTION_STRINGS`, compiles the semantic
layer, runs the agent against `datasets/bird/evaluation.json`
(→ `input/bird_<model>.csv`), then LLM-judges the result
(→ `output/bird_<model>_scores.csv`). Individual stages can be skipped with
`--skip-ingest`, `--skip-semantic`, `--skip-eval`, `--skip-judge`. The semantic
stage automatically replaces BIRD's shipped descriptions with the tracked set
under `annotations/bird/` when it is present.

### Manual (per-stage) equivalent

```bash
# 1. Ingest all 11 DBs, then compile semantics.
uv run python -m ontology_sql_eval.ingestion.ingest --benchmark-name bird
uv run python -m ontology_sql_eval.ingestion.semantic --benchmark-name bird

# 2. Run the agent against the eval set -> input/bird_<model>.csv.
# This matches the "Eval" configuration in .vscode/launch.json.
uv run python -m ontology_sql_eval.retrieval.eval_chatbot \
  --database-name bird \
  --workers 10 \
  --sql-examples prompt_inputs/bird/predicted_structural_exemplars.csv \
  --sql-examples-k 5 \
  --value-anchors prompt_inputs/bird/value_anchors.csv

# 3. Re-score every CSV in input/ with the LLM judge -> output/<name>_scores.csv
uv run ontology-sql-eval
```

Both ingestion commands walk every entry in `CONNECTION_STRINGS`, so all 11
databases are handled in one invocation each.

The eval inputs shown above are optional:

- `--sql-examples` loads question-keyed structural examples generated from the
  BIRD Train corpus; `--sql-examples-k 5` supplies at most five to the agent.
- `--value-anchors` loads question-keyed, precomputed database-value matches.
- `--workers 10` evaluates up to ten questions concurrently.

These files provide prompt context only. They do not replace
`datasets/bird/evaluation.json`, and the eval questions' gold SQL is not used to
select examples or anchors.

### Evaluating a slice

`eval_chatbot` takes any file in `evaluation.json` format, so a hand-picked
subset of questions can be run on its own — one database, one difficulty, one
reproduction. Keep those files in `subsets/` and name the output after the
subset so a probe never overwrites a full run:

```bash
uv run python -m ontology_sql_eval.retrieval.eval_chatbot \
    --input datasets/bird/subsets/formula_1.json \
    --output input/formula_1_<model>.csv
```

Scoring a subset needs no extra flags: `--dataset-name bird` still resolves the
right databases, because the judge maps each `question_id` back to its `db_id`
through the full `evaluation.json`.

## Official scoring (EX + VES)

In addition to the generic LLM judge, BIRD ships its own deterministic
**Execution Accuracy (EX)** and **Valid Efficiency Score (VES)** metrics,
ported from the official BIRD evaluation scripts:

```bash
uv run python -m ontology_sql_eval.judge.bird \
  --input input/bird_<model>.csv \
  --dataset-name bird \
  --skip-ves            # omit to also compute VES (slower — times EX-passing rows)
```

`--dataset-name` resolves both `evaluation.json` (for the `question_id → db_id`
map) and the `dev/` SQLite root; override either with `--evaluation-json` /
`--db-root`. Results are written to `output/<input_stem>_bird_scores.csv`.

See [ontology_sql_eval/judge/bird.py](../../ontology_sql_eval/judge/bird.py)
for the remaining flags (`--num-cpus`, `--meta-time-out`, `--iterate-num`,
`--no-output-csv`, `--debug`).

This is a separate, additional scoring step — it does not replace the LLM judge
stage in `main.py`.
