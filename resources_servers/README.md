# NeMo Gym environments

Two resources servers that answer one question: **how much does Auto Ontology's ontology
grounding add over a competent LLM that sees only the schema?**

| Server | Arm | What the model gets |
|---|---|---|
| `schema_only_sql` | control | The question plus a raw schema dump, one shot, no tools |
| `auto_ontology_sql` | treatment | Nothing directly — Auto Ontology's agent retrieves its own grounding |

Both are graded by the same verifier (`ontology_sql_eval/gym/exec_match.py`) against
the same databases, so any difference in the reported number comes from the SQL,
never from the grader.

The comparison rule is per dataset, because benchmarks define execution accuracy
differently and we want numbers that are comparable to each one's own leaderboard:

| Dataset | Rule |
|---|---|
| everything else | BIRD: `set(gold) == set(pred)` on raw rows |
| `beaverbench` | BEAVER `ex_acc`: values stringified and stripped first |

They genuinely disagree. BEAVER treats `1` and `"1"` as equal and strips
whitespace, but **not** `Decimal("150.250")` and `Decimal("150.25")` -- which on a
MySQL corpus is a common case, not an exotic one. `tests/test_gym_beaver_match.py`
holds our implementation against the official evaluator in
`ontology_sql_eval/judge/beaver.py`.

BEAVER also differs in how it runs and reports, and both arms follow it there
too:

- **10-second query timeout** for `beaverbench`, whatever
  `sql_execution_timeout_s` is set to (`QUERY_TIMEOUTS` in `exec_match.py`). A
  query that is slower than that is counted as an empty result, as it is
  officially.
- **Accuracy without empty-gold questions.** Every rollout records
  `gold_result` (`empty_gold` / `nonempty_gold`), and the metrics carry
  `nonempty_gold/pass@1[avg-of-k]/accuracy` beside the overall number;
  `scripts/compare_arms.py` prints a matching `nonempty gold` row. An empty
  gold result is matched by any prediction that returns nothing -- including
  one that errored -- so this is the more honest of the two.

- **Every query that produces no rows is an empty result** -- an error, a
  timeout, or no prediction at all (an empty model answer, an agent crash), on
  the gold side as well as the prediction side (`ERRORS_AS_EMPTY` in
  `exec_match.py`). So a missing prediction against empty gold scores 1.0, and a
  task whose gold query fails stays in the denominator instead of being
  excluded; its rollouts carry `gold_result: failed_gold` and are counted in
  `health/gold_failure_scored_as_empty_count`. A missing prediction still keeps
  its `no_model_output` / `unknown_error` failure code, so the health gate sees
  an outage even where it scored.

Every other dataset keeps the stricter handling: a failed prediction is a miss,
and a failed gold query drops the task (`health/excluded_unusable_gold_count`).

Neither rule involves an LLM. The LLM judge (`judge/scorer.py`) is a separate
tool and is not used by either arm.

## Layout

Logic lives in `ontology_sql_eval/gym/` because CI only lints `./ontology_sql_eval`;
the servers here stay thin so Gym's workspace layout is satisfied without putting
real code outside the linted package.

```
ontology_sql_eval/gym/
  ddl.py         schema dumping (SQLite introspection, Postgres DDL files)
  tasks.py       evaluation.json -> Gym task JSONL
  exec_match.py  executors + BIRD set-equality + failure codes
resources_servers/
  schema_only_sql/{app.py,configs/,data/}
  auto_ontology_sql/{app.py,configs/,data/}
```

## Building task data

```bash
python scripts/build_gym_tasks.py --dataset bird --arm schema_only
python scripts/build_gym_tasks.py --dataset bird --arm auto_ontology
```

Datasets: `bird` (1,534q), `fdabench` (169q), `wideworldimporters` (41q,
Postgres), `beaverbench` (MySQL). Output is deterministic, so regenerating is
safe. `--limit N` takes a prefix; `--start-index` / `--end-index` slice, so a
long run can be sharded or resumed.

Each build overwrites `data/tasks.jsonl`, which is the file the server config
points at. That is deliberate: a per-dataset filename meant the config had to
name one dataset up front, so every *other* dataset silently ran that one and
wrote the results under its own name. `main.py` additionally asserts the built
file's `dataset` matches what was requested.

The schema dump *is* the control arm's independent variable. Freeze it once a full
run starts — changing it silently changes what "schema-only" means.

## Running an arm

```bash
scripts/run_gym_arm.sh schema_only_sql  runs/control/bird.jsonl
scripts/run_gym_arm.sh auto_ontology_sql runs/treatment/bird.jsonl --max-output-tokens 16
```

Those two commands drive one arm each, against whatever `tasks.jsonl` happens to
be on disk. **For a real comparison use `main.py` instead** -- it rebuilds the
task files, checks they hold the dataset you asked for, runs both arms and the
comparison, and stops on the first failure:

```bash
uv run python main.py --database-name bird --skip-ingest --skip-semantic \
  --eval-workers 10 --run-dir runs/full_$(date +%Y%m%d-%H%M%S)
```

Reach for `run_gym_arm.sh` directly when you are driving a single arm on
purpose and already know the task file is right.

`run_gym_arm.sh` passes the model key as `+policy_api_key=${oc.env:DEFAULT_MODELS_API_KEY}`,
which keeps it out of argv, and reuses existing server venvs with
`+skip_venv_if_present=true`. Gym otherwise rebuilds a server venv from
`requirements.txt` on every run, which costs minutes and can trip the
metadata-timestamp race below. **Set `GYM_REBUILD_VENV=1` after editing a server's
`requirements.txt` or the root `pyproject.toml`'s `[tool.uv]` overrides, and once
after pulling a change to either.** A venv that already exists is reused as-is,
so otherwise the new pins silently never take effect. The one exception is
checked for you: when a server venv's nemo-gym or Ray differs from `.venv`'s, the
script rebuilds without being asked, since either mismatch breaks the run.

Gym builds each server venv with `uv pip install` run from inside this repo, so uv
also applies the root `pyproject.toml`'s `[tool.uv] override-dependencies` there.
That is deliberate: it is how the server venvs get the CVE floors without a second
list to keep in sync. It is also why `ray` must never be added to that list: an
override replaces Gym's exact `ray[default]==<head version>` pin, the server venv
resolves the newest Ray, and it fails to join the head's cluster with
`Version mismatch`.

Both arms run **in-process**: `auto_ontology_sql` imports Auto Ontology and drives its graph
directly, so no Auto Ontology server is required. `--max-output-tokens 16` on the Auto Ontology arm is
deliberate — that arm ignores the policy model's output, so the call is capped to
stop it competing for rate limit with Auto Ontology's own ~19 calls per question.

Useful environment variables:

| Variable | Default | Notes |
|---|---|---|
| `GYM_CONCURRENCY` | 3 | Size against the **model endpoint's rate limit**, not CPU. 10 is measured-safe on the `sk-` key — two full 1,534-question ontology runs at 10 produced zero rate-limit failures. The conservative default stays 3 because the ceiling belongs to the key, not the code. Setting this alone is not enough for the ontology arm: its server gates agent runs behind a semaphore of its own (`max_concurrency`, also 3), so the arm stays throttled unless both move. `main.py --eval-workers` sets both. |
| `GYM_API_KEY` | `.env`'s `DEFAULT_MODELS_API_KEY` | Overrides the key for *both* the policy model and Auto Ontology's own calls |
| `GYM_MODEL` / `GYM_MODEL_URL` | bedrock-claude-opus-4-8 @ inference-api | |

`GYM_API_KEY` works by exporting `DEFAULT_MODELS_API_KEY`, which is what both
consumers read: the Auto Ontology arm authenticates its own in-process calls
with that variable, and the policy model's key is an interpolation of it (see
[The API key never goes on the command line](#the-api-key-never-goes-on-the-command-line)).
Pointing only the policy model at a new key would leave the actual workload --
roughly 19 of every 20 calls -- on the old one.

Give each arm its own output **directory**. Gym writes `preprocessed_datasets/`
beside the output file, and a second arm's collation aborts on the first arm's
leftover metrics ("Found conflicting aggregate metrics").

Separate output directories do not cover every case of that error, though. Gym
also caches `tasks_metrics.json` and `tasks_prepare.jsonl` beside the **task**
file, under `resources_servers/<arm>/data/`, so rebuilding `tasks.jsonl` at a
different question count aborts the next run of the *same* arm at collation --
after the compute is spent. `main.py` clears them before each build, which is
another reason to prefer it over driving the arms by hand.

## Environment

One venv holds both `nemo_gym` and Auto Ontology's dependencies, with no overrides
needed between them: Auto Ontology pins `openai==2.44.0` to match nemo-gym.

Auto Ontology's runtime dependencies are not redeclared in `pyproject.toml` — they
arrive through the `auto-ontology-server` Git dependency. Add a package there only
when this project needs it on top of Auto Ontology.

nemo-gym requires Python **>=3.13.14**, which is why `requires-python` was
raised from 3.12.

Auto Ontology is installed from the parent project's pinned Git dependency.
The resource-server environment receives the same revision through its editable
installation of this project.

Four things about Gym's CLI that are easy to lose an hour to:

1. `--split` accepts only `train` / `validation` / `benchmark`, and `benchmark`
   requires a `prepare_script` and `prompt_config` we do not need. Datasets here
   are typed `validation`.
2. Each server needs its own `requirements.txt`. Without one, Gym silently resolves
   the server directory to its *installed* copy in site-packages.
3. That `requirements.txt` must list `nemo-gym` explicitly. Gym decides whether to
   install it by looking for a `pyproject.toml` two levels up, finds this repo's,
   and wrongly concludes it is an editable Gym checkout.
4. `--agent` takes the **fully-qualified instance name** (`<server>_simple_agent`),
   i.e. the top-level config key — not the inner `simple_agent`, which is the server
   *type*. nemo-gym 0.6.0+ validates this eagerly; 0.4.0 did not validate it at all
   and silently ignored a wrong value, so runs could appear to work with a bad name.

## How this relates to the repo's other scorers

Three scorers live in this repo. They are not redundant, and only one is used by
the benchmark.

| Scorer | What it measures | Used by |
|---|---|---|
| **Gym verifier** (`ontology_sql_eval/gym/exec_match.py`) | Execution match, binary reward | both benchmark arms |
| **BIRD official** (`ontology_sql_eval/judge/bird.py`) | EX **and VES** | standalone CLI, over an eval CSV |
| **LLM judge** (`ontology_sql_eval/judge/scorer.py`) | Logic / semantic rubric | `main.py`'s judge stage, over an eval CSV |

**The Gym verifier's EX is BIRD's official EX.** Both are
`set(gold_rows) == set(pred_rows)`: `judge/bird.py` ports it from DAMO-ConvAI,
and this one came from NeMo Gym's `bird_sql`, which mirrors the same evaluator.
They agree on row order, duplicate collapsing and extra columns. Running both
over the same BIRD predictions should give the same accuracy -- if it does not,
one of them has a bug.

**VES has no equivalent here.** Valid Efficiency Score is a *runtime ratio* over
EX-passing questions -- how fast the predicted query is versus gold -- which
`judge/bird.py` computes by re-running each passing query (`--iterate-num`,
default 100) with outlier removal. The benchmark measures correctness only. For
VES, run `judge/bird.py` over an eval CSV; the benchmark writes Gym rollouts,
not a CSV.

`judge/bird.py` also reports an `extra_col_nearmiss` diagnostic -- predictions
that would have matched but for extra selected columns. That is exactly the
failure mode `shorten_answer` and the control arm's matching projection rule
exist to prevent.

**The LLM judge is used by neither arm.** Scoring here is deterministic by
design, and NeMo Gym's own guidance is to prefer a deterministic verifier where
one exists. The judge stays available as a separate re-scoring pass over CSVs.

## Dataset coverage

| Dataset | Dialect | In the benchmark? |
|---|---|---|
| `bird` (1,534q) | SQLite | yes |
| `fdabench` (169q) | SQLite | yes |
| `wideworldimporters` (41q) | Postgres | yes |
| `beaverbench` | MySQL | yes — seed with `scripts/seed_beaverbench.py --import-mysql` |
| `bird_interact` | Postgres | **no, by design** |

`beaverbench` needs a `mysql://` entry in `CONNECTION_STRINGS` (the seeder writes
one). Each `db_id` is its own physical database, so DSNs are keyed by database
name rather than collapsed into one connection — `dw_real` questions deliberately
execute against the `dw` database. Its schema comes from the shipped mysqldump
with the INSERTs stripped; those dumps are hundreds of MB and inlining one into a
prompt would be absurd.

**`bird_interact` is deliberately absent.** It is a multi-turn benchmark: JSONL
rather than a JSON array, `sol_sql` lists instead of a single `SQL`, `follow_up`
turns, test cases, and a user-simulation loop driven by the c-Interact
orchestrator. Both arms here are single-shot by construction — the control gets
one prompt and the Auto Ontology arm runs one `stream_agent_response` to completion — so
representing it would mean a new multi-turn environment, not a registry entry.
Use `scripts/run_bird_interact_cinteract.py` for that benchmark.

## Keeping the arms comparable

Anything that changes what the model is told must land on **both** arms or
neither. The current pairing:

| Control (`schema_only_sql`) | Auto Ontology (`auto_ontology_sql`) |
|---|---|
| `SYSTEM_PROMPT` in `ontology_sql_eval/gym/tasks.py` carries "Return exactly the requested output fields and NO others" | `shorten_answer=True` appends the same rule to Auto Ontology's projection rules |

That instruction is not cosmetic: extra columns fail `set(gold) == set(pred)`
outright, so giving it to one arm only would measure projection guidance rather
than ontology grounding.

The Auto Ontology arm also has to carry `calculation_only=True` and
`validate_sql_values=True`, which are not comparability knobs but parity with
`ontology_sql_eval/retrieval/eval_chatbot.py` — the harness whose numbers this
arm is checked against. The first pins the intent to CALCULATION (every BIRD
question is answered by running SQL, so letting the classifier route elsewhere
only creates ways to lose); the second enables the `validate_sql_values` node,
which checks filter literals against the database. Omitting them is not visible
in the rollouts, only in the node trace, and cost about two points.

### Prompt inputs are off by default

`sql_examples_csv` and `value_anchors_csv` attach the question-keyed exemplars
and value anchors from `prompt_inputs/` to the agent's payload. They are
commented out in `configs/auto_ontology_sql.yaml` and must be asked for
explicitly, so the committed default is the comparable one:

```bash
uv run python main.py --database-name bird --skip-ingest --skip-semantic \
  --eval-workers 10 \
  --sql-examples-csv prompt_inputs/bird/predicted_structural_exemplars.csv \
  --sql-examples-k 5 \
  --value-anchors-csv prompt_inputs/bird/value_anchors.csv
```

`main.py` turns those into Hydra overrides on the ontology arm. Driving
`run_gym_arm.sh` yourself, the same thing is:

```bash
CFG=+auto_ontology_sql_resources_server.resources_servers.auto_ontology_sql
scripts/run_gym_arm.sh auto_ontology_sql runs/treatment/bird.jsonl \
  --max-output-tokens 16 \
  "$CFG.sql_examples_csv=prompt_inputs/bird/predicted_structural_exemplars.csv" \
  "$CFG.sql_examples_k=5" \
  "$CFG.value_anchors_csv=prompt_inputs/bird/value_anchors.csv"
```

The control arm cannot receive the equivalent, so **a delta measured with these
on covers the exemplars and anchors as well as the ontology grounding** — say so
when reporting it. Each rollout records `sql_examples_used` and
`value_anchors_used` so a run can prove what it actually received rather than
what it was configured to receive; the server also logs a `Prompt inputs:`
summary at startup.

The control's first sentence is upstream `bird_sql`'s prompt verbatim; the
projection rule is our addition, so the baseline is no longer strictly identical
to upstream's. That is the deliberate trade: internal comparability over
comparability to published BIRD numbers.

## The API key never goes on the command line

`run_gym_arm.sh` hands the key over as an OmegaConf *interpolation* rather than
a value:

```
+policy_api_key=${oc.env:DEFAULT_MODELS_API_KEY}
```

`gym eval run --model-api-key X` is only sugar for the override
`+policy_api_key=X`, so passing the override directly costs nothing and keeps
the secret out of `argv` -- where it would be readable by any user on the box
through `ps`, and would end up in shell history and in any log that echoes the
command. OmegaConf resolves it from the environment when the config is read.

Keep it that way. If you add a flag that takes a credential, check whether it
is sugar for an override you can interpolate instead.

## Comparing arms

```bash
python scripts/compare_arms.py --control <control_rollouts> --treatment <auto_ontology_rollouts>
```

Reports accuracy overall and by difficulty, the delta, and which individual
questions the ontology fixed or broke.

## The rate-limit guard, and why it exists

A 60-question BIRD run at 8 workers against `aws/anthropic/bedrock-claude-opus-4-8` came back
at 13.3% execution accuracy. That number was fiction: 747 of 1126 HTTP calls were
429s. Auto Ontology retries three times, gives up, and routes to a *valid* terminal state
(`unconstructable_sql_response`) — so 37 of 60 questions produced empty SQL, scored
zero, and left the error column blank. The run looked like a weak model.

Hence two deliberate features:

- `FailureCode.NO_MODEL_OUTPUT` is distinct from "wrong SQL", and both servers
  report `health/*` counters alongside accuracy.
- `compare_arms.py` **refuses** to print a headline number when either arm is above
  5% broken-run failures (`no_model_output` or `unknown_error`), unless
  `--ignore-health` is passed. Tasks whose *gold* query never ran are excluded
  from the denominator entirely rather than counted as misses.

Size `max_concurrency` against the model endpoint's rate limit, not against CPU.
Three workers is clean on the `sk-` inference-api key; eight is not. Numbers in
`results/CONCURRENCY-STUDY.md` were measured on a different model and do not
transfer.

## Arm B's couplings with Auto Ontology

`auto_ontology_sql` imports Auto Ontology in-process, which forces three things:

- `load_dotenv()` runs **before** the Auto Ontology imports —
  `auto_ontology.retrieval.text_to_sql.main`
  builds its LLM client and compiles the graph at import time.
- The parent project and resource-server environment must be installed before
  starting the arm.
- `stream_agent_response` is a blocking generator, so it runs on a worker thread
  under a semaphore.

Gym's policy model is **not** in the loop for this arm — `verify()` runs the agent
itself and ignores the rollout's model output (`policy_output_ignored: true`). That
keeps the arm behaviourally identical to the existing harness, at the cost of it not
being RL-trainable.

Arm B also needs the pre-ingested Auto Ontology catalog (Postgres + pgvector) for every
database it touches; build it with `main.py --skip-eval --skip-judge`.
