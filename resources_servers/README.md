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
python scripts/build_gym_tasks.py --dataset bird60 --arm schema_only
python scripts/build_gym_tasks.py --dataset bird60 --arm auto_ontology
```

Datasets: `bird` (500q), `bird60` (60q subset, reuses BIRD's databases),
`fdabench` (169q), `wideworldimporters` (41q, Postgres). Output is deterministic,
so regenerating is safe.

The schema dump *is* the control arm's independent variable. Freeze it once a full
run starts — changing it silently changes what "schema-only" means.

## Running an arm

```bash
scripts/run_gym_arm.sh schema_only_sql  runs/control/bird60.jsonl
scripts/run_gym_arm.sh auto_ontology_sql runs/gsf/bird60.jsonl --max-output-tokens 16
```

Both arms run **in-process**: `auto_ontology_sql` imports Auto Ontology and drives its graph
directly, so no Auto Ontology server is required. `--max-output-tokens 16` on the Auto Ontology arm is
deliberate — that arm ignores the policy model's output, so the call is capped to
stop it competing for rate limit with Auto Ontology's own ~19 calls per question.

Useful environment variables:

| Variable | Default | Notes |
|---|---|---|
| `GYM_CONCURRENCY` | 3 | Size against the **model endpoint's rate limit**, not CPU |
| `GYM_API_KEY` | `.env`'s `DEFAULT_MODELS_API_KEY` | Overrides the key for *both* the policy model and Auto Ontology's own calls |
| `AUTO_ONTOLOGY_PATH` | `../auto-ontology-gym` | Which Auto Ontology checkout to import |
| `GYM_MODEL` / `GYM_MODEL_URL` | bedrock-claude-opus-4-8 @ inference-api | |

`GYM_API_KEY` has to override `DEFAULT_MODELS_API_KEY` rather than just
`--model-api-key`: the Auto Ontology arm authenticates its in-process calls with the former,
so changing only the CLI flag would move the policy model to the new key and leave
the actual workload on the old one.

Give each arm its own output **directory**. Gym writes `preprocessed_datasets/`
beside the output file, and a second arm's collation aborts on the first arm's
leftover metrics ("Found conflicting aggregate metrics").

## Environment

One venv holds both `nemo_gym` and Auto Ontology's dependencies. Getting there took two
dependency decisions, recorded in `overrides.txt` and `pyproject.toml`:

- **`prometheus-fastapi-instrumentator>=8` override.** nemo-retriever caps it at
  `<8`, and every release in that range pins `starlette<1.0.0`, which collides with
  nemo-gym's `ray[serve]` (`starlette>=1.0.1`). The cap is stale — nemo_retriever
  never imports the instrumentator (it imports `prometheus_client`, a different
  package, and only inside its own HTTP service).
- **`langchain-openai<1.3.5`.** nemo-gym 0.6.0 pins `openai==2.44.0`; 1.3.5+ wants
  `openai>=2.45.0`. Unlike the override this is a real squeeze — both packages
  genuinely use the SDK — so it is expressed as a version bound.

nemo-gym 0.6.0 requires Python **>=3.13.14**, which is why `requires-python` was
raised from 3.12.

Auto Ontology is reached via `PYTHONPATH` (see `AUTO_ONTOLOGY_PATH`) because it is not yet an
installable package. **When Auto Ontology becomes installable, it turns into an ordinary line
in each server's `requirements.txt` and the `PYTHONPATH` export disappears.**

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
   *type*. nemo-gym 0.6.0 validates this eagerly; 0.4.0 did not validate it at all
   and silently ignored a wrong value, so runs could appear to work with a bad name.

Gym builds each server its own venv from its `requirements.txt`. If that build fails
with `Package metadata version ... does not match ... from the wheel filename`, it is
a race in nemo-retriever's timestamp-derived version, not a real conflict — retry.

## Dataset coverage

| Dataset | Dialect | In the benchmark? |
|---|---|---|
| `bird` (500q) | SQLite | yes |
| `bird60` (60q) | SQLite | yes — local subset, not committed |
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

The control's first sentence is upstream `bird_sql`'s prompt verbatim; the
projection rule is our addition, so the baseline is no longer strictly identical
to upstream's. That is the deliberate trade: internal comparability over
comparability to published BIRD numbers.

## Known issue: the API key is passed in argv

`scripts/run_gym_arm.sh` passes the model key via `--model-api-key`, which makes
it visible in the process table (`ps`) to any user on the machine, and can leak
into shell history and logs. Accepted on a single-user dev box; **fix before
running this anywhere shared or in CI.**

The fix is to stop putting it on the command line -- resolve it inside a config
instead, e.g. `api_key: ${oc.env:DEFAULT_MODELS_API_KEY}` in a merged config
passed via `--config`, so the value only ever lives in the environment.

## Comparing arms

```bash
python scripts/compare_arms.py --control <control_rollouts> --treatment <gsf_rollouts>
```

Reports accuracy overall and by difficulty, the delta, and which individual
questions the ontology fixed or broke.

## The rate-limit guard, and why it exists

A bird60 run at 8 workers against `aws/anthropic/bedrock-claude-opus-4-8` came back
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

## Arm B's couplings with GSF

`auto_ontology_sql` imports Auto Ontology in-process, which forces three things:

- `load_dotenv()` runs **before** the `gsf` imports — `gsf.retrieval.text_to_sql.main`
  builds its LLM client and compiles the graph at import time.
- `PYTHONPATH` must include the sibling Auto Ontology checkout; Auto Ontology is not installed here.
- `stream_agent_response` is a blocking generator, so it runs on a worker thread
  under a semaphore.

Gym's policy model is **not** in the loop for this arm — `verify()` runs the agent
itself and ignores the rollout's model output (`policy_output_ignored: true`). That
keeps the arm behaviourally identical to the existing harness, at the cost of it not
being RL-trainable.

Arm B also needs the pre-ingested Auto Ontology catalog (Postgres + pgvector) for every
database it touches; build it with `main.py --skip-eval --skip-judge`.
