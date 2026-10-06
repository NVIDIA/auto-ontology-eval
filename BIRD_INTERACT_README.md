# BIRD-Interact: how to run it and read the results

This is the single onboarding doc for BIRD-Interact in this repo. Individual
scripts have their own docstrings/`--help` with full flag details — this page
ties them together and calls out the things that are easy to miss.

## Scope — read this first

- **c-Interact only.** a-Interact is not implemented.
- **Query-category tasks only.** Management tasks are explicitly excluded —
  `scripts/run_all_bird_interact.py` hardcodes a `category == "query"` filter
  (`_load_tasks`, look for `# Hardcoded: query category only`). If you write a
  new run script, carry this filter forward or call out that it doesn't.
- Two dataset variants are supported: **Lite** (~300 tasks / 18 DBs) and
  **Full** (600 tasks / 22 DBs).
- See also the [BIRD-Interact entry](README.md#bird-interact) in the top-level README.md.

## Pipeline flow

Each task runs `init → Phase 1 (clarify + generate SQL) → [Phase 1 debug retry
if wrong] → Phase 2 (follow-up + generate SQL) → [Phase 2 debug retry if
wrong]`. Phase 1 is the only phase that asks the user clarifying questions —
Phase 2 asks none.

**Clarification (Phase 1 only).** Each turn checks output type, resolves
entities against the schema/KB, and scans for incomplete formulas, then an
LLM decides ASK or PROCEED. After each answer, **merge question** (`merge.py`)
folds it into the running question, preserving prior formulas/definitions, so
SQL generation sees one coherent question instead of a scattered Q&A
transcript.

**Debug phase.** If a submitted SQL fails, an execution error is relayed back
verbatim; a wrong-but-executable result gets a generic retry prompt plus a
couple of correctness checklists instead. Straight to another SQL-generation
attempt.

**Follow-up (Phase 2).** No clarification. **Merge follow-up question**
(`followup_merge.py`) rewrites the follow-up into a self-contained question by
resolving references against the Phase 1 question and its SQL, carrying forward
the columns/tables/formulas/values it depends on.

```
init
├─ Phase 1
│   ├─ clarify loop: ask user ⇄ merge question   (repeats until ready)
│   ├─ generate SQL → submit
│   └─ if failed → debug retry (no clarify) → submit
├─ phase_transition
├─ Phase 2
│   ├─ merge follow-up question
│   ├─ generate SQL → submit
│   └─ if failed → debug retry → submit
└─ cleanup
```

## 1. Seed the dataset

```bash
uv run python scripts/seed_bird_interact.py --dataset lite   # or: full
```

This sparse-clones the upstream ADK repo (`github.com/bird-bench/BIRD-Interact`,
ADK code + Docker `env/` only) into `third_party/BIRD-Interact/`, downloads the
task JSONL from HuggingFace (`birdsql/bird-interact-<variant>`), and builds
`datasets/bird_interact[_full]/` (public JSONL, combined JSONL, `manifest.json`,
`gt/` placeholder). `--dataset` is required, no default. Use `--force` to
re-download.

## 2. Get Ground Truth (GT)

GT SQLs and test cases are **not** in the public repo/dataset — without them,
scoring is incomplete (the seeder falls back to the public JSONL as a
plumbing-only placeholder).

Two ways to get the GT file:
- **Email** `bird.bench25@gmail.com` with subject exactly
  `[bird-interact-<variant> GT&Test Cases]` (e.g. `[bird-interact-lite GT&Test Cases]`).
  Files arrive automatically within ~30 minutes.
- **Shared team drive** — the GT files have also been saved there; check before
  emailing if you just need a copy a teammate already has.

Drop the received file at:

```
datasets/bird_interact[_full]/gt/bird_interact_data_with_gt.jsonl
```

Then re-run the seeder (step 1) to regenerate the combined data file that the
runners actually read.

## 3. Start the Bird services

```bash
scripts/start_bird_services.sh
```

Starts ADK's `user_simulator` (`:6001`) and `db_environment` (`:6002`).
**Always use this wrapper, never ADK's own `start_services.sh` directly** — it
applies our required ADK patches first (`patches/adk_*.patch`) and refuses to
start unpatched. Without the patch, Phase 2 scoring for
`labor_certification_applications` (FULL dataset) silently mis-grades due to a
Postgres 63-char identifier truncation collision — see
`ontology_sql_eval/bird_interact/known_issues.py` for the full mechanism.

Resolves the ADK checkout via `$BIRD_INTERACT_ADK_DIR` if set, else
`third_party/BIRD-Interact/BIRD-Interact-ADK`. `START_BIRD_SERVICES_DRY_RUN=1`
prints the command instead of running it.

You'll also need the Bird PostgreSQL task DBs running (Docker) and
`pg_wrappers` shims set up for `:6002` scoring — the seeder prints the exact
commands at the end of step 1 (`docker compose up -d ...`,
`scripts/setup_pg_wrappers.py`).

## 4. Run

Three run scripts, same underlying pipeline (`init → phase1 [→ debug] → phase2 [→ debug]`):

| Script | Use for |
| --- | --- |
| `scripts/smoke_bird_interact.py` | One task, minimal — sanity-check the three-port setup end to end, then shuts the adapter down. |
| `scripts/eval_bird_interact.py` | One task, full parity with the real orchestrator flow (pick by `--instance-id`, `--db`, or `--random`). |
| `scripts/run_all_bird_interact.py` | Batch/overnight — all (or filtered) query tasks, adapter started once and reused, results written incrementally so a crash doesn't lose progress. |
| `scripts/run_bird_interact_cinteract.py` | Runs the **official** Bird c-Interact orchestrator (not our own re-implementation) against the Auto Ontology adapter. ⚠️ Its manual uvicorn bootstrap does **not** go through `start_bird_services.sh`, so it does **not** apply the ADK patches — prefer one of the scripts above unless you specifically need the official orchestrator, and if you do use it, apply `patches/adk_p1snap_name_collision.patch` to the ADK checkout manually first (the script prints a loud warning if it detects the checkout is unpatched). |

All three (except `run_bird_interact_cinteract.py`) require `:6001` and
`:6002` already up (step 3) and start the Auto Ontology adapter themselves (default port
`:6003`).

`run_all_bird_interact.py` is the one you'll use for scored runs. Key flags:

| Flag | Purpose |
| --- | --- |
| `--data` | Path to the combined GT jsonl (default: `datasets/bird_interact/bird_interact_data_with_gt.jsonl`). Point this at `bird_interact_full/...` to run Full. |
| `--output` | Output stem → `results/<stem>/<stem>.jsonl` + `.csv` (default `cinteract_query_all`). |
| `--limit N` | Cap number of tasks. |
| `--difficulty {simple,moderate,challenging}` | Filter to one tier. |
| `--shuffle` | Randomize task order. **Not seeded** — there is no `random.seed()` call anywhere in the pipeline, so a shuffled run's task order is not reproducible between runs. If you need a reproducible subset, use `--include`/`--force-include` with explicit instance IDs instead. |
| `--include IDS...` / `--exclude IDS...` | Run only / skip specific instance_ids. |
| `--force-include IDS...` | Guarantee these instance_ids are in the run, fill the rest of `--limit` from the remaining pool. |
| `--agent-port` (default 6003) | Auto Ontology adapter port. |
| `--phase-timeout` (default 600s) | Per `run_session` HTTP call — matches the official orchestrator's timeout. |
| `--health-timeout` (default 10s) | Per-service health-check timeout before giving up. |
| `--overwrite` | Overwrite an existing output stem instead of failing. |

Recommended for long runs (prevents the Mac from sleeping mid-run):

```bash
caffeinate -i uv run python scripts/run_all_bird_interact.py --output my_run_name
```

## 5. Results — where and what format

Each run writes to `results/<stem>/`:

- **`<stem>.jsonl`** — rich per-task run records (dialogue history, extracted
  entities, timing, generated SQL per phase, etc.). Join with the source
  dataset by `instance_id` for difficulty/ambiguity fields — the run record
  itself only carries run-time data.
- **`<stem>.csv`** — lightweight, written incrementally (crash-safe). Columns:
  `instance_id, dataset, database, max_turn, turns_used, phase1_passed,
  phase1_debug_ran, phase2_passed, phase2_debug_ran, total_reward,
  exec_error_p1, exec_error_p2, error`.
- **`<stem>_adapter.log`** — Auto Ontology adapter stdout/stderr for the run.

Then run analysis on the JSONL:

```bash
python scripts/analyze_bird_interact.py results/<stem>/<stem>.jsonl
```

This produces (all under `results/<stem>/`, no LLM calls, joins with the
source dataset for difficulty/ambiguity fields):

- **`<stem>_report.txt`** — full text report (tee'd from stdout).
- **`<stem>_analysis.json`** — structured version of the same analysis.
- **`<stem>_charts.png`** / **`<stem>_charts_diag.png`** — score distribution
  and diagnostic charts.
- **`<stem>_instances.txt`** — plain list of instance IDs in the run.

**"SR" = Success Rate.** The report's "Leaderboard-style SR" section prints
four SR variants (denominator = number of tasks that ran without a hard
error), each as `n/total (pct%)`:

- `P1 SR (no debug)` — Phase 1 passed on the first try.
- `P1 SR (+debug)` — Phase 1 passed either on the first try or after the debug retry.
- `Follow-up SR (no debug)` — Phase 2 passed on the first try (only tasks with a follow-up).
- `Follow-up SR (+debug)` — Phase 2 passed first try or after debug retry.

## Environment variables

| Variable | Used by | Purpose |
| --- | --- | --- |
| `BIRD_INTERACT_ADK_DIR` | `start_bird_services.sh`, `eval_bird_interact.py`, `run_bird_interact_cinteract.py` | Overrides the ADK checkout path (default `third_party/BIRD-Interact/BIRD-Interact-ADK`). Not in `.env.example`. |
| `START_BIRD_SERVICES_DRY_RUN` | `start_bird_services.sh` | Set to print the command instead of running it. |

General prerequisites (Postgres, `.env`, `uv sync`) are covered in the
top-level [README.md](README.md#prerequisites) — they apply here too.

## Known issues

- **`labor_certification_applications` name-collision bug** — see
  `ontology_sql_eval/bird_interact/known_issues.py`. Fixed by
  `patches/adk_p1snap_name_collision.patch`, applied automatically (and
  mandatorily) by `start_bird_services.sh`. If you bypass that script (e.g.
  `run_bird_interact_cinteract.py`'s manual bootstrap), this bug can silently
  overwrite a real Phase 1 pass with `reward=0` and mis-grade Phase 2 against
  Phase 1's solution instead of the follow-up's.

## Work process and solution method

### Clarification

Initially, most failures were due to insufficient clarification. The fix was
to examine conversation flows to identify which questions users tended not to
answer, then add guards to catch them — LLM-based guards like a completeness
check (especially useful for questions with an incompletely specified
formula), and deterministic guards like flagging 2+ close VDB hits (an
embedding-distance gap check, not regex).

Eventually, a large majority of remaining failures were categorized as
SQL-generation failures rather than clarification failures. That drove the
next debugging stage, handled two ways:

### SQL-generation

- **Error buckets** — use an LLM to compare GT SQL against submitted SQL
  across hundreds of tasks, producing a short failure reason for each. Cluster
  those reasons by similarity into failure "buckets," each tagged with how
  many tasks it affects. That count, combined with how solvable the bucket
  looked, set the priority for which bugs to address first. Typical fixes
  were adding a guard, an entry in a prompt, or another line in the debug
  seed.

- **Outliers** — scan the codebase for non-standard errors, such as timeouts
  or unusual cases (e.g. the submitted SQL being identical to gold, which
  pointed to a different class of bug). Example fixes included adding
  fallback options so a single LLM crash doesn't lose all information for
  that task.