"""Workarounds for known, evidence-confirmed bugs in BIRD-Interact-ADK.

Nothing here modifies ADK itself — these are strictly client-side
corrections applied to *our own* interpretation of ADK's ``/submit``
response, after the fact, for exact, narrowly-matched error signatures we've
independently confirmed are false negatives (ADK determined the submission
correct, then crashed in unrelated bookkeeping *after* that determination,
and a blanket exception handler overwrote the pass with reward=0).

This module is the single choke point both eval scripts
(``scripts/eval_bird_interact.py`` and ``scripts/run_all_bird_interact.py``)
go through, since both call the adapter's ``/run_session`` endpoint (see
``server.py``), which is the only place that calls ``bird_interact_http
.submit_sql`` — so a fix here is automatically shared by both, no per-script
duplication.
"""

from __future__ import annotations

import re
from typing import Optional

# ── labor_certification_applications Phase-1-snapshot name collision ────────
#
# ADK's create_task_db() (BIRD-Interact-ADK/shared/db_utils.py) names a
# per-task Postgres DB f"{base_db}__{task_id}", then — only after a Phase 1
# submission is graded correct — snapshots it for Phase 2 as
# f"{task_db}__p1snap" (db_environment/server.py, inside `if passed:`).
#
# Postgres silently truncates identifiers over 63 characters (NAMEDATALEN).
# For labor_certification_applications (32 chars), task_id is itself
# "labor_certification_applications_<N>" (BIRD-Interact FULL's instance-id
# convention embeds the full db name), so:
#
#   task_db = "labor_certification_applications__labor_certification_applications_<N>"
#           = 68+ chars → truncates to the SAME 63-char string every time,
#             regardless of <N>
#   snap_db = f"{task_db}__p1snap" → also truncates to that identical
#             63-char string (the "__p1snap" suffix never survives)
#
# So `createdb --template <task_db> <snap_db>` always resolves both
# arguments to the same identifier, and createdb refuses ("database already
# exists") — 100% reproducible for every instance of this one DB, confirmed
# directly by string-length arithmetic against the real ADK source and by
# the literal NOTICE Postgres emits on truncation. No other FULL-benchmark
# DB name is long enough to trigger this (next-longest is
# cold_chain_pharma_compliance at 28 chars, which stays under the limit).
#
# The crash happens *after* `passed = True` was already determined
# server-side, and is caught by a blanket `except Exception` at the top of
# ADK's submit handler that returns `passed=False, reward=0.0` — silently
# converting a real pass into a reported failure. Confirmed live: the exact
# reconstructed SQL that triggered this (`LOWER(TRIM(statustag)) =
# LOWER(TRIM('Certified'))`) is semantically equivalent to gold, and the
# same signature was already present in `results/4_0%_DBs/4_0%_DBs_adapter.log`
# for labor_certification_applications_{2,4,6,7,8,18} before this workaround
# existed.
_P1SNAP_COLLISION_RE = re.compile(
    r"createdb.*__p1snap.*returned non-zero exit status", re.IGNORECASE
)

# Copied from BIRD-Interact-ADK/db_environment/server.py's `phase_rewards_first`
# / `phase_rewards_debug` constants (c-interact mode). These are fixed,
# task-independent constants in ADK today — if ADK ever changes them, this
# copy goes stale silently, so it's worth re-checking against ADK's source
# if this workaround starts producing rewards that don't match what a clean
# run of the same instance reports.
_PHASE1_REWARD_FIRST_TRY = 0.7
_PHASE1_REWARD_DEBUG_RETRY = 0.5


def is_p1snap_collision(message: str) -> bool:
    """Whether *message* matches the known Phase-1-snapshot name-collision crash."""
    return bool(message) and bool(_P1SNAP_COLLISION_RE.search(message))


def corrected_phase1_result(message: str, is_first_try: bool) -> Optional[dict]:
    """The corrected ``{"reward": ..., "phase_completed": 1}`` if *message* is
    the known Phase-1-snapshot collision; ``None`` otherwise.

    Callers must only apply the correction when the submission being
    evaluated was actually for Phase 1 (the bug can't occur for Phase 2 —
    ADK only runs this snapshot step immediately after a Phase 1 pass) and
    must leave every other error message untouched.
    """
    if not is_p1snap_collision(message):
        return None
    reward = _PHASE1_REWARD_FIRST_TRY if is_first_try else _PHASE1_REWARD_DEBUG_RETRY
    return {"reward": reward, "phase_completed": 1}


__all__ = ["is_p1snap_collision", "corrected_phase1_result"]
