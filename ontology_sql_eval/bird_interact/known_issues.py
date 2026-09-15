"""Detector for a known, evidence-confirmed BIRD-Interact-ADK bug signature.

ADK's create_task_db() (BIRD-Interact-ADK/shared/db_utils.py) used to be
able to crash on a DB-name-length collision immediately after determining a
Phase 1 submission correct, silently overwriting a real pass with reward=0
(and, as a second-order effect, causing Phase 2 to be graded against Phase
1's solution instead of the follow-up's — see the patch for the full
mechanism). That root cause is now fixed at the source:
``patches/adk_p1snap_name_collision.patch``, applied automatically and
mandatorily by ``scripts/start_bird_services.sh`` (which is the only
supported way to start ADK's services — see that script) before ADK's
services ever start. Since that script fails hard if the patch can't be
applied, "ADK services are running" now implies "this bug can't fire".

So this signature should be unreachable in any run through
``scripts/start_bird_services.sh``. This module exists to make that
invariant loud rather than silent if it's ever violated (e.g. a run that
bypassed the wrapper, or an ADK version bump the patch stopped matching) —
``server.py`` calls ``is_p1snap_collision()`` on a `/submit` error message
and raises immediately rather than guessing at a corrected score, since a
run this can happen in has an unverified setup problem worth stopping for,
not papering over.
"""

from __future__ import annotations

import re

# Postgres silently truncates identifiers over 63 characters (NAMEDATALEN).
# For labor_certification_applications (32 chars), task_id is itself
# "labor_certification_applications_<N>" (BIRD-Interact FULL's instance-id
# convention embeds the full db name), so the Phase-1 task DB name and its
# "__p1snap" Phase-2 snapshot name both truncated to the identical 63-char
# string — making `createdb --template <task_db> <snap_db>` fail 100% of the
# time, confirmed directly by string-length arithmetic against ADK's source
# and by the literal NOTICE Postgres emits on truncation. No other
# FULL-benchmark DB name is long enough to trigger this (next-longest is
# cold_chain_pharma_compliance at 28 chars, which stays under the limit).
_P1SNAP_COLLISION_RE = re.compile(
    r"createdb.*__p1snap.*returned non-zero exit status", re.IGNORECASE
)


def is_p1snap_collision(message: str) -> bool:
    """Whether *message* matches the known Phase-1-snapshot name-collision crash."""
    return bool(message) and bool(_P1SNAP_COLLISION_RE.search(message))


__all__ = ["is_p1snap_collision"]
