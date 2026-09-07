#!/usr/bin/env bash
# Wrapper around BIRD-Interact-ADK/scripts/start_services.sh that applies
# our documented, evidence-confirmed ADK patches (see ../patches/ and
# ontology_sql_eval/bird_interact/known_issues.py) before launching the real
# ADK services — the patching decision lives here, not inside ADK's own
# start_services.sh, so that file stays byte-for-byte pristine upstream.
#
# Use this instead of calling BIRD-Interact-ADK/scripts/start_services.sh
# directly. Resolves the ADK checkout the same way scripts/eval_bird_interact.py
# does: $BIRD_INTERACT_ADK_DIR if set, else third_party/BIRD-Interact/BIRD-Interact-ADK.
set -euo pipefail

ONTOLOGY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PATCH_DIR="$ONTOLOGY_DIR/patches"
_default_adk="$ONTOLOGY_DIR/third_party/BIRD-Interact/BIRD-Interact-ADK"
ADK_DIR="${BIRD_INTERACT_ADK_DIR:-$_default_adk}"

if [ ! -d "$ADK_DIR" ]; then
    echo "ERROR: BIRD-Interact-ADK not found at $ADK_DIR (set BIRD_INTERACT_ADK_DIR)" >&2
    exit 1
fi

# Apply every patch idempotently; fail hard (not just warn) if one can't be
# applied and isn't already applied — services must never start unpatched,
# since ontology_sql_eval/bird_interact/known_issues.py relies on that as an
# invariant rather than working around it at run time.
for p in "$PATCH_DIR"/adk_*.patch; do
    [ -f "$p" ] || continue
    if git -C "$ADK_DIR" apply --check "$p" 2>/dev/null; then
        git -C "$ADK_DIR" apply "$p"
        echo "Applied ADK patch: $(basename "$p")"
    elif git -C "$ADK_DIR" apply --reverse --check "$p" 2>/dev/null; then
        : # already applied — nothing to do
    else
        echo "FATAL: ADK patch $(basename "$p") did not apply to $ADK_DIR (upstream ADK may have changed) — refusing to start services unpatched. See $PATCH_DIR" >&2
        exit 1
    fi
done

if [ -n "${START_BIRD_SERVICES_DRY_RUN:-}" ]; then
    echo "DRY RUN: would exec $ADK_DIR/scripts/start_services.sh $*"
    exit 0
fi
exec "$ADK_DIR/scripts/start_services.sh" "$@"
