#!/usr/bin/env bash
# Run one arm of the ontology evaluation through NeMo Gym.
#
#   scripts/run_gym_arm.sh schema_only_sql runs/control/bird.jsonl [args...]
#
# Give each arm its own output DIRECTORY. Gym writes preprocessed_datasets/
# next to the output file, and a second arm's collation aborts on the first
# arm's leftover metrics ("Found conflicting aggregate metrics").
#
# Separate output directories do not cover every case of that error. Gym also
# caches tasks_metrics.json and tasks_prepare.jsonl beside the *task* file, in
# resources_servers/<arm>/data/, so rebuilding tasks.jsonl with a different
# question count makes the next run of the same arm abort at collation -- after
# the compute is spent. main.py clears those before each build, which is why a
# comparison should go through it rather than calling this script twice.
#
# Concurrency defaults to 3: the sk- inference-api key rate-limits hard above
# that, and Auto Ontology's retry turns a 429 into empty SQL rather than a visible error.
# 10 is measured-safe on the sk- key across two full 1,534-question runs.
set -euo pipefail

SERVER="${1:?usage: run_gym_arm.sh <resources-server> <output.jsonl> [args...]}"
OUTPUT="${2:?usage: run_gym_arm.sh <resources-server> <output.jsonl> [args...]}"
shift 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "run_gym_arm.sh: no .env in $ROOT (copy .env.example and fill it in)" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
. ./.env
set +a

# The schema-only arm imports ontology_sql_eval.gym.* (the shared executors and
# grader) but deliberately does not install this project, since that would pull
# in Auto Ontology. It finds the package through this PYTHONPATH entry: Gym puts
# only each server's own directory there, and keeps whatever it inherits. The
# repo root used to ride along with the GSF checkout path on this line, and went
# when that was removed.
# Set after .env is sourced, so a PYTHONPATH there cannot drop the root.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Ray, which Gym starts every server under, treats a `uv run` anywhere among a
# process's ancestors as "this is a uv project" and then refuses to start any
# server whose working directory does not contain that project's pyproject.toml.
# Gym's legacy_agent environment server runs from site-packages, so the
# documented `uv run python main.py` died before the first question. Each server
# already has its own venv, so there is nothing for Ray to replicate.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0


# GYM_API_KEY overrides the key for *both* consumers, and must be applied after
# sourcing .env (which would otherwise clobber it). The Auto Ontology arm runs the agent
# in-process, so Auto Ontology's own ~19 calls per question authenticate with
# DEFAULT_MODELS_API_KEY -- and so does the policy model, which reads it through
# the interpolation below. Pointing only the policy model at a new key would
# leave the actual workload on the old one.
if [[ -n "${GYM_API_KEY:-}" ]]; then
  export DEFAULT_MODELS_API_KEY="$GYM_API_KEY"
fi

# Named explicitly: under `set -u` an unset key would otherwise abort with a bare
# "DEFAULT_MODELS_API_KEY: unbound variable" at the exec line below.
if [[ -z "${DEFAULT_MODELS_API_KEY:-}" ]]; then
  echo "run_gym_arm.sh: no model API key. Set DEFAULT_MODELS_API_KEY in .env," \
       "or pass GYM_API_KEY=<key> to override it for this run." >&2
  exit 1
fi

# The key is handed over as an OmegaConf interpolation, not a value. `gym eval
# run --model-api-key X` is only sugar for the override `+policy_api_key=X`, so
# passing the override directly keeps the secret out of argv -- where it would
# otherwise be readable by any user on the box via `ps`, and would land in
# shell history and in any log that echoes the command.
#
# Single-quoted so the shell leaves ${...} alone; OmegaConf resolves it from the
# environment when the config is read. DEFAULT_MODELS_API_KEY is exported above,
# so Gym's child processes inherit it either way.
GYM_FLAGS=('+policy_api_key=${oc.env:DEFAULT_MODELS_API_KEY}')

# Reuse the server venv Gym already built. Rebuilding costs minutes and can trip
# a race in nemo-retriever's timestamp-derived version ("Package metadata
# version ... does not match ... from the wheel filename").
#
# Set GYM_REBUILD_VENV=1 after editing a server's requirements.txt or the root
# pyproject's [tool.uv] overrides, which uv applies there too. Skipping is safe
# only while the declared dependencies and the built venv agree, and a
# dependency edit that silently never takes effect is the failure this whole
# script is trying to make impossible.
#
# The packages whose mismatch breaks a run are checked here rather than left to
# memory. A server venv on a different nemo-gym runs old Gym code under the new
# head; a different ray cannot join the head's cluster; a newer nemo-retriever
# silently retrieves no tables. OpenAI is deliberately excluded: Auto Ontology
# needs the parent's newer client while nemo-gym pins 2.44.0 in its HTTP-isolated
# server venvs; allow_openai_version_skew above makes that split explicit. Only
# the venvs Gym uses for this run are checked, at the <type>/<name>/.venv paths
# it keeps them under --search-dir.
CHECKED_DISTS=(nemo-gym ray openai nemo-retriever)
dist_versions() {  # <venv>: a "name version" line per CHECKED_DISTS entry it has
  "$1/bin/python" - "${CHECKED_DISTS[@]}" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version
for name in sys.argv[1:]:
    try:
        print(name, version(name))
    except PackageNotFoundError:
        pass
PY
}
if [[ ! -x .venv/bin/python ]] || ! parent="$(dist_versions .venv)"; then
  echo "run_gym_arm.sh: no working .venv in $ROOT; run \`uv sync\` first" >&2
  exit 1
fi
# The comparison below is only as good as .venv itself: after a pull without
# `uv sync`, .venv and the server venvs agree on the *old* versions.
pinned="$(sed -n 's/^ *"nemo-gym==\([^"]*\)".*/\1/p' pyproject.toml)"
if [[ -n "$pinned" ]] && ! grep -qx "nemo-gym $pinned" <<<"$parent"; then
  echo "run_gym_arm.sh: .venv does not have pyproject.toml's nemo-gym==$pinned;" \
       "run \`uv sync\` first" >&2
  exit 1
fi
run_venvs=(
  "resources_servers/$SERVER/.venv"
  responses_api_agents/simple_agent/.venv
  responses_api_models/inference_provider/.venv
  environment_servers/legacy_agent/.venv
)
stale_venvs=()
for venv in "${run_venvs[@]}"; do
  [[ -x "$venv/bin/python" ]] || continue
  if ! have="$(dist_versions "$venv")"; then
    stale_venvs+=("$venv: its python fails to start")
    continue
  fi
  while read -r name version; do
    [[ -n "$name" ]] || continue
    want="$(awk -v n="$name" '$1 == n { print $2 }' <<<"$parent")"
    if [[ "$version" != "$want" ]]; then
      stale_venvs+=("$venv has $name $version, .venv has ${want:-none}")
    fi
  done <<<"$have"
done
if [[ ${#stale_venvs[@]} -gt 0 && -z "${GYM_REBUILD_VENV:-}" ]]; then
  echo "run_gym_arm.sh: rebuilding server venvs that are out of step with .venv:" >&2
  printf '  %s\n' "${stale_venvs[@]}" >&2
  GYM_REBUILD_VENV=1
fi
if [[ -z "${GYM_REBUILD_VENV:-}" ]]; then
  GYM_FLAGS+=(+skip_venv_if_present=true)
fi

# Per-question token counts. With observability on, Gym's model servers record
# every call under the rollout that made it, and each rollout row gets an
# ng_model_call_capture block with its token totals. The directory must be
# absolute; it sits beside the output so each arm keeps its own.
CAPTURE_DIR="$(dirname "$OUTPUT")/model_call_capture"
mkdir -p "$CAPTURE_DIR"
CAPTURE_DIR="$(cd "$CAPTURE_DIR" && pwd)"
GYM_FLAGS+=(+observability_enabled=true "+model_call_capture_dir=$CAPTURE_DIR")

# The Auto Ontology arm sends its own ~19 calls a question through two Gym model
# servers, so they are captured too. Their endpoint/model/key come from Auto
# Ontology's own resolution, exported here for the arm's YAML; see
# ontology_sql_eval/gym/model_routing.py. Assigned before eval so a failure
# stops the script under `set -e`, which `eval "$(...)"` would not.
if [[ "$SERVER" == auto_ontology_sql ]]; then
  routing_env="$(.venv/bin/python -m ontology_sql_eval.gym.model_routing)"
  eval "$routing_env"
fi

# --agent takes the fully-qualified server-instance name: nemo-gym 0.6.0 and
# later reject the bare inner key ("simple_agent") that 0.4.0 accepted.
#
# Gym builds each server its own venv from its requirements.txt. Auto Ontology
# is installed transitively from the parent project's pinned Git dependency.
#
# Gym is started in a process group of its own, and the whole group is signalled
# when this script exits. Gym starts each server as `bash -c "... && python
# app.py"` and, at shutdown, signals only that bash. macOS's bash 3.2 runs the
# python as a child rather than exec'ing it, so the server never hears the
# signal; when Gym then SIGKILLs the bash, the server is orphaned and keeps
# running, two to four of them per arm. They stay in Gym's process group, which
# is how the trap below still reaches them.
set -m  # job control: the background job gets its own process group
.venv/bin/gym eval run \
  --search-dir . \
  --resources-server "$SERVER" \
  --model-type inference_provider \
  --agent "${SERVER}_simple_agent" \
  --split validation \
  --output "$OUTPUT" \
  --concurrency "${GYM_CONCURRENCY:-3}" \
  --model "${GYM_MODEL:-aws/anthropic/bedrock-claude-opus-4-8}" \
  --model-url "${GYM_MODEL_URL:-https://inference-api.nvidia.com/v1}" \
  "${GYM_FLAGS[@]}" \
  "$@" &
gym_pgid=$!

# Ctrl-C reaches only this script's process group, so pass it on to Gym's
# group for a graceful shutdown. On any exit, clean up whatever is left in it.
trap 'kill -INT -- "-$gym_pgid" 2>/dev/null || true' INT TERM
trap 'kill -TERM -- "-$gym_pgid" 2>/dev/null || true' EXIT

# A trapped signal interrupts `wait`, so keep waiting until Gym has exited;
# the last wait carries its exit status.
status=0
wait "$gym_pgid" || status=$?
while kill -0 "$gym_pgid" 2>/dev/null; do
  status=0
  wait "$gym_pgid" || status=$?
done
exit "$status"
