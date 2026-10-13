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
if [[ -z "${GYM_REBUILD_VENV:-}" ]]; then
  GYM_FLAGS+=(+skip_venv_if_present=true)
fi

# --agent takes the fully-qualified server-instance name: nemo-gym 0.6.0
# rejects the bare inner key ("simple_agent") that 0.4.0 accepted.
#
# Gym builds each server its own venv from its requirements.txt. Auto Ontology
# is installed transitively from the parent project's pinned Git dependency.
exec .venv/bin/gym eval run \
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
  "$@"
