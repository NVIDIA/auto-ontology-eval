#!/usr/bin/env bash
# Run one arm of the ontology evaluation through NeMo Gym.
#
#   scripts/run_gym_arm.sh schema_only_sql runs/control/bird.jsonl [args...]
#
# Give each arm its own output DIRECTORY. Gym writes preprocessed_datasets/
# next to the output file, and a second arm's collation aborts on the first
# arm's leftover metrics ("Found conflicting aggregate metrics").
#
# Concurrency defaults to 3: the sk- inference-api key rate-limits hard above
# that, and Auto Ontology's retry turns a 429 into empty SQL rather than a visible error.
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
# DEFAULT_MODELS_API_KEY -- overriding only --model-api-key would change the
# policy model's key and leave Auto Ontology on the old one.
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

# Absolute, because Gym runs each server after `cd`-ing into its own directory.
# The default is the public NVIDIA/auto-ontology checkout cloned beside this repo.
export PYTHONPATH="${AUTO_ONTOLOGY_PATH:-$ROOT/../auto-ontology}:$ROOT"

# --agent takes the fully-qualified server-instance name: nemo-gym 0.6.0
# rejects the bare inner key ("simple_agent") that 0.4.0 accepted.
#
# Gym builds each server its own venv from its requirements.txt. Auto Ontology
# arrives from the public source checkout via PYTHONPATH so the Gym server and
# the project use the same checked-out revision despite their dependency pins.
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
  --model-api-key "$DEFAULT_MODELS_API_KEY" \
  "$@"
