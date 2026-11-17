#!/usr/bin/env bash
# Build the Auto Ontology agent's sandbox image, with BIRD dev's catalog in it.
#
#   scripts/build_agent_image.sh [MODELS_ENV] [TAG]
#
# MODELS_ENV (default .env) holds the model keys the catalog build uses:
# DEFAULT_MODELS_API_KEY, and EMBED_* / NON_REASONING_* where they differ. It
# reaches the build as a BuildKit secret, so no layer holds it. TAG defaults to
# auto-ontology-eval/agent:bird, the image the agent's config names.
#
# Seeds BIRD dev first when it is missing. The catalog build runs ingest and
# the semantic compile over all 11 databases, which takes a while and calls
# the embedding and compile models.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODELS_ENV="${1:-.env}"
TAG="${2:-auto-ontology-eval/agent:bird}"

[[ -f "$MODELS_ENV" ]] || { echo "build_agent_image.sh: no $MODELS_ENV" >&2; exit 1; }
if ! compgen -G "datasets/bird/dev/*/*.sqlite" >/dev/null; then
  uv run python scripts/seed/seed_bird.py
fi

DOCKER_BUILDKIT=1 docker build \
  --file docker/auto_ontology_agent/Dockerfile \
  --secret "id=models,src=$MODELS_ENV" \
  --tag "$TAG" \
  .
