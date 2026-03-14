#!/usr/bin/env bash
# Generate a story from a prompt
# Usage: ./scripts/story.sh "The little dog"

set -euo pipefail

PROMPT="${1:-Once upon a time}"

python scripts/generate.py \
  --checkpoint checkpoints/model.orbax \
  --device gpu \
  --temperature 0.8 \
  --max-tokens 200 \
  --prompt "$PROMPT"
