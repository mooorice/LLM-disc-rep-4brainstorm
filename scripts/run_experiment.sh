#!/usr/bin/env bash
#
# Run the whole experiment: both prompt conditions, all four stages each.
#
#   ./scripts/run_experiment.sh
#
# Safe to re-run. Generation skips essays already on disk, so an interrupted run
# picks up where it stopped and a rerun costs nothing. The later stages simply
# recompute, which is cheap.
#
# Before the first run, check credentials with:
#   python src/generate.py --smoke

set -euo pipefail

cd "$(dirname "$0")/.."

# Use the project virtualenv if there is one, otherwise whatever python is on
# the path.
PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python"

PROMPTS=("brainstorm_australian" "brainstorm_generic")

for prompt in "${PROMPTS[@]}"; do
    echo
    echo "======================================================================"
    echo "  Condition: $prompt"
    echo "======================================================================"

    "$PYTHON" src/generate.py --prompt "$prompt"
    "$PYTHON" src/segment.py  --prompt "$prompt"
    "$PYTHON" src/embed.py    --prompt "$prompt"
    "$PYTHON" src/analyse.py  --prompt "$prompt"
done

echo
echo "======================================================================"
echo "  Comparison across conditions"
echo "======================================================================"
"$PYTHON" src/compare.py

echo
echo "Done. Per-condition results in results/<prompt>/summary.md"
echo "Cross-condition comparison in results/comparison.md"
