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

# Both human discourse maps. "post" (six post-deliberative discourses) is the
# primary benchmark; "pre" (four pre-deliberative mapping-study discourses) is
# the secondary comparison and the only one with population weights.
BASELINES=("post" "pre")

# Both judges. One judge cannot separate a property of the essays from a
# property of the rater -- and on this corpus the two disagree sharply.
JUDGES=("google/gemma-4-31b-it" "openai/gpt-oss-120b")

# Both Q-statement phrasings, since neither is obviously correct.
FORMS=("depersonalised" "original")

for prompt in "${PROMPTS[@]}"; do
    echo
    echo "======================================================================"
    echo "  Condition: $prompt"
    echo "======================================================================"

    # Generation and segmentation are baseline-independent: the same essays are
    # scored against both discourse maps.
    "$PYTHON" src/generate.py  --prompt "$prompt"
    "$PYTHON" src/segment.py   --prompt "$prompt"
    "$PYTHON" src/sentences.py --prompt "$prompt"

    # Stance scoring is also baseline-independent -- it scores sentences against
    # the surveyed statements, which are the same instrument in both maps.
    for form in "${FORMS[@]}"; do
        "$PYTHON" src/stance.py --prompt "$prompt" --form "$form"
    done
    # Anaphora robustness pass. Not the primary scoring -- see
    # src/analyse_context.py, which shows it double-counts the prepended
    # sentence -- but the comparison is what establishes that.
    "$PYTHON" src/stance.py --prompt "$prompt" --form depersonalised --context

    for baseline in "${BASELINES[@]}"; do
        echo
        echo "  --- baseline: $baseline ---"
        "$PYTHON" src/embed.py           --prompt "$prompt" --baseline "$baseline"
        "$PYTHON" src/embed_sentences.py --prompt "$prompt" --baseline "$baseline"
        "$PYTHON" src/analyse.py         --prompt "$prompt" --baseline "$baseline"
        for form in "${FORMS[@]}"; do
            "$PYTHON" src/analyse_sentences.py \
                --prompt "$prompt" --baseline "$baseline" --form "$form"
        done
        "$PYTHON" src/analyse_sentences.py \
            --prompt "$prompt" --baseline "$baseline" \
            --form depersonalised --context
        "$PYTHON" src/analyse_context.py --prompt "$prompt" --baseline "$baseline"
        # Whose voice carries each discourse. Runs after analyse_sentences.py,
        # which is where it picks up the dismissal verdicts.
        "$PYTHON" src/attribution.py --prompt "$prompt" --baseline "$baseline"
        # Is the baseline measuring discourse, or topic? Always worth printing.
        "$PYTHON" src/validate_baseline.py --prompt "$prompt" --baseline "$baseline"
        # LLM-as-judge: the only instrument that asks the question directly.
        # Resumable, so a rerun costs nothing once the cache is warm.
        for judge in "${JUDGES[@]}"; do
            "$PYTHON" src/judge.py --prompt "$prompt" --baseline "$baseline" \
                --judge-model "$judge"
            "$PYTHON" src/analyse_judge.py --prompt "$prompt" \
                --baseline "$baseline" --judge-model "$judge"
        done
        # Do the judges agree? With one judge you cannot tell a flat result
        # from a flat rater.
        "$PYTHON" src/compare_judges.py --prompt "$prompt" --baseline "$baseline"
    done
done

echo
echo "======================================================================"
echo "  Comparison across conditions"
echo "======================================================================"
"$PYTHON" src/compare.py

echo
echo "Done. Per-condition results in results/<prompt>/<baseline>/summary.md"
echo "Cross-condition comparison in results/comparison.md"
