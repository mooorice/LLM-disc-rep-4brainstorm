"""
Step 5: compare the prompt conditions against each other.

Each condition is analysed independently by src/analyse.py; this script puts
their results side by side so the effect of the prompt itself is visible. The
question it answers is whether naming Australia in the prompt changes which
discourses the models voice -- and if it does, the Australian-weighted target is
only strictly meaningful for the condition that names Australia.

Reads results/<prompt>/proportions_by_model.csv for every prompt in
config.PROMPTS and writes results/comparison.md.

Usage:
    python src/compare.py
"""

import pandas as pd

import config
from analyse import pct, total_variation_distance


def load_condition(prompt_name: str) -> pd.DataFrame | None:
    """Load one condition's per-model proportions, or None if it has not been run."""
    path = config.RESULTS_DIR / prompt_name / "proportions_by_model.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def pooled_row(by_model: pd.DataFrame, codes: list[str]) -> pd.Series:
    """
    Collapse the per-model proportions into one distribution over all essays.

    Each model's shares are weighted by how many paragraphs it actually
    contributed, so a model that wrote longer essays counts for more. This is
    the same number as recomputing from the paragraph table, without reloading it.
    """
    assigned = by_model["n_paragraphs"] * (1 - by_model["unassigned_share"])
    weights = assigned / assigned.sum()
    return pd.Series(
        {code: float((by_model[code] * weights).sum()) for code in codes}
    )


def main() -> None:
    # Expected shares, restricted to the active discourses and renormalised the
    # same way analyse.py does it.
    weights_table = pd.read_csv(config.WEIGHTS_FILE).set_index("discourse")
    expected = weights_table["weight_all_loadings"].reindex(config.ACTIVE_DISCOURSES)
    expected = expected / expected.sum()
    codes = list(expected.index)

    conditions = {name: load_condition(name) for name in config.PROMPTS}
    missing = [name for name, frame in conditions.items() if frame is None]
    conditions = {name: frame for name, frame in conditions.items() if frame is not None}

    if not conditions:
        raise FileNotFoundError(
            "No condition has been analysed yet. Run src/analyse.py first, or "
            "use scripts/run_experiment.sh to run everything."
        )

    lines = [
        "# Prompt conditions compared",
        "",
        "`brainstorm_australian` names the deliberating public as Australian; "
        "`brainstorm_generic` is the identical prompt with the country removed. "
        "The Australian population weights are the target in both cases, but "
        "they are only strictly the right target for the first.",
        "",
    ]
    if missing:
        lines += [f"Not yet run, and omitted below: {', '.join(missing)}.", ""]

    # --- pooled across all models ------------------------------------------
    lines += [
        "## Pooled across all models",
        "",
        "| Condition | " + " | ".join(codes) + " | TVD from target |",
        "|---" * (len(codes) + 2) + "|",
        "| *target* | " + " | ".join(pct(expected[c]) for c in codes) + " | — |",
    ]
    pooled = {}
    for name, frame in conditions.items():
        shares = pooled_row(frame, codes)
        pooled[name] = shares
        tvd = total_variation_distance(shares, expected)
        lines.append(
            f"| `{name}` | " + " | ".join(pct(shares[c]) for c in codes)
            + f" | {tvd:.3f} |"
        )

    # The gap between conditions is itself a total variation distance: how much
    # of the distribution moves when the prompt changes.
    if len(pooled) == 2:
        first, second = list(pooled)
        shift = total_variation_distance(pooled[first], pooled[second])
        lines += [
            "",
            f"Distance between the two conditions: **{shift:.3f}** — the share "
            "of paragraphs that change discourse when the prompt stops naming "
            "Australia.",
        ]

    # --- per model ----------------------------------------------------------
    lines += [
        "",
        "## By model",
        "",
        "Whether the prompt effect is a property of the setup or of a "
        "particular model.",
        "",
        "| Model | Condition | " + " | ".join(codes) + " | TVD from target |",
        "|---" * (len(codes) + 3) + "|",
    ]
    models = sorted({model for frame in conditions.values() for model in frame.index})
    for model in models:
        for name, frame in conditions.items():
            if model not in frame.index:
                continue
            row = frame.loc[model]
            shares = " | ".join(pct(row[c]) for c in codes)
            lines.append(f"| `{model}` | `{name}` | {shares} | {row['tvd']:.3f} |")

    lines.append("")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "comparison.md"
    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
