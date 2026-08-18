"""
Step 6c: do two judges agree?

One judge cannot tell you whether a flat result is a property of the corpus or
of the rater. The first run returned `neutral` for every applicable treatment
judgement and `articulated` for 90% of presence judgements -- both constants,
and a constant is not a measurement. A second judge is the control.

This script puts the two side by side and reports:

  * pairwise agreement and Cohen's kappa on presence, per discourse;
  * where they disagree, and in which direction;
  * whether their coverage and balance conclusions survive the change of rater.

The honest outcome of a disagreement is not to pick a winner. If the coverage
figure moves when the judge changes, then coverage as measured here is a
property of the judge as much as of the essays, and it should be reported with
that range attached.

Usage:
    python src/compare_judges.py
    python src/compare_judges.py --baseline pre
"""

import argparse
import json

import numpy as np
import pandas as pd

import config
from analyse import pct, points, target_distribution, total_variation_distance
from analyse_judge import EXTENT_ORDER, PRESENCE_ORDER, load_judgements, modal
from analyse_sentences import normalised_entropy
from embed import load_discourses


def collapse(raw: pd.DataFrame) -> pd.DataFrame:
    """Majority-vote the replicates down to one verdict per (essay, discourse)."""
    verdicts = raw.groupby(["model", "repetition", "discourse"]).agg(
        presence=("presence", modal),
        treatment=("treatment", modal),
        extent=("extent", modal),
        agreement=("presence", lambda s: s.value_counts().iloc[0] / len(s)),
    ).reset_index()

    minimum = PRESENCE_ORDER[config.JUDGE_PRESENCE_MIN]
    verdicts["presence_score"] = verdicts["presence"].map(PRESENCE_ORDER)
    verdicts["extent_score"] = verdicts["extent"].map(EXTENT_ORDER)
    verdicts["available"] = (
        (verdicts["presence_score"] >= minimum)
        & (verdicts["treatment"] != "dismissed")
    )
    return verdicts


def cohens_kappa(left: pd.Series, right: pd.Series) -> float:
    """
    Chance-corrected agreement between two raters on a categorical variable.

    Raw agreement flatters a rater that always says the same thing: two judges
    that both answer "articulated" 90% of the time agree 81% of the time by
    accident alone. Kappa subtracts that. 0 is chance, 1 is perfect, and
    negative means worse than chance.
    """
    categories = sorted(set(left) | set(right))
    if len(categories) < 2:
        return float("nan")  # no variation to agree about

    observed = float((left.to_numpy() == right.to_numpy()).mean())
    expected = sum(
        (left == category).mean() * (right == category).mean()
        for category in categories
    )
    if np.isclose(expected, 1.0):
        return float("nan")
    return float((observed - expected) / (1 - expected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    args = parser.parse_args()
    prompt_name = config.apply_overrides(args)

    # Load each configured judge that actually has results on disk.
    loaded = {}
    for judge_model in config.JUDGE_MODELS:
        config.JUDGE_MODEL = judge_model
        try:
            loaded[judge_model] = collapse(load_judgements(prompt_name))
        except FileNotFoundError:
            print(f"  no judgements for {judge_model}, skipping")

    if len(loaded) < 2:
        raise SystemExit(
            "Need at least two judges with results. Run:\n"
            + "\n".join(f"  python src/judge.py --judge-model {m}"
                        for m in config.JUDGE_MODELS)
        )

    discourses = load_discourses()
    codes = [c for c in discourses["code"] if c in config.ACTIVE_DISCOURSES]
    names = dict(zip(discourses["code"], discourses["name"]))
    target = target_distribution(codes)

    judges = list(loaded)
    key = ["model", "repetition", "discourse"]
    merged = loaded[judges[0]][key + ["presence", "treatment", "extent",
                                      "extent_score", "available"]]
    merged = merged.rename(columns=lambda c: f"{c}_0" if c not in key else c)
    second = loaded[judges[1]][key + ["presence", "treatment", "extent",
                                      "extent_score", "available"]]
    merged = merged.merge(
        second.rename(columns=lambda c: f"{c}_1" if c not in key else c), on=key
    )

    n_essays = merged.groupby(["model", "repetition"]).ngroups

    lines = [
        "# Two judges, same question",
        "",
        f"Baseline: **{config.baseline()['label']}**  |  Prompt: `{prompt_name}`",
        "",
        f"- Judge 0: `{judges[0]}`",
        f"- Judge 1: `{judges[1]}`",
        "",
        f"{len(merged)} matched judgements over {n_essays} essays x "
        f"{len(codes)} discourses, each the majority of "
        f"{config.JUDGE_REPLICATES} replicates.",
        "",
        "## 1. How much do they agree?",
        "",
        "Raw agreement flatters a rater that always says the same thing, so "
        "Cohen's kappa is reported alongside: it subtracts the agreement two "
        "judges would reach by chance given how often each uses each label.",
        "",
        "| Variable | Raw agreement | Cohen's kappa |",
        "|---|---|---|",
    ]
    for field in ["presence", "treatment", "extent"]:
        left, right = merged[f"{field}_0"], merged[f"{field}_1"]
        kappa = cohens_kappa(left, right)
        kappa_text = "—" if np.isnan(kappa) else f"{kappa:+.3f}"
        lines.append(
            f"| {field} | {pct((left.to_numpy() == right.to_numpy()).mean())} | "
            f"{kappa_text} |"
        )

    availability_agreement = float(
        (merged["available_0"].to_numpy() == merged["available_1"].to_numpy()).mean()
    )
    kappa_available = cohens_kappa(
        merged["available_0"].astype(str), merged["available_1"].astype(str)
    )
    lines += [
        f"| **available** | **{pct(availability_agreement)}** | "
        + ("—" if np.isnan(kappa_available) else f"**{kappa_available:+.3f}**")
        + " |",
        "",
        "An em dash means one judge used only a single label for that variable, "
        "so there is no variation for kappa to correct against.",
    ]

    # --- 2. how each judge uses the scale ---------------------------------
    lines += [
        "",
        "## 2. How each judge uses the scale",
        "",
        "| Presence | " + " | ".join(f"`{j.split('/')[-1]}`" for j in judges) + " |",
        "|---|---|---|",
    ]
    for level in PRESENCE_ORDER:
        lines.append(
            f"| {level} | "
            + " | ".join(
                pct((merged[f"presence_{i}"] == level).mean())
                for i in range(2)
            )
            + " |"
        )

    lines += [
        "",
        "| Treatment | " + " | ".join(f"`{j.split('/')[-1]}`" for j in judges) + " |",
        "|---|---|---|",
    ]
    for level in ["endorsed", "neutral", "dismissed", "not_applicable"]:
        lines.append(
            f"| {level} | "
            + " | ".join(
                pct((merged[f"treatment_{i}"] == level).mean())
                for i in range(2)
            )
            + " |"
        )

    # Direction of disagreement: is one judge systematically more generous?
    stricter = int((merged["presence_score_0"].fillna(0)
                    > merged["presence_score_1"].fillna(0)).sum()
                   ) if "presence_score_0" in merged else None
    higher_0 = int((merged["presence_0"].map(PRESENCE_ORDER)
                    > merged["presence_1"].map(PRESENCE_ORDER)).sum())
    higher_1 = int((merged["presence_1"].map(PRESENCE_ORDER)
                    > merged["presence_0"].map(PRESENCE_ORDER)).sum())
    lines += [
        "",
        f"Where they differ on presence, `{judges[0].split('/')[-1]}` rates it "
        f"higher {higher_0} times and `{judges[1].split('/')[-1]}` rates it "
        f"higher {higher_1} times. Disagreement is "
        + ("**one-directional**: one judge is simply more generous than the "
           "other, rather than the two disagreeing case by case."
           if min(higher_0, higher_1) < 0.2 * max(higher_0, higher_1)
           else "spread in both directions, so the two are reading individual "
                "essays differently rather than applying a uniformly different "
                "threshold."),
    ]

    # --- 3. does the conclusion survive? -----------------------------------
    lines += [
        "",
        "## 3. Does the conclusion survive the change of judge?",
        "",
        "### Availability per discourse",
        "",
        "| Discourse | " + " | ".join(f"`{j.split('/')[-1]}`" for j in judges)
        + " | Agreement | Kappa |",
        "|---|---|---|---|---|",
    ]
    for code in codes:
        subset = merged[merged["discourse"] == code]
        agreement = float(
            (subset["available_0"].to_numpy() == subset["available_1"].to_numpy()).mean()
        )
        kappa = cohens_kappa(subset["presence_0"], subset["presence_1"])
        lines.append(
            f"| {code} {names.get(code, '')} | "
            f"{int(subset['available_0'].sum())}/{len(subset)} | "
            f"{int(subset['available_1'].sum())}/{len(subset)} | "
            f"{pct(agreement)} | "
            + ("—" if np.isnan(kappa) else f"{kappa:+.3f}") + " |"
        )

    coverage = {}
    for index, judge_model in enumerate(judges):
        per_essay = merged.pivot_table(
            index=["model", "repetition"], columns="discourse",
            values=f"available_{index}", aggfunc="first",
        ).reindex(columns=codes).fillna(False)
        coverage[judge_model] = per_essay.sum(axis=1)

    lines += [
        "",
        "### Coverage",
        "",
        "| Judge | Mean coverage | Full coverage | Range |",
        "|---|---|---|---|",
    ]
    for judge_model in judges:
        series = coverage[judge_model]
        lines.append(
            f"| `{judge_model}` | {series.mean():.2f} / {len(codes)} | "
            f"{int((series == len(codes)).sum())}/{len(series)} | "
            f"{int(series.min())}–{int(series.max())} |"
        )

    low = min(coverage[j].mean() for j in judges)
    high = max(coverage[j].mean() for j in judges)
    lines += [
        "",
        f"**Coverage ranges from {low:.2f} to {high:.2f} of {len(codes)} "
        f"depending on which model is asked.**",
        "",
        ("That spread is wider than any difference between the three models "
         "under test, which means coverage as measured here is at least as much "
         "a property of the judge as of the essays. It has to be reported as a "
         "range, with the rubric that produced it."
         if high - low > 0.5 else
         "The two judges land close enough that the coverage figure can be "
         "reported as a single number with the spread as its uncertainty."),
    ]

    # --- 4. by model under test --------------------------------------------
    lines += [
        "",
        "### Do the judges rank the three models the same way?",
        "",
        "This is the comparison the experiment is actually for. Absolute levels "
        "can move with the rater as long as the ordering holds.",
        "",
        "| Model under test | " + " | ".join(f"`{j.split('/')[-1]}`" for j in judges)
        + " |",
        "|---|---|---|",
    ]
    for model in sorted(merged["model"].unique()):
        cells = []
        for judge_model in judges:
            series = coverage[judge_model]
            cells.append(f"{series[series.index.get_level_values('model') == model].mean():.2f}")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")

    rankings = {}
    for judge_model in judges:
        series = coverage[judge_model]
        by_model = series.groupby(level="model").mean().sort_values(ascending=False)
        rankings[judge_model] = list(by_model.index)
    same_order = len({tuple(v) for v in rankings.values()}) == 1
    lines += [
        "",
        (f"**The two judges rank the three models identically**: "
         f"{' > '.join(m.split('/')[-1] for m in rankings[judges[0]])}. "
         "The comparative claim survives the change of rater even where the "
         "absolute levels do not."
         if same_order else
         "**The two judges rank the models differently.** "
         + "  ".join(
             f"`{j.split('/')[-1]}`: "
             + " > ".join(m.split('/')[-1] for m in rankings[j]) for j in judges)
         + ". Nothing comparative survives here without a third opinion."),
    ]

    # --- 5. balance ---------------------------------------------------------
    lines += [
        "",
        "### Balance",
        "",
        "| Discourse | " + " | ".join(f"`{j.split('/')[-1]}`" for j in judges)
        + " | Reference |",
        "|---|---|---|---|",
    ]
    shares = {}
    for index, judge_model in enumerate(judges):
        totals = merged.groupby("discourse")[f"extent_score_{index}"].sum()
        totals = totals.reindex(codes).fillna(0)
        shares[judge_model] = totals / totals.sum() if totals.sum() else totals
    for code in codes:
        lines.append(
            f"| {code} | "
            + " | ".join(pct(shares[j][code]) for j in judges)
            + f" | {pct(target[code])} |"
        )
    lines += [
        "",
        "| Judge | TVD from reference | Evenness |",
        "|---|---|---|",
    ]
    for judge_model in judges:
        lines.append(
            f"| `{judge_model}` | "
            f"{total_variation_distance(shares[judge_model], target):.3f} | "
            f"{normalised_entropy(shares[judge_model]):.3f} |"
        )

    if len(codes) > 2:
        correlation = float(np.corrcoef(
            shares[judges[0]].reindex(codes), shares[judges[1]].reindex(codes)
        )[0, 1])
        lines += [
            "",
            f"Correlation between the two judges' share vectors: "
            f"**r = {correlation:+.3f}**.",
        ]

    lines += [
        "",
        "## Caveats",
        "",
        "- Two judges is the minimum for this comparison, not a sufficient "
        "number. Where they disagree, nothing here identifies which is right.",
        "- Both judges saw an identical prompt and identical blinded "
        "descriptions, so any shared bias from the rubric affects both and is "
        "invisible to this comparison. Agreement is evidence against rater "
        "idiosyncrasy, not against a badly framed question.",
        "- Kappa is undefined where a judge used a single label throughout. "
        "That is itself the finding for that variable.",
        "",
    ]

    out_dir = config.results_dir(prompt_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "judge_comparison.csv", index=False)
    summary = "\n".join(lines)
    (out_dir / "judge_comparison.md").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"\nWritten to {out_dir}/judge_comparison.md")


if __name__ == "__main__":
    main()
