"""
Step 4: measure how the essays represent the human discourses.

Every paragraph is compared by cosine similarity against each active discourse
description and assigned to whichever it resembles most. The share of paragraphs
going to each discourse is the model's *observed* representation, judged against
a reference distribution that depends on the baseline (see target_distribution).

Four deliberate choices, all revisable in config.py:

  * Which discourses compete is set by the baseline. The pre-deliberative map
    excludes D, whose expected share in the Australian population is exactly
    zero and so is not a proportion any model can be scored against. The
    post-deliberative map keeps all six.

  * Assignment is winner-takes-all over the remaining cosine similarities. A
    margin threshold can require the winner to lead the runner-up by some
    amount, leaving ambiguous paragraphs "unassigned" -- the analogue of the 12%
    of survey respondents with a confounded loading. The primary run uses a
    margin of 0, so every paragraph counts.

  * Proportions are computed over assigned paragraphs only. This matches the
    weights, which are themselves conditional on loading on some discourse.

  * The headline distance is total variation distance: bounded, in the same
    units as the shares themselves, and readable directly as "the share of
    paragraphs that would have to move to make the essay proportional".

Alongside the raw result the script reports a *centred* variant. Cosine
similarity between any paragraph about genome editing and any of the four
baselines is dominated by shared topic vocabulary -- the baselines sit at ~0.85
cosine from each other -- and on top of that common component some baselines are
generically more attractive than others. Centring each discourse's similarity
column on its own mean across the corpus removes that per-baseline offset before
the argmax. It does not force the proportions to be equal; it asks which
discourse a paragraph is *unusually* close to, rather than which is closest.

Usage:
    python src/analyse.py
    python src/analyse.py --baseline pre
    python src/analyse.py --prompt brainstorm_generic
"""

import argparse

import numpy as np
import pandas as pd

import config


def target_distribution(codes: list[str]) -> pd.Series:
    """
    The distribution the observed shares are judged against.

    Which one depends on the baseline, and the difference is theoretical rather
    than technical:

      * "population" -- the pre-deliberative map's observed prevalence in the
        Australian population. Asks whether the essays mirror raw opinion.

      * "uniform" -- an even split. Used for the six post-deliberative
        discourses, where the criterion is not prevalence but availability: a
        brainstorming tool should not systematically marginalise any relevant
        way of reasoning. The uniform reference is a yardstick for evenness, not
        a claim that the six are equally common in the population.

    Either way the result is renormalised over the active discourses, so
    expected and observed are always distributions over the same categories.
    """
    kind = config.baseline()["target"]

    if kind == "uniform":
        return pd.Series(1.0 / len(codes), index=codes)

    if kind == "population":
        weights = pd.read_csv(config.WEIGHTS_FILE).set_index("discourse")
        target = weights["weight_all_loadings"].reindex(codes).astype(float)
        return target / target.sum()

    raise ValueError(f"Unknown target kind {kind!r} for baseline {config.BASELINE!r}")


def load_inputs():
    """Load paragraphs, embeddings, discourse baselines and the target distribution."""
    prompt_name = config.PROMPT_NAME
    processed = config.PROCESSED_DIR / prompt_name

    paragraphs = pd.read_csv(processed / "paragraphs.csv")
    paragraph_embeddings = np.load(processed / "paragraph_embeddings.npy")
    discourses = pd.read_csv(config.discourse_table_path())
    discourse_embeddings = np.load(config.discourse_embeddings_path())

    if len(paragraphs) != len(paragraph_embeddings):
        raise ValueError(
            f"{len(paragraphs)} paragraphs but {len(paragraph_embeddings)} "
            "embeddings. Re-run src/embed.py."
        )

    # All four baselines stay loaded. Excluded discourses are dropped later, at
    # the point of assignment, so that the per-paragraph table still records how
    # similar each paragraph was to them.
    missing = set(config.ACTIVE_DISCOURSES) - set(discourses["code"])
    if missing:
        raise ValueError(
            f"config.ACTIVE_DISCOURSES names {sorted(missing)}, which is not in "
            f"the baseline file. Available: {list(discourses['code'])}"
        )
    excluded = [c for c in discourses["code"] if c not in config.ACTIVE_DISCOURSES]

    return (paragraphs, paragraph_embeddings, discourses, discourse_embeddings,
            excluded)


def assign(similarities: pd.DataFrame, margin: float) -> pd.DataFrame:
    """
    Assign each paragraph to its most similar discourse.

    `similarities` has one column per discourse code. Returns a frame with the
    winning code, the runner-up margin, and the assignment after applying the
    margin threshold ("unassigned" when the top two are too close).
    """
    codes = list(similarities.columns)
    values = similarities.to_numpy()

    # Sort each row descending to find the winner and its lead over second place.
    order = np.argsort(-values, axis=1)
    top_index = order[:, 0]
    top_value = values[np.arange(len(values)), order[:, 0]]
    second_value = values[np.arange(len(values)), order[:, 1]]

    top_code = np.array(codes)[top_index]
    lead = top_value - second_value

    return pd.DataFrame({
        "top_discourse": top_code,
        "top_similarity": top_value,
        "runner_up_margin": lead,
        "assigned": np.where(lead >= margin, top_code, "unassigned"),
    }, index=similarities.index)


def proportions(assigned: pd.Series, codes: list[str]) -> pd.Series:
    """Share of assigned paragraphs falling to each discourse (unassigned excluded)."""
    counted = assigned[assigned != "unassigned"]
    if counted.empty:
        return pd.Series(np.nan, index=codes)
    return counted.value_counts(normalize=True).reindex(codes).fillna(0.0)


def total_variation_distance(observed: pd.Series, expected: pd.Series) -> float:
    """
    Half the sum of absolute differences between two distributions.

    0 means perfectly proportional representation; 1 means no overlap at all.
    Chosen because it stays defined when an expected share is zero, which is the
    case for discourse D.
    """
    return float(0.5 * np.abs(observed - expected).sum())


def pct(value: float) -> str:
    """Format a share as a percentage, showing an em dash when it is undefined."""
    if pd.isna(value):
        return "—"
    return f"{value:.1%}"


def points(value: float) -> str:
    """
    Format the difference between two shares in percentage points.

    Kept distinct from pct() on purpose: a gap between two percentages is
    measured in points, not per cent, and writing "-17.1%" for a 17.1-point
    shortfall invites the reader to take it as a relative change.
    """
    if pd.isna(value):
        return "—"
    return f"{value * 100:+.1f}"


def build_summary(results: dict, discourses: pd.DataFrame, weights: pd.Series,
                  paragraphs: pd.DataFrame, excluded: list[str]) -> str:
    """Render the analysis as a readable markdown report."""
    codes = list(weights.index)
    names = dict(zip(discourses["code"], discourses["name"]))

    settings = config.baseline()
    target_heading = (
        "Expected representation (Australian population)"
        if settings["target"] == "population"
        else "Reference distribution (even split)"
    )
    target_note = (
        "Observed prevalence of each discourse in the Australian population, "
        "renormalised over the active discourses."
        if settings["target"] == "population"
        else "The six post-deliberative discourses have no usable population "
             "weights, and on the brainstorming criterion they should not be "
             "judged by prevalence anyway. The reference is an even split: a "
             "yardstick for whether any way of reasoning is systematically "
             "marginalised, not a claim that the six are equally common."
    )

    lines = [
        "# Discursive representation results",
        "",
        f"Baseline: **{settings['label']}**",
        "",
        f"Prompt: `{config.PROMPT_NAME}`  |  "
        f"Embedding model: `{config.EMBEDDING_MODEL}`  |  "
        f"Assignment margin: {config.ASSIGNMENT_MARGIN}",
        "",
        f"{len(paragraphs)} paragraphs from "
        f"{paragraphs['run_id'].nunique()} essays across "
        f"{paragraphs['model'].nunique()} models.",
        "",
        f"## {target_heading}",
        "",
        target_note,
        "",
        "| Discourse | Name | Expected share |",
        "|---|---|---|",
    ]
    for code in codes:
        lines.append(f"| {code} | {names.get(code, '')} | {weights[code]:.1%} |")

    if excluded:
        lines += [
            "",
            f"Discourse {', '.join(excluded)} excluded from the analysis: "
            "expected share of zero in the Australian population. Paragraphs "
            "that most resemble an excluded discourse are assigned to their "
            "nearest surviving one rather than being set aside.",
            "",
            f"**{pct(results['excluded_share'])} of paragraphs** would have gone "
            f"to {', '.join(excluded)} had it stayed in the running. That is how "
            "much of the distribution below is redistributed by the exclusion.",
        ]

    lines += [
        "",
        "## Observed representation, by model",
        "",
        "| Model | " + " | ".join(f"{c}" for c in codes) + " | TVD from target |",
        "|---" * (len(codes) + 2) + "|",
    ]
    for model, row in results["by_model"].iterrows():
        shares = " | ".join(pct(row[c]) for c in codes)
        lines.append(f"| `{model}` | {shares} | {row['tvd']:.3f} |")

    lines += [
        "",
        "Deviation from the expected share, in percentage points "
        "(positive = over-represented):",
        "",
        "| Model | " + " | ".join(f"{c} (pp)" for c in codes) + " |",
        "|---" * (len(codes) + 1) + "|",
    ]
    for model, row in results["by_model"].iterrows():
        deviations = " | ".join(points(row[c] - weights[c]) for c in codes)
        lines.append(f"| `{model}` | {deviations} |")

    lines += [
        "",
        "## Spread across the ten repetitions",
        "",
        "Standard deviation of each model's per-essay shares. Large values mean "
        "the model's representation is unstable from one essay to the next, which "
        "matters as much as the average for a brainstorming interface.",
        "",
        "| Model | " + " | ".join(codes) + " |",
        "|---" * (len(codes) + 1) + "|",
    ]
    spread = results["by_run"].groupby("model")[codes].std()
    for model, row in spread.iterrows():
        lines.append(f"| `{model}` | " + " | ".join(f"{row[c]:.3f}" for c in codes) + " |")

    lines += [
        "",
        "## Mean cosine similarity, by model",
        "",
        "Assignment is winner-takes-all, so it hides how close the contest was. "
        "These are the raw mean similarities to each baseline.",
        "",
        "| Model | " + " | ".join(codes) + " |",
        "|---" * (len(codes) + 1) + "|",
    ]
    for model, row in results["mean_similarity"].iterrows():
        lines.append(f"| `{model}` | " + " | ".join(f"{row[c]:.3f}" for c in codes) + " |")

    lines += [
        "",
        "## Robustness: centred similarities",
        "",
        "The same proportions after each discourse's similarity column is centred "
        "on its corpus mean, removing the advantage of a baseline that is "
        "generically close to everything. If these differ sharply from the "
        "headline table, the headline is being driven by baseline attractiveness "
        "rather than by discourse content.",
        "",
        "| Model | " + " | ".join(codes) + " | TVD from target |",
        "|---" * (len(codes) + 2) + "|",
    ]
    for model, row in results["by_model_centred"].iterrows():
        shares = " | ".join(pct(row[c]) for c in codes)
        lines.append(f"| `{model}` | {shares} | {row['tvd']:.3f} |")

    lines += [
        "",
        "## Sensitivity to the assignment margin",
        "",
        "| Margin | Unassigned | " + " | ".join(codes) + " |",
        "|---" * (len(codes) + 2) + "|",
    ]
    for _, row in results["sensitivity"].iterrows():
        shares = " | ".join(pct(row[c]) for c in codes)
        lines.append(
            f"| {row['margin']:.3f} | {pct(row['unassigned_share'])} | {shares} |"
        )

    lines += [
        "",
        "## Caveats",
        "",
        (f"- Excluding a discourse redistributes rather than removes its "
         f"paragraphs: anything closest to {', '.join(excluded)} now counts "
         f"towards {', '.join(codes)}. `paragraph_similarities.csv` keeps the "
         "raw similarity to every baseline, so the size of that effect is "
         "recoverable.") if excluded else None,
        "- The baselines are not equally distinctive. Check the "
        "baseline-to-baseline similarities printed by `src/embed.py`: if two "
        "baselines are very close, the split between them is not reliable.",
        "- Winner-takes-all assignment turns small similarity differences into "
        "whole paragraphs. The margin sensitivity table above is the check on that.",
        "",
    ]
    # Some caveats are baseline-specific and render as None when they do not apply.
    return "\n".join(line for line in lines if line is not None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    config.apply_overrides(parser.parse_args())

    (paragraphs, paragraph_emb, discourses, discourse_emb,
     excluded) = load_inputs()

    all_codes = discourses["code"].tolist()
    codes = [c for c in all_codes if c in config.ACTIVE_DISCOURSES]
    weights = target_distribution(codes)

    # Both embedding sets are unit-normalised, so the dot product is the cosine.
    # similarity_all keeps every baseline for the record; `similarities` is the
    # subset the assignment actually competes over.
    similarity_all = pd.DataFrame(paragraph_emb @ discourse_emb.T, columns=all_codes)
    similarities = similarity_all[codes]

    # Centred similarities: each discourse column minus its own corpus mean, so
    # that a baseline which is generically close to everything loses that
    # advantage. Used for the robustness check, not the headline number.
    centred = similarities - similarities.mean(axis=0)

    # --- per-paragraph table ---------------------------------------------
    assignment = assign(similarities, config.ASSIGNMENT_MARGIN)
    centred_assignment = assign(centred, config.ASSIGNMENT_MARGIN)[["assigned"]]
    centred_assignment.columns = ["assigned_centred"]

    per_paragraph = pd.concat(
        [paragraphs.reset_index(drop=True), similarity_all.add_prefix("sim_"),
         assignment, centred_assignment],
        axis=1,
    )

    # --- proportions by model and by essay --------------------------------
    by_model_rows = {}
    for model, group in per_paragraph.groupby("model"):
        shares = proportions(group["assigned"], codes)
        shares["tvd"] = total_variation_distance(shares[codes], weights)
        shares["n_paragraphs"] = len(group)
        shares["unassigned_share"] = (group["assigned"] == "unassigned").mean()
        by_model_rows[model] = shares
    by_model = pd.DataFrame(by_model_rows).T

    by_run_rows = []
    for (model, run_id), group in per_paragraph.groupby(["model", "run_id"]):
        shares = proportions(group["assigned"], codes)
        by_run_rows.append({
            "model": model, "run_id": run_id, "n_paragraphs": len(group),
            **shares.to_dict(),
            "tvd": total_variation_distance(shares, weights),
        })
    by_run = pd.DataFrame(by_run_rows)

    # Robustness: the same proportions computed on centred similarities.
    centred_rows = {}
    for model, group in per_paragraph.groupby("model"):
        shares = proportions(group["assigned_centred"], codes)
        shares["tvd"] = total_variation_distance(shares[codes], weights)
        centred_rows[model] = shares
    by_model_centred = pd.DataFrame(centred_rows).T

    mean_similarity = per_paragraph.groupby("model")[
        [f"sim_{c}" for c in codes]
    ].mean()
    mean_similarity.columns = codes

    # How often an excluded discourse would have won had it stayed in the
    # running. This is the exact size of the redistribution the exclusion
    # causes, and it belongs in the report rather than in a footnote.
    excluded_share = (
        float(similarity_all.idxmax(axis=1).isin(excluded).mean())
        if excluded else 0.0
    )

    # --- sensitivity to the margin threshold ------------------------------
    sensitivity_rows = []
    for margin in config.SENSITIVITY_MARGINS:
        alternative = assign(similarities, margin)
        shares = proportions(alternative["assigned"], codes)
        sensitivity_rows.append({
            "margin": margin,
            "unassigned_share": (alternative["assigned"] == "unassigned").mean(),
            **shares.to_dict(),
            "tvd": total_variation_distance(shares, weights),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)

    # --- write everything out ---------------------------------------------
    out_dir = config.results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_paragraph.to_csv(out_dir / "paragraph_similarities.csv", index=False)
    by_model.to_csv(out_dir / "proportions_by_model.csv")
    by_run.to_csv(out_dir / "proportions_by_run.csv", index=False)
    sensitivity.to_csv(out_dir / "sensitivity_margin.csv", index=False)

    by_model_centred.to_csv(out_dir / "proportions_by_model_centred.csv")

    results = {
        "by_model": by_model, "by_run": by_run,
        "by_model_centred": by_model_centred,
        "mean_similarity": mean_similarity, "sensitivity": sensitivity,
        "excluded_share": excluded_share,
    }
    summary = build_summary(results, discourses, weights, paragraphs, excluded)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"\nWritten to {out_dir}/")


if __name__ == "__main__":
    main()
