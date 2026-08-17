"""
Step 4: measure how the essays represent the four discourses.

Every paragraph is compared by cosine similarity against each active discourse
description and assigned to whichever it resembles most. The share of paragraphs
going to each discourse is the model's *observed* representation; the Australian
population weights are the *expected* representation. The gap between them is
the result.

Four deliberate choices, all revisable in config.py:

  * Discourse D is excluded (config.ACTIVE_DISCOURSES). Its expected share in
    the Australian population is exactly zero, which is not a proportion any
    model can be scored against. The analysis therefore runs over A, B and C,
    whose expected shares already account for the whole distribution.

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
    python src/analyse.py --prompt brainstorm_generic
"""

import argparse

import numpy as np
import pandas as pd

import config


def load_inputs():
    """Load paragraphs, embeddings, discourse baselines and target weights."""
    prompt_name = config.PROMPT_NAME
    processed = config.PROCESSED_DIR / prompt_name

    paragraphs = pd.read_csv(processed / "paragraphs.csv")
    paragraph_embeddings = np.load(processed / "paragraph_embeddings.npy")
    discourses = pd.read_csv(config.PROCESSED_DIR / "discourses.csv")
    discourse_embeddings = np.load(config.PROCESSED_DIR / "discourse_embeddings.npy")
    weights = pd.read_csv(config.WEIGHTS_FILE)

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
            weights, excluded)


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


def pct(value: float, signed: bool = False) -> str:
    """Format a share as a percentage, showing an em dash when it is undefined."""
    if pd.isna(value):
        return "—"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def build_summary(results: dict, discourses: pd.DataFrame, weights: pd.Series,
                  paragraphs: pd.DataFrame, excluded: list[str]) -> str:
    """Render the analysis as a readable markdown report."""
    codes = list(weights.index)
    names = dict(zip(discourses["code"], discourses["name"]))

    lines = [
        "# Discursive representation results",
        "",
        f"Prompt: `{config.PROMPT_NAME}`  |  "
        f"Embedding model: `{config.EMBEDDING_MODEL}`  |  "
        f"Assignment margin: {config.ASSIGNMENT_MARGIN}",
        "",
        f"{len(paragraphs)} paragraphs from "
        f"{paragraphs['run_id'].nunique()} essays across "
        f"{paragraphs['model'].nunique()} models.",
        "",
        "## Expected representation (Australian population)",
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
        "Deviation from the expected share (positive = over-represented):",
        "",
        "| Model | " + " | ".join(codes) + " |",
        "|---" * (len(codes) + 1) + "|",
    ]
    for model, row in results["by_model"].iterrows():
        deviations = " | ".join(pct(row[c] - weights[c], signed=True) for c in codes)
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
        "- Excluding a discourse redistributes rather than removes its "
        "paragraphs: anything closest to D now counts towards A, B or C. If the "
        "models voice Agnosticism often, that inflates whichever discourse sits "
        "nearest to it in embedding space. `paragraph_similarities.csv` keeps "
        "the raw similarity to every baseline, so the size of that effect is "
        "recoverable.",
        "- The baselines are not equally distinctive. Check the "
        "baseline-to-baseline similarities printed by `src/embed.py`: if two "
        "baselines are very close, the split between them is not reliable.",
        "- Winner-takes-all assignment turns small similarity differences into "
        "whole paragraphs. The margin sensitivity table above is the check on that.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    config.apply_overrides(parser.parse_args())

    (paragraphs, paragraph_emb, discourses, discourse_emb,
     weights_table, excluded) = load_inputs()

    all_codes = discourses["code"].tolist()
    codes = [c for c in all_codes if c in config.ACTIVE_DISCOURSES]
    weights = weights_table.set_index("discourse")["weight_all_loadings"].reindex(codes)

    # Renormalise the target over the active discourses, so that expected and
    # observed are distributions over exactly the same categories. With D
    # excluded this changes nothing -- D's expected share is zero, so A, B and C
    # already sum to 1 -- but it keeps the comparison correct if the active set
    # ever changes.
    weights = weights / weights.sum()

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
    out_dir = config.RESULTS_DIR / config.PROMPT_NAME
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
