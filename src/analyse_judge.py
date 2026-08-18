"""
Step 6b: summarise the judge's verdicts, and check the judge itself.

The judge answers the four brainstorming questions directly, without argmax,
without a presence threshold, and without a proxy for voicing:

  EXISTENCE   -- is each discourse recovered anywhere in the corpus
  COVERAGE    -- how many does a single essay make available
  RELIABILITY -- how consistently, across the ten runs per model
  BALANCE     -- how much room each one gets

But a judge is an instrument like any other, and the whole point of this
project has been not to trust instruments that have not been tested. Section 5
therefore turns the measurement on the judge: how often do independent
replicates of the same judgement agree, how often is the supporting quote
actually in the essay, and how often do two different discourses get supported
by the same passage.

Usage:
    python src/analyse_judge.py
    python src/analyse_judge.py --baseline pre
"""

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

import config
from analyse import pct, points, target_distribution, total_variation_distance
from analyse_sentences import normalised_entropy
from embed import load_discourses

# Presence is ordered, so it can be averaged and thresholded.
PRESENCE_ORDER = {"absent": 0, "mentioned": 1, "articulated": 2}
EXTENT_ORDER = {
    "none": 0, "a sentence or two": 1, "a paragraph": 2, "several paragraphs": 3,
}


def load_judgements(prompt_name: str) -> pd.DataFrame:
    """
    Read every cached judgement made by the active judge on the active baseline.

    Rows whose response could not be read are dropped here rather than counted:
    a parse failure is missing data, and letting it stand as a rating would let
    truncated responses masquerade as findings. The count is returned alongside
    so the report can state how much was lost.
    """
    directory = (config.PROCESSED_DIR / prompt_name / "judge"
                 / config.BASELINE / config.judge_slug())
    if not directory.exists():
        raise FileNotFoundError(
            f"No judgements at {directory}. Run: python src/judge.py "
            f"--baseline {config.BASELINE} --judge-model {config.JUDGE_MODEL}"
        )
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(directory.glob("*.json"))]
    if not records:
        raise FileNotFoundError(f"{directory} is empty.")

    frame = pd.DataFrame(records)
    unreadable = int((frame["presence"] == "error").sum())
    if unreadable:
        print(f"  dropping {unreadable} unreadable judgement(s) of {len(frame)}")
    frame.attrs["unreadable"] = unreadable
    frame.attrs["total_calls"] = len(frame)
    return frame[frame["presence"] != "error"].reset_index(drop=True)


def modal(values: pd.Series) -> str:
    """
    Most common value, ties broken toward the more conservative reading.

    With three replicates a 1-1-1 split is possible; taking the lowest presence
    rating in that case means the judge has to actually agree before a discourse
    is credited.
    """
    counts = Counter(values)
    best = max(counts.values())
    tied = [v for v, n in counts.items() if n == best]
    if len(tied) == 1:
        return tied[0]
    if all(t in PRESENCE_ORDER for t in tied):
        return min(tied, key=lambda t: PRESENCE_ORDER[t])
    if all(t in EXTENT_ORDER for t in tied):
        return min(tied, key=lambda t: EXTENT_ORDER[t])
    return sorted(tied)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    config.add_judge_argument(parser)
    args = parser.parse_args()
    prompt_name = config.apply_overrides(args)
    config.apply_judge_override(args)

    raw = load_judgements(prompt_name)

    discourses = load_discourses()
    codes = [c for c in discourses["code"] if c in config.ACTIVE_DISCOURSES]
    names = dict(zip(discourses["code"], discourses["name"]))
    target = target_distribution(codes)

    # --- collapse replicates to one verdict per (essay, discourse) ---------
    grouped = raw.groupby(["model", "repetition", "discourse"])
    verdicts = grouped.agg(
        presence=("presence", modal),
        treatment=("treatment", modal),
        extent=("extent", modal),
        replicates=("replicate", "count"),
        presence_agreement=("presence", lambda s: s.value_counts().iloc[0] / len(s)),
        treatment_agreement=("treatment", lambda s: s.value_counts().iloc[0] / len(s)),
        quote_verified=("quote_verified", "mean"),
        quote_match=("quote_match", "mean"),
        parse_failures=("parse_error", lambda s: (s.astype(str) != "").sum()),
    ).reset_index()

    verdicts["presence_score"] = verdicts["presence"].map(PRESENCE_ORDER)
    verdicts["extent_score"] = verdicts["extent"].map(EXTENT_ORDER)
    minimum = PRESENCE_ORDER[config.JUDGE_PRESENCE_MIN]
    verdicts["available"] = (
        (verdicts["presence_score"] >= minimum)
        & (verdicts["treatment"] != "dismissed")
    )
    verdicts["present_any"] = verdicts["presence_score"] >= PRESENCE_ORDER["mentioned"]

    n_essays = verdicts.groupby(["model", "repetition"]).ngroups

    lines = [
        "# LLM-as-judge: is each way of reasoning available?",
        "",
        f"Baseline: **{config.baseline()['label']}**",
        "",
        f"Judge: `{config.JUDGE_MODEL}` @ temperature {config.JUDGE_TEMPERATURE}  |  "
        f"Prompt: `{prompt_name}`  |  {len(raw)} judgements over "
        f"{n_essays} essays x {len(codes)} discourses x "
        f"{config.JUDGE_REPLICATES} replicates"
        + (f"  |  **{raw.attrs.get('unreadable', 0)} unreadable responses "
           f"dropped** of {raw.attrs.get('total_calls', len(raw))}"
           if raw.attrs.get("unreadable") else ""),
        "",
        "The judge sees one unlabelled discourse description and one essay, and "
        "rates how developed that way of reasoning is (**presence**), how the "
        "essay positions it (**treatment**) and how much room it gets "
        "(**extent**). Descriptions are blinded — position letters and names are "
        "stripped — so the judge reads the reasoning rather than a label. "
        "Replicates are collapsed by majority vote, ties broken toward the more "
        "conservative rating.",
        "",
        f"A discourse counts as **available** when presence reaches "
        f"`{config.JUDGE_PRESENCE_MIN}` and treatment is not `dismissed`.",
        "",
        "## 1. Existence — what the judge finds, corpus-wide",
        "",
        "| Discourse | Articulated | Mentioned only | Absent | Dismissed | Available |",
        "|---|---|---|---|---|---|",
    ]
    for code in codes:
        subset = verdicts[verdicts["discourse"] == code]
        counts = subset["presence"].value_counts()
        lines.append(
            f"| {code} {names.get(code, '')} | "
            f"{int(counts.get('articulated', 0))}/{len(subset)} | "
            f"{int(counts.get('mentioned', 0))}/{len(subset)} | "
            f"{int(counts.get('absent', 0))}/{len(subset)} | "
            f"{int((subset['treatment'] == 'dismissed').sum())}/{len(subset)} | "
            f"**{int(subset['available'].sum())}/{len(subset)}** |"
        )

    never = [c for c in codes
             if not verdicts[verdicts["discourse"] == c]["available"].any()]
    lines += [
        "",
        f"**{len(codes) - len(never)} of {len(codes)} discourses are available "
        "in at least one essay**"
        + (f"; {', '.join(never)} never reach the bar." if never
           else " — every one is recovered somewhere."),
    ]

    # --- 2. coverage and reliability ---------------------------------------
    per_essay = verdicts.pivot_table(
        index=["model", "repetition"], columns="discourse",
        values="available", aggfunc="first",
    ).reindex(columns=codes).fillna(False)
    coverage = per_essay.sum(axis=1)

    lines += [
        "",
        "## 2. Coverage — how much of the space one essay opens",
        "",
        "| Model | Mean coverage | Worst | Best | Full coverage |",
        "|---|---|---|---|---|",
    ]
    for model, group in coverage.groupby(level="model"):
        lines.append(
            f"| `{model}` | {group.mean():.1f} / {len(codes)} | "
            f"{int(group.min())} | {int(group.max())} | "
            f"{int((group == len(codes)).sum())} / {len(group)} |"
        )
    lines.append(
        f"| **pooled** | **{coverage.mean():.1f} / {len(codes)}** | "
        f"{int(coverage.min())} | {int(coverage.max())} | "
        f"**{int((coverage == len(codes)).sum())} / {len(coverage)}** |"
    )

    lines += [
        "",
        "### Reliability — how often each discourse is available",
        "",
        "| Model | " + " | ".join(codes) + " |",
        "|---" * (len(codes) + 1) + "|",
    ]
    for model, group in per_essay.groupby(level="model"):
        lines.append(
            f"| `{model}` | "
            + " | ".join(f"{int(group[c].sum())}/{len(group)}" for c in codes)
            + " |"
        )
    lines.append(
        "| **pooled** | "
        + " | ".join(f"{int(per_essay[c].sum())}/{len(per_essay)}" for c in codes)
        + " |"
    )

    lines += [
        "",
        "Distribution of coverage:",
        "",
        "| Discourses available | Essays |",
        "|---|---|",
    ]
    for level, count in coverage.value_counts().sort_index(ascending=False).items():
        lines.append(f"| {int(level)} / {len(codes)} | {int(count)} |")

    # --- 3. balance ---------------------------------------------------------
    # Extent is an ordinal amount of room, so normalising it across discourses
    # gives a share that owes nothing to argmax.
    extent = verdicts.pivot_table(
        index=["model", "repetition"], columns="discourse",
        values="extent_score", aggfunc="first",
    ).reindex(columns=codes).fillna(0)
    shares = extent.sum(axis=0) / extent.to_numpy().sum()
    uniform = 1.0 / len(codes)
    floor = config.MARGINALISATION_RATIO * uniform
    marginalised = [c for c in codes if shares[c] < floor]

    lines += [
        "",
        "## 3. Balance — how much room each discourse gets",
        "",
        "Share of total *extent*, where each judgement contributes 0 for none up "
        "to 3 for several paragraphs. Unlike the cosine measure, nothing here "
        "depends on assigning each sentence to exactly one discourse, so a "
        "passage that serves two perspectives can count for both.",
        "",
        "| Discourse | Share of extent | Reference | Deviation (pp) |",
        "|---|---|---|---|",
    ]
    for code in codes:
        marker = " ⚠" if shares[code] < floor else ""
        lines.append(
            f"| {code} | {pct(shares[code])}{marker} | {pct(target[code])} | "
            f"{points(shares[code] - target[code])} |"
        )
    lines += [
        "",
        f"**TVD from reference: {total_variation_distance(shares, target):.3f}**  |  "
        f"Evenness: {normalised_entropy(shares):.3f}  |  "
        f"{len(marginalised)} of {len(codes)} marginalised"
        + (f": {', '.join(marginalised)}" if marginalised else ""),
        "",
        "### Treatment — how the essays position each discourse",
        "",
        "| Discourse | Endorsed | Neutral | Dismissed | Not applicable |",
        "|---|---|---|---|---|",
    ]
    for code in codes:
        subset = verdicts[verdicts["discourse"] == code]
        counts = subset["treatment"].value_counts()
        lines.append(
            f"| {code} | {int(counts.get('endorsed', 0))} | "
            f"{int(counts.get('neutral', 0))} | "
            f"{int(counts.get('dismissed', 0))} | "
            f"{int(counts.get('not_applicable', 0))} |"
        )

    # --- 4. against the cosine pipeline -------------------------------------
    cosine_path = (config.results_dir(prompt_name)
                   / "essay_coverage_depersonalised.csv")
    if cosine_path.exists():
        cosine = pd.read_csv(cosine_path)
        lines += [
            "",
            "## 4. Judge against the cosine pipeline",
            "",
            "Two independent measurements of the same thing. Where they agree, "
            "the finding is not an artefact of either instrument.",
            "",
            "| Discourse | Judge: available | Cosine: available |",
            "|---|---|---|",
        ]
        for code in codes:
            cosine_present = int(cosine[f"present_{code}"].sum()) if \
                f"present_{code}" in cosine else None
            lines.append(
                f"| {code} | {int(per_essay[code].sum())}/{len(per_essay)} | "
                + (f"{cosine_present}/{len(cosine)} |" if cosine_present is not None
                   else "— |")
            )

        judge_rank = shares.rank(ascending=False)
        cosine_shares = pd.Series(
            {c: cosine[f"share_{c}"].mean() for c in codes if f"share_{c}" in cosine}
        )
        if len(cosine_shares) == len(codes):
            correlation = float(
                np.corrcoef(shares.reindex(codes), cosine_shares.reindex(codes))[0, 1]
            )
            lines += [
                "",
                "| Discourse | Judge share of extent | Cosine share of sentences |",
                "|---|---|---|",
            ]
            for code in codes:
                lines.append(
                    f"| {code} | {pct(shares[code])} | {pct(cosine_shares[code])} |"
                )
            lines += [
                "",
                f"Correlation between the two share vectors: **r = {correlation:+.3f}** "
                f"over {len(codes)} discourses.",
                "",
                ("The two instruments broadly agree on which discourses get room, "
                 "which is reassuring for both." if correlation > 0.5 else
                 "The two instruments disagree about which discourses get room. "
                 "Given that cosine was shown not to recover the human discourse "
                 "structure at all (`src/validate_baseline.py`), the judge is the "
                 "more credible of the two — but the disagreement should be "
                 "reported, not resolved by preference."),
            ]

    # --- 5. checking the judge ---------------------------------------------
    presence_unanimous = float((verdicts["presence_agreement"] == 1.0).mean())
    treatment_unanimous = float((verdicts["treatment_agreement"] == 1.0).mean())

    lines += [
        "",
        "## 5. Checking the judge",
        "",
        "A judge is an instrument too. These are the checks that decide how much "
        "of the above to believe.",
        "",
        "### Replicate agreement",
        "",
        f"Each pair was judged {config.JUDGE_REPLICATES} times independently at "
        f"temperature {config.JUDGE_TEMPERATURE}.",
        "",
        "| | Unanimous | Mean agreement |",
        "|---|---|---|",
        f"| Presence | {pct(presence_unanimous)} | "
        f"{verdicts['presence_agreement'].mean():.3f} |",
        f"| Treatment | {pct(treatment_unanimous)} | "
        f"{verdicts['treatment_agreement'].mean():.3f} |",
        "",
        "| Discourse | Presence agreement | Treatment agreement |",
        "|---|---|---|",
    ]
    for code in codes:
        subset = verdicts[verdicts["discourse"] == code]
        lines.append(
            f"| {code} | {subset['presence_agreement'].mean():.3f} | "
            f"{subset['treatment_agreement'].mean():.3f} |"
        )

    non_absent = raw[raw["presence"] != "absent"]
    lines += [
        "",
        "### Quote verification",
        "",
        "Every non-absent judgement had to supply a verbatim quote, checked "
        "against the essay automatically. Quotes joined by an ellipsis are "
        "checked fragment by fragment.",
        "",
        "| | Value |",
        "|---|---|",
        f"| Judgements requiring a quote | {len(non_absent)} |",
        f"| Fully verified | {pct(non_absent['quote_verified'].mean())} |",
        f"| Mean share of quoted words found | "
        f"{non_absent['quote_match'].mean():.3f} |",
        f"| Wholly unfindable (0% matched) | "
        f"{pct((non_absent['quote_match'] == 0).mean())} |",
        f"| Unparseable responses | {int((raw['parse_error'].astype(str) != '').sum())} "
        f"of {len(raw)} |",
        "",
        "A quote that cannot be found is the judge inventing support for a "
        "rating. That rate is the ceiling on how much any individual verdict can "
        "be trusted.",
    ]

    # Do two discourses get justified by the same passage? If the judge cites
    # one sentence for several perspectives, it is not discriminating between
    # them any better than argmax was.
    overlaps = []
    for (model, repetition), group in raw[raw["quote"].astype(bool)].groupby(
            ["model", "repetition"]):
        seen = {}
        for row in group.itertuples():
            key = " ".join(str(row.quote).lower().split())[:120]
            if key in seen and seen[key] != row.discourse:
                overlaps.append((model, repetition, seen[key], row.discourse))
            seen[key] = row.discourse

    lines += [
        "",
        "### Do different discourses get the same evidence?",
        "",
        f"{len(overlaps)} cases where one passage was cited as support for two "
        "different discourses within the same essay.",
        "",
        ("Some overlap is expected and legitimate — the report's own Table 5 has "
         "several discourse pairs correlating above 0.5, so a passage can "
         "genuinely serve both. Heavy overlap would mean the judge is no better "
         "at separating the six than the embedding was."
         if overlaps else
         "No passage was cited for two discourses, which is the strongest "
         "available sign that the judge is reading each description on its own "
         "terms rather than matching topic."),
    ]
    if overlaps:
        pair_counts = Counter(tuple(sorted((a, b))) for _, _, a, b in overlaps)
        lines += [
            "",
            "| Discourse pair | Shared quotes | Essays |",
            "|---|---|---|",
        ]
        for (left, right), count in pair_counts.most_common(10):
            lines.append(f"| {left} – {right} | {count} | {count}/{n_essays} |")

        # How many genuinely distinct passages does the judge find per essay? If
        # it cites four passages to justify six discourses, two of the six are
        # not being separately evidenced, and coverage is overstated by exactly
        # that much.
        quoted = raw[raw["quote"].astype(bool)].copy()
        quoted["key"] = quoted["quote"].map(
            lambda q: " ".join(str(q).lower().split())[:120]
        )
        distinct = quoted.sort_values("replicate").groupby(
            ["model", "repetition", "discourse"]
        ).first().reset_index().groupby(["model", "repetition"])["key"].nunique()

        lines += [
            "",
            f"Across the six discourses, the judge cites a mean of "
            f"**{distinct.mean():.1f} genuinely distinct passages per essay**.",
            "",
            "| Distinct passages cited | Essays |",
            "|---|---|",
        ]
        for level, count in distinct.value_counts().sort_index().items():
            lines.append(f"| {int(level)} | {int(count)} |")

        worst_pair, worst_count = pair_counts.most_common(1)[0]
        if worst_count >= n_essays / 3:
            merged = per_essay.copy()
            left, right = worst_pair
            merged["_merged"] = merged[left] | merged[right]
            remaining = [c for c in codes if c not in worst_pair]
            merged_coverage = merged[["_merged"] + remaining].sum(axis=1)
            lines += [
                "",
                f"**{left} and {right} are supported by the same passage in "
                f"{worst_count} of {n_essays} essays.** They are the most "
                "correlated pair in the report's own factor arrays, so some "
                "overlap is expected — but at this rate the judge is not "
                "separately evidencing them, and coverage counts them twice.",
                "",
                f"Treating {left} and {right} as one discourse gives mean "
                f"coverage {merged_coverage.mean():.2f} of {len(codes) - 1}, "
                f"against {coverage.mean():.2f} of {len(codes)} as reported "
                "above. Read the headline coverage figure with that in mind.",
            ]

    # The treatment scale is only a measurement if it varies. If the judge
    # returns one label for everything, that column carries no information and
    # saying so is more useful than reporting the constant as a finding.
    treatment_values = verdicts.loc[
        verdicts["treatment"] != "not_applicable", "treatment"
    ].value_counts()
    if len(treatment_values) <= 1:
        only = treatment_values.index[0] if len(treatment_values) else "none"
        lines += [
            "",
            "### The treatment scale did not vary",
            "",
            f"**Every one of the {int(treatment_values.sum())} applicable "
            f"judgements came back `{only}`. Not one `dismissed`, not one "
            "`endorsed`.**",
            "",
            "A constant is not a measurement. Either these briefings really are "
            "uniformly even-handed — plausible, since they were written to "
            "canvass all sides — or the judge will not apply the ends of this "
            "scale. Nothing here distinguishes the two, so the treatment column "
            "should be reported as uninformative rather than as evidence of "
            "even-handedness.",
            "",
            "This matters because it contradicts the NLI stage, which put "
            "dismissal at 31.5% for A, 30.2% for E and 35.8% for F. Those "
            "figures were already suspect — part of the signal traced to a "
            "rewriting template in the depersonalised statements — but the "
            "disagreement between the two instruments is now total, and "
            "unresolved. The dismissal question needs an instrument that has "
            "been shown to be capable of returning both answers.",
        ]

    lines += [
        "",
        "## Caveats",
        "",
        f"- One judge model. `{config.JUDGE_MODEL}` has its own reading of what "
        "counts as a way of reasoning, and nothing here separates that from the "
        "essays. A second judge would be the obvious control.",
        "- Blinding removes the position labels, but the descriptions still "
        "carry the report's own vocabulary. A judge can match distinctive "
        "phrasing without engaging the reasoning, exactly as the embedding did.",
        "- Presence, treatment and extent are the judge's ratings, not ground "
        "truth. Replicate agreement bounds their stability; it says nothing "
        "about their accuracy. A hand-labelled calibration set is still the "
        "missing piece.",
        "- Extent is a coarse four-point scale, so the balance shares in "
        "section 3 are approximate and should not be read past the first "
        "significant figure.",
        "",
    ]

    out_dir = config.results_dir(prompt_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = config.judge_slug()
    verdicts.to_csv(out_dir / f"judge_verdicts_{slug}.csv", index=False)
    raw.drop(columns=["raw_response"], errors="ignore").to_csv(
        out_dir / f"judge_raw_{slug}.csv", index=False
    )
    summary = "\n".join(lines)
    (out_dir / f"judge_summary_{slug}.md").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"\nWritten to {out_dir}/judge_summary_{slug}.md")


if __name__ == "__main__":
    main()
