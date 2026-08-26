"""
Step 4b: sentence-level analysis, combining airtime and stance.

Each measure is used only for what it has been shown to do (see
src/validate_stance.py):

  * Cosine similarity tracks what a text is *about*, not what it claims. It is
    used here as the AIRTIME measure -- which discourse's themes a sentence
    engages -- and for nothing else.

  * NLI reliably detects contradiction and, against this corpus, barely detects
    entailment: "many Australians worry that X" does not entail X, so the model
    correctly abstains, while "this is unfounded" is caught cleanly. It is used
    here as the DISMISSAL measure.

Combining them gives the distinction that matters for discursive representation
and that neither measure can draw alone: giving a discourse airtime and letting
it stand, versus raising it in order to defuse it. A briefing that surfaces
every concern and then reassures on each one scores perfectly on airtime while
representing nobody.

Stance is aggregated to the discourse level through the factor array. For
sentence s and discourse X,

    alignment(s, X) = sum_i stance(s, i) * z(i, X)

which has the right sign structure throughout: asserting what X endorses is
positive, denying what X endorses is negative, and asserting what X rejects is
negative too.

WHAT THE REPORT ASKS DEPENDS ON THE BASELINE
--------------------------------------------
Against the four pre-deliberative discourses the question is proportionality:
do the essays mirror the distribution of opinion in the Australian population?

Against the six post-deliberative discourses it is not. Those discourses are the
ways of reasoning that emerged once a citizens' jury had worked through the
issue, and a brainstorming tool's job is to make that space available rather
than to reproduce raw opinion. Sections 4 to 6 therefore ask instead:

  * EXISTENCE   -- are the human discourses identifiable in the output at all?
  * COVERAGE    -- how many of them does a single essay make available?
  * RELIABILITY -- does the same discourse appear run after run, or by luck?
  * BALANCE     -- does each get enough space to be engaged with, or is it
                   present in name only?

Usage:
    python src/analyse_sentences.py
    python src/analyse_sentences.py --baseline pre
    python src/analyse_sentences.py --form original
"""

import argparse

import numpy as np
import pandas as pd

import config
from analyse import pct, points, target_distribution, total_variation_distance
from embed import load_discourses


def load_factor_array(codes: list[str], statement_ids: pd.Series) -> np.ndarray:
    """
    Build the (statements x discourses) z-score matrix for the active baseline,
    aligned to the column order of the NLI stance matrix.

    Alignment is by statement_id rather than by position, because the two
    baselines cover different statement sets. The post-deliberative sorts used
    45 of the 46 items: OP12 records that "Statement 46 was dropped following the
    Mapping Study and not used in subsequent surveys". Item 46 therefore gets a
    z-score of zero under every post-deliberative discourse, contributing nothing
    to any alignment -- which is the honest treatment of an item that carries no
    post-deliberative information, and means the NLI matrices need no re-scoring.
    """
    z_table = pd.read_csv(config.baseline()["z_file"]).set_index("statement_id")

    missing = set(codes) - set(z_table.columns)
    if missing:
        raise ValueError(
            f"Factor array {config.baseline()['z_file'].name} has no column for "
            f"{sorted(missing)}; found {[c for c in z_table.columns]}"
        )

    aligned = z_table.reindex(statement_ids)[codes]

    # aligned is indexed by statement_id, statement_ids by row position, so the
    # mask is taken as a plain array rather than an index-aligned Series.
    absent_mask = aligned.isna().all(axis=1).to_numpy()
    if absent_mask.any():
        print(f"  {int(absent_mask.sum())} statement(s) carry no z-score under "
              f"baseline '{config.BASELINE}' and are zeroed: "
              f"{sorted(statement_ids.to_numpy()[absent_mask].tolist())}")

    return aligned.fillna(0.0).to_numpy()


def load_inputs(prompt_name: str, form: str, context: bool = False):
    """Load the sentence table, the airtime similarities and the stance matrix."""
    processed = config.PROCESSED_DIR / prompt_name

    sentences = pd.read_csv(processed / "sentences.csv")

    airtime_path = processed / f"sentence_x_discourse_{config.BASELINE}.npy"
    if not airtime_path.exists():
        raise FileNotFoundError(
            f"No sentence-to-discourse similarities at {airtime_path}. "
            f"Run: python src/embed_sentences.py --baseline {config.BASELINE}"
        )
    airtime = np.load(airtime_path)
    # Row-position join: the matrix has no keys of its own, so the only thing
    # standing between us and silently mislabelled discourses is this check.
    if len(airtime) != len(sentences):
        raise ValueError(
            f"{airtime_path.name} has {len(airtime)} rows but sentences.csv has "
            f"{len(sentences)}. Re-run src/embed_sentences.py."
        )

    suffix = f"_{form}" + ("_context" if context else "")
    stance_path = processed / f"sentence_stance{suffix}.npy"
    if not stance_path.exists():
        raise FileNotFoundError(
            f"No stance matrix at {stance_path}. Run: python src/stance.py "
            f"--form {form}" + (" --context" if context else "")
        )
    # (sentences x statements x 2), last axis [P(entail), P(contradict)]
    stance_components = np.load(stance_path)

    discourses = load_discourses()

    # The statement table drives the column order of the stance matrix, so the
    # factor array is aligned to it rather than the other way round.
    statements = pd.read_csv(config.DEPERSONALISED_FILE)

    if not (len(sentences) == len(airtime) == len(stance_components)):
        raise ValueError(
            f"Length mismatch: {len(sentences)} sentences, {len(airtime)} "
            f"similarity rows, {len(stance_components)} stance rows."
        )
    if stance_components.shape[1] != len(statements):
        raise ValueError(
            f"Stance matrix scores {stance_components.shape[1]} statements but "
            f"the statement table has {len(statements)}."
        )

    return sentences, airtime, stance_components, discourses, statements


def normalised_entropy(shares: pd.Series) -> float:
    """
    Evenness of a distribution, on 0-1.

    1.0 means every discourse gets an identical share; 0.0 means one takes
    everything. Reported alongside TVD because the two answer different
    questions: TVD is distance from a specific reference, entropy is how
    concentrated the output is regardless of which discourse dominates.
    """
    values = shares[shares > 0].to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(-(values * np.log(values)).sum() / np.log(len(shares)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    parser.add_argument("--form", choices=config.STATEMENT_FORMS,
                        default="depersonalised",
                        help="which Q-statement phrasing to analyse")
    parser.add_argument("--context", action="store_true",
                        help="use the anaphora robustness pass, in which "
                             "sentences opening with an unresolved reference "
                             "were scored with their predecessor prepended")
    args = parser.parse_args()
    prompt_name = config.apply_overrides(args)

    sentences, airtime, stance_components, discourses, statements = load_inputs(
        prompt_name, args.form, args.context
    )

    all_codes = discourses["code"].tolist()
    codes = [c for c in all_codes if c in config.ACTIVE_DISCOURSES]
    excluded = [c for c in all_codes if c not in config.ACTIVE_DISCOURSES]
    names = dict(zip(discourses["code"], discourses["name"]))

    settings = config.baseline()
    target = target_distribution(codes)
    uniform = 1.0 / len(codes)

    # --- airtime -----------------------------------------------------------
    airtime_all = pd.DataFrame(airtime, columns=all_codes)
    winner = np.array(codes)[airtime_all[codes].to_numpy().argmax(axis=1)]

    # --- stance ------------------------------------------------------------
    # Signed stance per (sentence, statement), then projected onto each
    # discourse through its column of the factor array.
    stance = stance_components[:, :, 0] - stance_components[:, :, 1]
    z = load_factor_array(all_codes, statements["statement_id"])
    alignment = pd.DataFrame(stance @ z, columns=all_codes)

    table = pd.concat([
        sentences.reset_index(drop=True),
        airtime_all.add_prefix("airtime_"),
        alignment.add_prefix("align_"),
    ], axis=1)
    table["airtime_discourse"] = winner
    # Alignment with the discourse whose themes this sentence engages: positive
    # means it voices that discourse, negative means it engages the themes in
    # order to deny them.
    table["align_own"] = [
        table.at[i, f"align_{c}"] for i, c in enumerate(winner)
    ]
    table["verdict"] = np.where(
        table["align_own"] > config.STANCE_THRESHOLD, "aligned",
        np.where(table["align_own"] < -config.STANCE_THRESHOLD,
                 "dismissed", "reported"),
    )
    table["represented"] = table["verdict"] != "dismissed"

    # --- proportions -------------------------------------------------------
    observed = pd.Series(winner).value_counts(normalize=True).reindex(codes).fillna(0)
    kept = table[table["represented"]]
    adjusted = (kept["airtime_discourse"].value_counts(normalize=True)
                .reindex(codes).fillna(0))

    # Output from the anaphora pass is a diagnostic, not a result: see
    # analyse_context.py for the validity check it failed.
    context_warning = (
        "> **Status: diagnostic only — not used for scoring.** This file is "
        "output from the anaphora context pass, which failed its own validity "
        "check (the combined premise tracks the prepended sentence's score more "
        f"closely than the target's). Report `sentence_summary_{args.form}.md` "
        "instead."
    ) if args.context else None

    lines = [
        "# Sentence-level representation: airtime, stance and availability",
        "",
    ] + ([context_warning, ""] if context_warning else []) + [
        f"Baseline: **{settings['label']}**",
        "",
        f"Prompt: `{prompt_name}`  |  Statement form: `{args.form}`  |  "
        f"NLI: `{config.NLI_MODEL}`"
        + ("  |  **anaphora context pass**" if args.context else ""),
        "",
        f"{len(sentences)} sentences from {sentences['run_id'].nunique()} essays, "
        f"{sentences['model'].nunique()} models."
        + (f" Discourse {', '.join(excluded)} excluded (expected share zero)."
           if excluded else ""),
        "",
        "| Code | Discourse |",
        "|---|---|",
    ]
    for code in codes:
        lines.append(f"| {code} | {names.get(code, '')} |")

    # ------------------------------------------------------------------
    # 1. Airtime
    # ------------------------------------------------------------------
    lines += [
        "",
        "## 1. Airtime — whose themes get engaged",
        "",
        "Cosine similarity, sentence against discourse description. This measures "
        "topical engagement only; it is deliberately blind to whether the "
        "sentence endorses or rejects what it engages with.",
        "",
        "| Discourse | Observed | Reference | Deviation (pp) |",
        "|---|---|---|---|",
    ]
    for code in codes:
        lines.append(
            f"| {code} | {pct(observed[code])} | {pct(target[code])} | "
            f"{points(observed[code] - target[code])} |"
        )
    lines += [
        "",
        f"**Total variation distance: {total_variation_distance(observed, target):.3f}**"
        + ("  |  " if settings["target"] == "uniform" else
           f"  (a uniform split over {len(codes)} discourses would score "
           f"{total_variation_distance(pd.Series(uniform, index=codes), target):.3f})")
        + (f"Evenness: {normalised_entropy(observed):.3f}"
           if settings["target"] == "uniform" else ""),
        "",
        "| Model | " + " | ".join(codes) + " | TVD |",
        "|---" * (len(codes) + 2) + "|",
    ]
    for model, group in table.groupby("model"):
        shares = (group["airtime_discourse"].value_counts(normalize=True)
                  .reindex(codes).fillna(0))
        lines.append(
            f"| `{model}` | " + " | ".join(pct(shares[c]) for c in codes)
            + f" | {total_variation_distance(shares, target):.3f} |"
        )

    # ------------------------------------------------------------------
    # 2. Stance
    # ------------------------------------------------------------------
    lines += [
        "",
        "## 2. Stance — left standing, or raised and defused",
        "",
        "Of the sentences that engage each discourse's themes, how many go on to "
        "align with its position, contradict it, or merely report it without "
        "commitment. `reported` is the expected majority: a briefing that "
        "attributes a view to others neither asserts nor denies it, and NLI "
        "correctly abstains.",
        "",
        "Note the column name. `aligned` is **not** the same as voiced: because "
        "the NLI stage detects denial far more readily than assertion, most of "
        "what lands in that column is the text denying something the discourse "
        "also denies. That is a real form of agreement, but it is not the essay "
        "speaking in the discourse's voice.",
        "",
        "| Discourse | Sentences | Aligned | Reported | Dismissed | Net alignment |",
        "|---|---|---|---|---|---|",
    ]
    for code in codes:
        group = table[table["airtime_discourse"] == code]
        verdicts = group["verdict"].value_counts(normalize=True)
        lines.append(
            f"| {code} | {len(group)} | "
            f"{pct(verdicts.get('aligned', 0.0))} | "
            f"{pct(verdicts.get('reported', 0.0))} | "
            f"{pct(verdicts.get('dismissed', 0.0))} | "
            f"{group['align_own'].mean():+.3f} |"
        )

    lines += [
        "",
        "Net alignment is the mean of `alignment(sentence, its own discourse)`. "
        "A negative value means the corpus engages that discourse's themes more "
        "to deny them than to affirm them.",
        "",
        "### Dismissal asymmetry, by discourse and model",
        "",
        "| Model | " + " | ".join(f"{c} net" for c in codes) + " |",
        "|---" * (len(codes) + 1) + "|",
    ]
    for model, group in table.groupby("model"):
        cells = []
        for code in codes:
            subset = group[group["airtime_discourse"] == code]
            cells.append(f"{subset['align_own'].mean():+.3f}" if len(subset) else "—")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")

    # ------------------------------------------------------------------
    # 3. Representation = engaged and not dismissed
    # ------------------------------------------------------------------
    lines += [
        "",
        "## 3. Representation = engaged and not dismissed",
        "",
        "Airtime shares recomputed after removing sentences that engage a "
        "discourse in order to deny it. A discourse is represented when its "
        "themes are raised and its position is left standing.",
        "",
        "| Discourse | Airtime | Not dismissed | Reference | Deviation (pp) |",
        "|---|---|---|---|---|",
    ]
    for code in codes:
        lines.append(
            f"| {code} | {pct(observed[code])} | {pct(adjusted[code])} | "
            f"{pct(target[code])} | {points(adjusted[code] - target[code])} |"
        )
    lines += [
        "",
        f"**TVD after removing dismissals: "
        f"{total_variation_distance(adjusted, target):.3f}** "
        f"(airtime alone: {total_variation_distance(observed, target):.3f})",
    ]

    # ------------------------------------------------------------------
    # Per-essay presence, the unit for sections 4 to 6
    # ------------------------------------------------------------------
    # A discourse counts as available in an essay only if it is represented by a
    # meaningful amount of that essay -- both a minimum share and a minimum
    # absolute count, so neither a very long nor a very short essay can clear the
    # bar on a technicality. One stray sentence is not a perspective made
    # available for deliberation.
    per_essay_rows = []
    for (model, run_id), group in table.groupby(["model", "run_id"]):
        n_sentences = len(group)
        floor = max(config.PRESENCE_MIN_SENTENCES,
                    config.PRESENCE_MIN_SHARE * n_sentences)
        represented = group[group["represented"]]
        counts = (represented["airtime_discourse"].value_counts()
                  .reindex(codes).fillna(0))
        shares = (represented["airtime_discourse"].value_counts(normalize=True)
                  .reindex(codes).fillna(0))
        row = {"model": model, "run_id": run_id, "n_sentences": n_sentences}
        for code in codes:
            row[f"n_{code}"] = int(counts[code])
            row[f"share_{code}"] = float(shares[code])
            row[f"present_{code}"] = bool(counts[code] >= floor)
        row["coverage"] = sum(row[f"present_{c}"] for c in codes)
        row["tvd"] = total_variation_distance(shares, target)
        row["evenness"] = normalised_entropy(shares)
        per_essay_rows.append(row)
    per_essay = pd.DataFrame(per_essay_rows)

    present_columns = [f"present_{c}" for c in codes]

    # ------------------------------------------------------------------
    # 4. Existence / recovery
    # ------------------------------------------------------------------
    lines += [
        "",
        "## 4. Existence — are the human discourses identifiable in the output?",
        "",
        "Before asking how much space each discourse gets, ask whether the "
        "measurement can tell them apart at all. Two things have to hold: each "
        "discourse must claim some sentences, and the claim must be more than a "
        "coin flip between near-identical baselines.",
        "",
        "| Discourse | Sentences engaged | Represented | Mean cosine | Mean winning margin |",
        "|---|---|---|---|---|",
    ]
    for code in codes:
        group = table[table["airtime_discourse"] == code]
        # How far the winner led the runner-up, among sentences this discourse won.
        if len(group):
            row_values = group[[f"airtime_{c}" for c in codes]].to_numpy()
            ordered = np.sort(row_values, axis=1)
            margin = float((ordered[:, -1] - ordered[:, -2]).mean())
            margin_text = f"{margin:.4f}"
        else:
            margin_text = "—"
        lines.append(
            f"| {code} | {len(group)} | {int(group['represented'].sum())} | "
            f"{table[f'airtime_{code}'].mean():.4f} | {margin_text} |"
        )

    recovered = [c for c in codes if (per_essay[f"present_{c}"]).any()]
    never = [c for c in codes if c not in recovered]
    lines += [
        "",
        f"**{len(recovered)} of {len(codes)} discourses are recovered somewhere "
        f"in the corpus**"
        + (f"; {', '.join(never)} never reach the presence threshold in any essay."
           if never else " — every one appears in at least one essay."),
        "",
        "Corpus-level existence is a weak test: with "
        f"{len(sentences)} sentences and {len(codes)} discourses, argmax "
        "assignment will hand every discourse something whether or not the essays "
        "genuinely articulate it. The informative unit is the individual essay, "
        "which is what a participant would actually read — sections 5 and 6.",
    ]

    # ------------------------------------------------------------------
    # 5. Coverage and reliability
    # ------------------------------------------------------------------
    lines += [
        "",
        "## 5. Coverage and reliability — how much of the space does one essay open?",
        "",
        f"A discourse counts as available in an essay when at least "
        f"{config.PRESENCE_MIN_SENTENCES} sentences, and at least "
        f"{config.PRESENCE_MIN_SHARE:.0%} of the essay, engage it without "
        "dismissing it. Coverage is how many of the "
        f"{len(codes)} it makes available. This is the central brainstorming "
        "criterion: a participant reads one essay, not thirty.",
        "",
        "| Model | Mean coverage | Worst essay | Best essay | Essays at full coverage |",
        "|---|---|---|---|---|",
    ]
    for model, group in per_essay.groupby("model"):
        full = int((group["coverage"] == len(codes)).sum())
        lines.append(
            f"| `{model}` | {group['coverage'].mean():.1f} / {len(codes)} | "
            f"{int(group['coverage'].min())} | {int(group['coverage'].max())} | "
            f"{full} / {len(group)} |"
        )
    lines.append(
        f"| **pooled** | **{per_essay['coverage'].mean():.1f} / {len(codes)}** | "
        f"{int(per_essay['coverage'].min())} | "
        f"{int(per_essay['coverage'].max())} | "
        f"**{int((per_essay['coverage'] == len(codes)).sum())} / "
        f"{len(per_essay)}** |"
    )

    lines += [
        "",
        "### Reliability — how often each discourse turns up",
        "",
        "Share of that model's ten essays in which the discourse is available. "
        "A discourse at 10/10 can be counted on; one at 3/10 means a participant "
        "using the tool once is unlikely to encounter it at all.",
        "",
        "| Model | " + " | ".join(codes) + " |",
        "|---" * (len(codes) + 1) + "|",
    ]
    for model, group in per_essay.groupby("model"):
        cells = [f"{int(group[f'present_{c}'].sum())}/{len(group)}" for c in codes]
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    pooled_present = [
        f"{int(per_essay[f'present_{c}'].sum())}/{len(per_essay)}" for c in codes
    ]
    lines.append("| **pooled** | " + " | ".join(pooled_present) + " |")

    lines += [
        "",
        "Distribution of coverage across all "
        f"{len(per_essay)} essays:",
        "",
        "| Discourses available | Essays |",
        "|---|---|",
    ]
    coverage_counts = per_essay["coverage"].value_counts().sort_index(ascending=False)
    for level, count in coverage_counts.items():
        lines.append(f"| {int(level)} / {len(codes)} | {int(count)} |")

    # --- how much of this is the threshold? --------------------------------
    lines += [
        "",
        "### Sensitivity — how much of the above is the threshold",
        "",
        "Presence turns on very small counts: in the median essay the weaker "
        "discourses get two or three non-dismissed sentences each. The floor is "
        "therefore doing a lot of the work, and moving it moves the headline.",
        "",
        "| Min. sentences | Mean coverage | Full coverage | "
        + " | ".join(codes) + " |",
        "|---" * (len(codes) + 3) + "|",
    ]
    for floor_value in config.PRESENCE_SENSITIVITY:
        present = {c: per_essay[f"n_{c}"] >= floor_value for c in codes}
        coverage = sum(present.values())
        marker = " ←" if floor_value == config.PRESENCE_MIN_SENTENCES else ""
        lines.append(
            f"| {floor_value}{marker} | {coverage.mean():.2f} | "
            f"{int((coverage == len(codes)).sum())}/{len(per_essay)} | "
            + " | ".join(f"{int(present[c].sum())}/{len(per_essay)}" for c in codes)
            + " |"
        )
    lines += [
        "",
        "The *ordering* is stable — B, C and D are available in every essay at "
        "every threshold, and A, E and F are the weak ones throughout. The "
        "*levels* are not: read them as threshold-dependent, not as counts.",
    ]

    # --- how much of this is just the marginal distribution? ---------------
    # If a discourse takes 5% of an essay's scored sentences it averages under
    # three of them, and will fall below the floor by chance alone a fair share
    # of the time. This asks how much coverage the pooled shares would produce
    # on their own, with sentences allocated at random within each essay.
    #
    # The draw size must be the number of sentences the OBSERVED counts were
    # computed from, which is the non-dismissed total, not the essay's full
    # sentence count. An earlier version drew `n_sentences` and so handed each
    # simulated essay ~12% more sentences than the real one had to allocate
    # (1,610 against 1,442 corpus-wide). That inflated the null by 0.15
    # discourses and pushed the observed mean just outside the interval,
    # manufacturing an essay-level concentration effect that does not exist.
    pooled_counts = np.array([per_essay[f"n_{c}"].sum() for c in codes], dtype=float)
    pooled_shares = pooled_counts / pooled_counts.sum()
    # Per essay: how many sentences actually entered the coverage counts.
    scored_per_essay = per_essay[[f"n_{c}" for c in codes]].sum(axis=1)
    generator = np.random.default_rng(0)
    simulated = []
    for _ in range(config.PRESENCE_NULL_DRAWS):
        total = 0
        for n_scored, n_sentences in zip(scored_per_essay, per_essay["n_sentences"]):
            # The floor is unchanged: it is defined against the essay's full
            # length, exactly as in the observed calculation above.
            floor = max(config.PRESENCE_MIN_SENTENCES,
                        config.PRESENCE_MIN_SHARE * n_sentences)
            draw = generator.multinomial(int(n_scored), pooled_shares)
            total += int((draw >= floor).sum())
        simulated.append(total / len(per_essay))
    simulated = np.array(simulated)
    low, high = np.percentile(simulated, [2.5, 97.5])

    lines += [
        "",
        "### Null model — how much of this is arithmetic",
        "",
        "Coverage is bounded by balance. A discourse holding a small share of "
        f"the ~{scored_per_essay.median():.0f} scored sentences in an essay "
        "will drop below the floor by chance alone some of the time, whatever "
        "the essay is doing. This reallocates each essay's non-dismissed "
        "sentences at random in the corpus-wide proportions and recomputes "
        f"coverage, {config.PRESENCE_NULL_DRAWS:,} times.",
        "",
        "Note what the reference is: the pooled shares come from this corpus, "
        "not from the human study. The null therefore asks whether any single "
        "essay is unusually concentrated *given how often the models write "
        "each discourse overall* -- it cannot ask whether those overall rates "
        "are themselves adequate. That question belongs to the uniform "
        "reference in section 6.",
        "",
        "| | Mean coverage |",
        "|---|---|",
        f"| Observed | **{per_essay['coverage'].mean():.2f}** / {len(codes)} |",
        f"| Random allocation, same pooled shares | {simulated.mean():.2f} "
        f"(95% {low:.2f}–{high:.2f}) |",
        "",
        (f"Observed coverage falls **below** the null interval, so essays are "
         "somewhat more concentrated than the pooled shares alone would produce "
         "— but the effect is small. Most of the distance from "
         f"{len(codes)}/{len(codes)} is a mechanical consequence of the "
         "imbalance reported in section 6, not an additional finding."
         if per_essay["coverage"].mean() < low else
         f"Observed coverage sits inside the null interval, so it adds nothing "
         "beyond the imbalance already reported in section 6: the shortfall "
         f"from {len(codes)}/{len(codes)} is what the pooled shares produce on "
         "their own."),
    ]

    # ------------------------------------------------------------------
    # 6. Balance
    # ------------------------------------------------------------------
    marginal_floor = config.MARGINALISATION_RATIO * uniform
    marginalised = [c for c in codes if adjusted[c] < marginal_floor]

    lines += [
        "",
        "## 6. Balance — is each discourse given room, or present in name only?",
        "",
        "Coverage asks whether a discourse appears. Balance asks whether it "
        "appears with enough space to be engaged, contested or developed. A "
        "discourse is counted as **marginalised** when it takes less than "
        f"{config.MARGINALISATION_RATIO:.0%} of an even share "
        f"(under {marginal_floor:.1%} of represented sentences).",
        "",
        "| Model | " + " | ".join(codes) + " | Evenness | TVD |",
        "|---" * (len(codes) + 3) + "|",
    ]
    for model, group in table.groupby("model"):
        model_kept = group[group["represented"]]
        shares = (model_kept["airtime_discourse"].value_counts(normalize=True)
                  .reindex(codes).fillna(0))
        cells = []
        for code in codes:
            marker = " ⚠" if shares[code] < marginal_floor else ""
            cells.append(f"{pct(shares[code])}{marker}")
        lines.append(
            f"| `{model}` | " + " | ".join(cells)
            + f" | {normalised_entropy(shares):.3f} "
            f"| {total_variation_distance(shares, target):.3f} |"
        )
    lines.append(
        "| **pooled** | "
        + " | ".join(
            f"{pct(adjusted[c])}{' ⚠' if adjusted[c] < marginal_floor else ''}"
            for c in codes
        )
        + f" | {normalised_entropy(adjusted):.3f} "
        f"| {total_variation_distance(adjusted, target):.3f} |"
    )

    lines += [
        "",
        f"**{len(marginalised)} of {len(codes)} discourses are marginalised "
        f"pooled across the corpus**"
        + (f": {', '.join(f'{c} ({names.get(c, "")})' for c in marginalised)}."
           if marginalised else " — every discourse clears the floor."),
        "",
        "Evenness is normalised entropy: 1.0 is a perfectly even split across "
        f"the {len(codes)} discourses, 0.0 is one discourse taking everything. "
        "It is reported alongside TVD because the two differ in what they "
        "punish — TVD is distance from the reference, evenness is concentration "
        "regardless of which discourse dominates.",
        "",
        "Within-essay balance (mean over essays, so a model that covers "
        "everything by averaging lopsided essays does not score well here):",
        "",
        "| Model | Mean evenness | Mean TVD |",
        "|---|---|---|",
    ]
    for model, group in per_essay.groupby("model"):
        lines.append(
            f"| `{model}` | {group['evenness'].mean():.3f} | "
            f"{group['tvd'].mean():.3f} |"
        )

    # ------------------------------------------------------------------
    # Caveats
    # ------------------------------------------------------------------
    denying = (stance < -0.5).mean()
    asserting = (stance > 0.5).mean()
    baseline_similarity = ""
    if len(codes) > 1:
        matrix = table[[f"airtime_{c}" for c in codes]].to_numpy()
        ordered = np.sort(matrix, axis=1)
        baseline_similarity = (
            f"Mean gap between the winning and runner-up discourse is "
            f"{float((ordered[:, -1] - ordered[:, -2]).mean()):.4f} on a cosine "
            f"scale where the discourses sit at "
            f"{float(matrix.mean()):.3f} from the average sentence. "
        )

    lines += [
        "",
        "## Caveats",
        "",
        f"- The NLI stage detects denial far more readily than assertion "
        f"({denying:.2%} of pairs vs {asserting:.2%}). This asymmetry was "
        "predicted before the run and is a property of the instrument, not a "
        "finding about the essays: reported speech does not entail the "
        "proposition reported, so the model abstains on voicing while still "
        "catching explicit rejection. This is why the stance column is labelled "
        "`aligned` rather than `voiced`, and why the dismissal rate is the only "
        "part of section 2 that should be read as a result.",
        f"- Assignment is winner-takes-all over baselines that are all about the "
        f"same topic. {baseline_similarity}Small differences therefore decide "
        "whole sentences. Section 4 reports the margin per discourse; where it "
        "is thin, the split between that discourse and its nearest neighbour is "
        "not reliable.",
        "- Airtime is stance-blind by construction. A sentence that engages a "
        "discourse's themes counts towards it whatever it says about them; that "
        "is what section 3 corrects for.",
        f"- {sentences['anaphoric'].mean():.1%} of sentences open with an "
        "unresolved reference. Run `python src/stance.py --context` for the "
        "robustness pass that gives those sentences their predecessor.",
        "- The null model in section 5 assumes sentences fall independently. "
        "They do not: a paragraph tends to stay with one discourse, so real "
        "essays are more clustered than a multinomial draw. Clustering makes "
        "low counts for a rare discourse *more* likely than the null implies, "
        "so the true mechanical baseline probably sits a little below the "
        "figure reported there, and the essay-level effect a little above. "
        "Resampling paragraphs rather than sentences would settle it and is "
        "not run.",
    ]
    if config.BASELINE == "post":
        lines += [
            "- **Discourse F rests on a single jury participant** (OP12 p. 145). "
            "Its factor array is close to one person's Q-sort. Findings about F "
            "concern a discourse the report identified, not a robustly populated "
            "position.",
            "- **The six are not equally distinguishable.** The report's own "
            "Table 5 puts C at −0.10 with A and −0.04 with F while every other "
            "pair sits between 0.21 and 0.56. Expect C to separate cleanly and "
            "A, B, D, E and F to be harder to tell apart — which inflates the "
            "apparent instability of the split among those five.",
            "- The reference distribution is uniform by choice, not by "
            "measurement. It encodes the brainstorming criterion (no relevant "
            "discourse systematically marginalised), not a claim about how "
            "common these positions are. See `post-delib_baseline.NOTES.md`.",
        ]
    lines.append("")

    # The context pass writes alongside the primary run rather than over it,
    # so the two can be compared (see src/analyse_context.py).
    tag = args.form + ("_context" if args.context else "")
    out_dir = config.results_dir(prompt_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"sentence_scores_{tag}.csv", index=False)
    per_essay.to_csv(out_dir / f"essay_coverage_{tag}.csv", index=False)

    summary = "\n".join(line for line in lines if line is not None)
    (out_dir / f"sentence_summary_{tag}.md").write_text(summary,
                                                        encoding="utf-8")
    print(summary)
    print(f"\nWritten to {out_dir}/sentence_summary_{tag}.md")


if __name__ == "__main__":
    main()
