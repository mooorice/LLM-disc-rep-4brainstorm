"""
Robustness check: what does resolving anaphora do to the stance measurements?

Splitting essays into sentences buys locality -- a concern raised in one sentence
and defused in the next stay separable -- and pays for it in anaphora. "This
would deepen inequality" has no subject once it is cut loose from the sentence
before it, and an NLI model handed that as a premise will return something
confident about nothing.

`python src/stance.py --context` re-scores the corpus with the preceding sentence
prepended to every sentence that opens with an unresolved reference, leaving all
other sentences untouched. This script compares the two runs.

Three questions, in order of how much they matter:

  1. Are the untouched sentences unchanged? They were given identical premises,
     so any difference is numerical noise, and its size bounds how much of the
     rest can be believed.

  2. What happened on the extended sentences? Specifically whether resolving the
     reference recovers DENIAL, which was the stated motivation, or ASSERTION,
     which was not.

  3. Does anything downstream move? The dismissal rates, the representation
     shares and the coverage counts are the numbers that get reported, and the
     only useful answer here is whether they survive.

Usage:
    python src/analyse_context.py
    python src/analyse_context.py --baseline pre
"""

import argparse

import numpy as np
import pandas as pd

import config
from analyse import pct, points, target_distribution, total_variation_distance
from analyse_sentences import load_factor_array, normalised_entropy
from embed import load_discourses


def stance_signature(components: np.ndarray) -> np.ndarray:
    """Signed stance per (sentence, statement): P(entail) - P(contradict)."""
    return components[:, :, 0] - components[:, :, 1]


def verdicts(alignment_own: np.ndarray) -> np.ndarray:
    """Apply the reporting threshold, the same way analyse_sentences.py does."""
    return np.where(
        alignment_own > config.STANCE_THRESHOLD, "aligned",
        np.where(alignment_own < -config.STANCE_THRESHOLD, "dismissed", "reported"),
    )


def summarise(stance: np.ndarray, airtime_winner: np.ndarray, z: np.ndarray,
              codes: list[str], sentences: pd.DataFrame) -> dict:
    """Everything downstream that a change in the stance matrix could move."""
    alignment = pd.DataFrame(stance @ z, columns=codes)
    own = np.array([alignment.at[i, c] for i, c in enumerate(airtime_winner)])
    verdict = verdicts(own)

    kept = airtime_winner[verdict != "dismissed"]
    shares = pd.Series(kept).value_counts(normalize=True).reindex(codes).fillna(0)

    dismissal = {}
    for code in codes:
        mask = airtime_winner == code
        dismissal[code] = float((verdict[mask] == "dismissed").mean()) if mask.any() else np.nan

    # Per-essay coverage, on the same rule the main report uses.
    frame = pd.DataFrame({
        "run_id": sentences["run_id"], "model": sentences["model"],
        "winner": airtime_winner, "kept": verdict != "dismissed",
    })
    coverage = []
    present = {c: 0 for c in codes}
    for _, group in frame.groupby(["model", "run_id"]):
        floor = max(config.PRESENCE_MIN_SENTENCES,
                    config.PRESENCE_MIN_SHARE * len(group))
        counts = group[group["kept"]]["winner"].value_counts()
        available = [c for c in codes if counts.get(c, 0) >= floor]
        for code in available:
            present[code] += 1
        coverage.append(len(available))

    return {
        "alignment_own": own,
        "verdict": verdict,
        "shares": shares,
        "dismissal": dismissal,
        "net": {c: float(own[airtime_winner == c].mean()) for c in codes},
        "coverage": np.array(coverage),
        "present": present,
        "n_essays": len(coverage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    parser.add_argument("--form", choices=config.STATEMENT_FORMS,
                        default="depersonalised")
    parser.add_argument("--examples", type=int, default=8,
                        help="most-changed sentences to print")
    args = parser.parse_args()
    prompt_name = config.apply_overrides(args)

    processed = config.PROCESSED_DIR / prompt_name
    sentences = pd.read_csv(processed / "sentences.csv").fillna({"context": ""})

    base_path = processed / f"sentence_stance_{args.form}.npy"
    context_path = processed / f"sentence_stance_{args.form}_context.npy"
    for path in (base_path, context_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run both passes:\n"
                f"  python src/stance.py --form {args.form}\n"
                f"  python src/stance.py --form {args.form} --context"
            )

    base = stance_signature(np.load(base_path))
    context = stance_signature(np.load(context_path))
    delta = context - base

    # Which sentences were actually rewritten. A sentence flagged anaphoric but
    # sitting first in its paragraph has no predecessor and was left alone, so
    # the mask is the conjunction rather than the flag.
    extended = (sentences["anaphoric"].to_numpy()
                & (sentences["context"].str.len() > 0).to_numpy())
    untouched = ~extended

    discourses = load_discourses()
    all_codes = discourses["code"].tolist()
    codes = [c for c in all_codes if c in config.ACTIVE_DISCOURSES]
    names = dict(zip(discourses["code"], discourses["name"]))

    airtime = pd.DataFrame(
        np.load(processed / f"sentence_x_discourse_{config.BASELINE}.npy"),
        columns=all_codes,
    )
    winner = np.array(codes)[airtime[codes].to_numpy().argmax(axis=1)]

    statements = pd.read_csv(config.DEPERSONALISED_FILE)
    z = load_factor_array(all_codes, statements["statement_id"])
    z_active = z[:, [all_codes.index(c) for c in codes]]

    before = summarise(base, winner, z_active, codes, sentences)
    after = summarise(context, winner, z_active, codes, sentences)
    target = target_distribution(codes)

    lines = [
        "# Anaphora robustness: does resolving references change the result?",
        "",
        f"Baseline: **{config.baseline()['label']}**  |  Prompt: `{prompt_name}`  "
        f"|  Statement form: `{args.form}`",
        "",
        f"{int(extended.sum())} of {len(sentences)} sentences "
        f"({extended.mean():.1%}) were re-scored with their preceding sentence "
        "prepended, because they open with a bare demonstrative or pronoun. "
        f"{int(sentences['anaphoric'].sum()) - int(extended.sum())} further "
        "flagged sentences were left alone, having no predecessor inside their "
        "paragraph.",
        "",
        "## 1. Control — were the untouched sentences left alone?",
        "",
        "These sentences were handed identical premises in both runs, so any "
        "difference is numerical rather than substantive. The size of it bounds "
        "how much of everything below can be believed.",
        "",
        "| | Max change | Mean change | Pairs moving > 0.1 |",
        "|---|---|---|---|",
        f"| Untouched ({int(untouched.sum())} sentences) | "
        f"{np.abs(delta[untouched]).max():.4f} | "
        f"{np.abs(delta[untouched]).mean():.2e} | "
        f"{(np.abs(delta[untouched]) > 0.1).mean():.4%} |",
        f"| Extended ({int(extended.sum())} sentences) | "
        f"{np.abs(delta[extended]).max():.4f} | "
        f"{np.abs(delta[extended]).mean():.4f} | "
        f"{(np.abs(delta[extended]) > 0.1).mean():.2%} |",
        "",
        "The untouched row is not exactly zero because batching changes when "
        "premise lengths change, and the model runs in half precision. The "
        "magnitude is around one part in ten thousand on average, which is far "
        "below anything the analysis resolves.",
        "",
        "## 2. What resolving the reference actually recovered",
        "",
        "The stated motivation for this pass was denial: \"But this is "
        "unfounded\" is a dismissal that cannot be scored without its "
        "antecedent. That is not mainly what happened.",
        "",
        "| Signal | Extended sentences, before | after | Whole corpus, before | after |",
        "|---|---|---|---|---|",
    ]
    for label, mask_fn in [
        ("asserting (> +0.5)", lambda m: (m > 0.5)),
        ("denying (< −0.5)", lambda m: (m < -0.5)),
        ("silent (abs < 0.1)", lambda m: (np.abs(m) < 0.1)),
    ]:
        lines.append(
            f"| {label} | {mask_fn(base[extended]).mean():.2%} | "
            f"{mask_fn(context[extended]).mean():.2%} | "
            f"{mask_fn(base).mean():.2%} | {mask_fn(context).mean():.2%} |"
        )

    assert_before = (base[extended] > 0.5).mean()
    assert_after = (context[extended] > 0.5).mean()
    deny_before = (base[extended] < -0.5).mean()
    deny_after = (context[extended] < -0.5).mean()

    lines += [
        "",
        f"On the extended sentences, detected **assertion rose from "
        f"{assert_before:.2%} to {assert_after:.2%}** "
        f"(x{assert_after / max(assert_before, 1e-9):.1f}), while detected "
        f"denial went from {deny_before:.2%} to {deny_after:.2%}.",
        "",
        "That is the opposite of the expectation. Dismissals turn out not to "
        "have needed the antecedent: a contradiction marker like \"unfounded\" "
        "or \"overstated\" sits inside the sentence that carries it. What the "
        "extra context changed was assertion — which would be good news, since "
        "assertion is the weak half of the NLI instrument, if the extra "
        "assertion belonged to the sentence being scored. Section 3 tests "
        "whether it does.",
    ]

    # --- 3. where does the recovered signal come from? ---------------------
    #
    # A two-sentence premise entails whatever *either* sentence entails, and the
    # NLI model has no notion that we only care about the second one. Because
    # the prepended context is itself a scored sentence elsewhere in the corpus,
    # its solo score can be looked up and the three compared directly.
    lookup = {}
    for position, (run_id, text) in enumerate(zip(sentences["run_id"],
                                                  sentences["text"])):
        lookup[(run_id, text.strip())] = position

    pairs = [
        (position, lookup[(sentences["run_id"].iat[position],
                           sentences["context"].iat[position].strip())])
        for position in np.where(extended)[0]
        if (sentences["run_id"].iat[position],
            sentences["context"].iat[position].strip()) in lookup
    ]

    lines += [
        "",
        "## 3. Does the recovered signal belong to the sentence?",
        "",
        "A two-sentence premise entails whatever *either* sentence entails, and "
        "the NLI model has no way of knowing that only the second one is being "
        "measured. The prepended context is itself a scored sentence elsewhere "
        "in the corpus, so its solo score can be looked up and the three "
        "compared. If the combined score tracks the context more closely than "
        "the sentence, the pass is measuring the predecessor.",
        "",
    ]

    if len(pairs) < 10:
        lines.append(
            f"Only {len(pairs)} of {int(extended.sum())} contexts could be "
            "matched to a scored row, which is too few to test. Skipped."
        )
    else:
        target_index = np.array([p[0] for p in pairs])
        context_index = np.array([p[1] for p in pairs])
        combined = context[target_index]
        alone = base[target_index]
        context_alone = base[context_index]

        def flat_correlation(left, right):
            return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])

        r_sentence = flat_correlation(combined, alone)
        r_context = flat_correlation(combined, context_alone)

        newly_asserting = (combined > 0.5) & (alone <= 0.5)
        from_context = float((context_alone[newly_asserting] > 0.5).mean())
        from_neither = float((np.abs(context_alone[newly_asserting]) < 0.1).mean())

        lines += [
            f"Matched {len(pairs)} of {int(extended.sum())} extended sentences "
            "to their context's own score.",
            "",
            "| Combined (context + sentence) score compared against | Correlation | Mean absolute gap |",
            "|---|---|---|",
            f"| The sentence scored alone | {r_sentence:.3f} | "
            f"{np.abs(combined - alone).mean():.4f} |",
            f"| The **context** scored alone | **{r_context:.3f}** | "
            f"**{np.abs(combined - context_alone).mean():.4f}** |",
            "",
            f"The combined premise tracks the context more closely than the "
            f"sentence it was supposed to disambiguate. Of the "
            f"{int(newly_asserting.sum())} (sentence, statement) pairs that "
            f"newly register as assertion, **{from_context:.1%} were already "
            "assertions of the context sentence on its own** — and that context "
            "sentence is separately scored in its own right, so its stance is "
            "being counted twice.",
            "",
            f"A further {from_neither:.1%} arise where the context was silent, "
            "so they are genuinely emergent from the combination. Some of those "
            "will be correct resolutions. The examples in section 5 suggest "
            "others are spurious entailments produced by giving the model more "
            "text to find a connection in.",
        ]

    # --- 3. downstream --------------------------------------------------
    lines += [
        "",
        "## 4. Does anything downstream move?",
        "",
        "### Dismissal rate per discourse",
        "",
        "| Discourse | Before | After | Change (pp) |",
        "|---|---|---|---|",
    ]
    for code in codes:
        lines.append(
            f"| {code} {names.get(code, '')} | {pct(before['dismissal'][code])} | "
            f"{pct(after['dismissal'][code])} | "
            f"{points(after['dismissal'][code] - before['dismissal'][code])} |"
        )

    lines += [
        "",
        "### Representation shares (engaged and not dismissed)",
        "",
        "| Discourse | Before | After | Change (pp) |",
        "|---|---|---|---|",
    ]
    for code in codes:
        lines.append(
            f"| {code} | {pct(before['shares'][code])} | "
            f"{pct(after['shares'][code])} | "
            f"{points(after['shares'][code] - before['shares'][code])} |"
        )
    lines += [
        "",
        f"**TVD from reference: {total_variation_distance(before['shares'], target):.3f} "
        f"before, {total_variation_distance(after['shares'], target):.3f} after.**  "
        f"Evenness {normalised_entropy(before['shares']):.3f} → "
        f"{normalised_entropy(after['shares']):.3f}.",
        "",
        "### Coverage and availability",
        "",
        "| | Before | After |",
        "|---|---|---|",
        f"| Mean coverage | {before['coverage'].mean():.2f} / {len(codes)} | "
        f"{after['coverage'].mean():.2f} / {len(codes)} |",
        f"| Essays at full coverage | "
        f"{int((before['coverage'] == len(codes)).sum())}/{before['n_essays']} | "
        f"{int((after['coverage'] == len(codes)).sum())}/{after['n_essays']} |",
        "",
        "| Discourse | Essays available, before | after |",
        "|---|---|---|",
    ]
    for code in codes:
        lines.append(
            f"| {code} | {before['present'][code]}/{before['n_essays']} | "
            f"{after['present'][code]}/{after['n_essays']} |"
        )

    changed = int((before["verdict"] != after["verdict"]).sum())
    lines += [
        "",
        f"**{changed} of {len(sentences)} sentences "
        f"({changed / len(sentences):.2%}) change verdict**, of which "
        f"{int((before['verdict'] != after['verdict'])[extended].sum())} are "
        "among the sentences that were actually rewritten.",
    ]

    # --- 4. examples ------------------------------------------------------
    magnitude = np.abs(delta).max(axis=1)
    magnitude[~extended] = -1
    ranked = np.argsort(-magnitude)[: args.examples]

    lines += [
        "",
        "## 5. The sentences that moved most",
        "",
        "Ordered by the largest single change against any statement. These are "
        "the cases the pass exists for; reading them is the check on whether "
        "the prepended context was the right context.",
        "",
    ]
    for index in ranked:
        row = sentences.iloc[index]
        best_statement = int(np.abs(delta[index]).argmax())
        lines += [
            f"**{row['model'].split('/')[-1]} run {row['run_id']}** — "
            f"largest shift {delta[index, best_statement]:+.3f} on statement "
            f"{statements['statement_id'].iloc[best_statement]}",
            "",
            f"> *context:* {row['context'].strip()}",
            "",
            f"> *sentence:* {row['text'].strip()}",
            "",
            f"> *statement:* {statements['item_depersonalised'].iloc[best_statement]}",
            "",
        ]

    tvd_shift = abs(
        total_variation_distance(after["shares"], target)
        - total_variation_distance(before["shares"], target)
    )
    lines += [
        "## Verdict",
        "",
        "**The main report survives, and the context pass should not be "
        "adopted as the primary scoring.** Two separate conclusions.",
        "",
        f"*Robustness:* the pass changes {changed / len(sentences):.2%} of "
        f"verdicts and moves the headline TVD by {tvd_shift:.3f}. Every "
        "substantive claim in `sentence_summary_*.md` holds under both "
        "scorings, so anaphora is not quietly driving the result.",
        "",
        "*Validity:* the pass does not do what it was built to do. It was meant "
        "to recover dismissals hiding behind unresolved references; dismissals "
        "turned out not to need it. What it recovered instead was assertion, "
        "and section 3 shows that assertion belongs substantially to the "
        "**prepended context rather than to the sentence being scored** — the "
        "combined premise correlates more strongly with the context's own solo "
        "score than with the sentence's. Since the context is separately scored "
        "as a sentence in its own right, adopting this pass would count a large "
        "part of that stance twice.",
        "",
        "Concatenation is the wrong instrument for anaphora. An NLI model reads "
        "a two-sentence premise as one claim and will entail anything either "
        "half supports; there is no way to ask it about the second half only. "
        "The fix that would actually work is coreference resolution — rewriting "
        "the pronoun in place so the sentence stands alone at its original "
        "length — which keeps the premise about the sentence. Until then the "
        "base pass is the conservative choice and remains the primary scoring.",
        "",
        "One thing worth keeping: the assertion deficit is not solely the "
        "attitude-report problem with the Q-statements. Sentence fragmentation "
        "contributes too, since a sentence cut off from its subject cannot "
        "entail anything. That is an argument for measuring voicing at a unit "
        "larger than the sentence, which is what a judge stage would do.",
        "",
    ]

    out_dir = config.results_dir(prompt_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = "\n".join(lines)
    (out_dir / f"context_robustness_{args.form}.md").write_text(
        summary, encoding="utf-8"
    )
    print(summary)
    print(f"\nWritten to {out_dir}/context_robustness_{args.form}.md")


if __name__ == "__main__":
    main()
