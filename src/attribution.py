"""
Step 4c: whose voice carries each discourse?

Airtime asks whether a discourse's themes are engaged. Stance asks whether its
position is contradicted. Neither asks the question a reader of these essays
notices immediately: some positions are stated flatly, as things that are the
case, while others are held at arm's length and put in somebody else's mouth.

    "Genome editing could eliminate devastating inherited disease."
    "Some religious groups argue that editing the germline crosses a line."

Both give the discourse airtime. Neither is dismissed. But the first is the
essay speaking and the second is the essay reporting, and for discursive
representation that difference matters: a perspective that only ever appears as
somebody else's opinion is being described to the reader rather than made
available as a way of reasoning they might adopt.

This script measures that. Every sentence is classified as ATTRIBUTED (the view
is ascribed to some party) or DIRECT (asserted in the essay's own voice), and
the rates are broken down by discourse.

THREE THINGS THAT MAKE THIS HARDER THAN A REGEX
-----------------------------------------------
1. Attribution has scope beyond its own sentence. "Critics argue X. They also
   point to Y." -- the second sentence has no attributor of its own but is
   plainly inside the attributed span. Sentences that open with an unresolved
   reference inherit their predecessor's attribution status.

2. Impersonal attribution has no attributor. "It is often argued that X" and
   "There are concerns that X" distance the claim without naming anyone. These
   count, and are reported separately because they are a softer form.

3. Precision cannot be assumed. A detector like this is exactly the kind of
   instrument that looks authoritative and quietly measures the wrong thing, so
   --samples prints matched and unmatched sentences for inspection, and the
   corpus rate is checked against the figure obtained independently at paragraph
   level.

Usage:
    python src/attribution.py
    python src/attribution.py --baseline pre
    python src/attribution.py --samples 12
"""

import argparse
import re

import numpy as np
import pandas as pd

import config
from analyse import pct, points
from embed import load_discourses

# --------------------------------------------------------------------------
# The detector
# --------------------------------------------------------------------------

# Parties a view can be ascribed to. Deliberately broad: the essays attribute to
# stakeholder groups ("disability advocates"), to quantifiers ("many", "some"),
# to professions ("bioethicists") and to the deliberating public itself
# ("participants", "Australians").
ATTRIBUTORS = r"""
critics? | proponents? | supporters? | opponents? | advocates? | sceptics? |
skeptics? | detractors? | champions? | defenders? | campaigners? |
commentators? | observers? | authors? | scholars? | academics? |
scientists? | researchers? | clinicians? | doctors? | physicians? |
geneticists? | ethicists? | bioethicists? | philosophers? | theologians? |
lawyers? | regulators? | policymakers? | legislators? | governments? |
experts? | specialists? | professionals? | practitioners? |
patients? | families | parents | carers? | communities |
australians? | citizens? | participants? | respondents? | jurors? |
publics? | populations? | groups? | organisations? | organizations? |
institutions? | bodies | committees? | councils? | panels? |
some | others? | many | most | several | certain | few | a\s+number |
a\s+group | a\s+minority | a\s+majority | those | people | individuals? |
they | he | she | we
"""

# Verbs that report a view rather than state one, split by how safe they are.
#
# UNAMBIGUOUS verbs are almost never nouns, so an attributor followed by one is
# reliable evidence of reported speech.
UNAMBIGUOUS_VERBS = r"""
argues? | argued | arguing | contends? | contended |
maintains? | maintained | asserts? | asserted | insists? | insisted |
believes? | believed | thinks? | thought | says? | said |
emphasi[sz]es? | emphasi[sz]ed | highlights? | highlighted |
underlines? | underlined | suggests? | suggested |
proposes? | proposed | urges? | urged | warns? | warned |
denies? | denied | rejects? | rejected | opposes? | opposed |
endorses? | endorsed | invokes? | invoked | contests? | contested |
replies | replied | responds? | responded | retorts? | retorted |
acknowledges? | acknowledged | concedes? | conceded | admits? | admitted |
worried | feared | hoped | reasons \s+ that | reasoned \s+ that |
points? \s+ out | pointed \s+ out | pointing \s+ out |
points? \s+ to | pointed \s+ to
"""

# AMBIGUOUS verbs double as nouns -- "questions", "views", "claims", "fears",
# "concerns", "worries", "doubts", "supports". "Participants might carry a few
# questions into deliberation" is not reported speech, and neither is "Some
# interventions are described as somatic". These are therefore only accepted
# when followed by a complementiser or a quote, which forces the reporting
# reading.
AMBIGUOUS_VERBS = r"""
claims? | claimed | notes? | noted | observes? | observed |
states? | stated | holds? | held | stresses? | stressed |
questions? | questioned | challenges? | challenged | disputes? | disputed |
doubts? | doubted | worries | fears? | hopes? | expects? | expected |
sees? | saw | views? | viewed | regards? | regarded | feels? | felt |
supports? | supported | cautions? | cautioned | objects? | objected |
counters? | countered | frames? | framed | characteri[sz]es? | characteri[sz]ed
"""


def _alternation(block: str) -> str:
    """Turn a whitespace-formatted alternation block into a compact regex group."""
    return "(?:" + "|".join(part.strip() for part in block.strip().split("|")) + ")"


ATTRIBUTOR_GROUP = _alternation(ATTRIBUTORS)
UNAMBIGUOUS_GROUP = _alternation(UNAMBIGUOUS_VERBS)
AMBIGUOUS_GROUP = _alternation(AMBIGUOUS_VERBS)

# Up to four intervening words, so "some religious groups strongly argue" and
# "many patient advocates argue" both match while distant coincidences do not.
_GAP = r"(?:\W+\w+){0,4}?\W+"

# 1. Named attribution: an attributor plus a reporting verb. The ambiguous verbs
#    additionally require "that", a colon or an opening quote.
NAMED_ATTRIBUTION = re.compile(
    rf"""
      \b {ATTRIBUTOR_GROUP} \b {_GAP} \b {UNAMBIGUOUS_GROUP} \b
    | \b {ATTRIBUTOR_GROUP} \b {_GAP} \b {AMBIGUOUS_GROUP} \b
      \s* (?: that \b | : | ["“] )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# 2. Framing phrases that ascribe without a verb of saying.
#
#    Deliberately NOT included: "this concern", "this worry", "this fear". Those
#    are usually the essay commenting in its own voice on a concern raised
#    earlier ("This concern is particularly resonant in Australia"), and where
#    they genuinely continue an attribution the scope rule in apply_scope()
#    picks them up as `inherited` instead -- which is the correct answer,
#    because it depends on what the previous sentence did.
FRAMING_ATTRIBUTION = re.compile(
    r"""
    \b according \s+ to \b
  | \b in \s+ the \s+ (?:view|eyes|opinion|words) \s+ of \b
  | \b from \s+ the \s+ (?:perspective|standpoint|viewpoint|
                          point \s+ of \s+ view) \s+ of \b
  | \b (?:from|on|in) \s+ (?:this|that|such|their|his|her) \s+
        (?:view|viewpoint|account|reading|perspective|standpoint|logic|
           reasoning|framing) \b
    # "For some, the genome is sacred" ascribes; "compassion for those who
    # suffer" does not. The comma is what separates the two, so it is required.
  | \b for \s+ (?:critics|proponents|supporters|opponents|advocates|sceptics|
                 skeptics|some|others|many|these|those)
        (?:\s+\w+){0,3} \s* ,
  | \b (?:their|his|her) \s+ (?:view|argument|claim|position|concern|
                               objection|worry|fear|reasoning) \b
  | \b (?:proponents|opponents|critics|supporters|advocates|defenders|
          champions) \s+ of \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# 3. Impersonal attribution: distances the claim without naming anyone.
IMPERSONAL_ATTRIBUTION = re.compile(
    r"""
    \b it \s+ (?:is|has \s+ been|was) \s+ (?:often \s+|widely \s+|sometimes \s+|
        frequently \s+|commonly \s+|also \s+)? (?:argued|claimed|said|suggested|
        contended|maintained|asserted|noted|observed|feared|hoped|believed|
        thought|held) \b
  | \b there \s+ (?:is|are|has \s+ been|have \s+ been) \s+ (?:also \s+|often \s+|
        widely \s+|significant \s+|considerable \s+|growing \s+|real \s+|
        genuine \s+|some \s+|many \s+|a \s+ number \s+ of \s+)?
        (?:concerns?|worries|fears?|anxieties|objections?|arguments?|claims?|
           doubts?|scepticism|skepticism|unease|disquiet|opposition|support|
           enthusiasm|calls?|demands?) \b
  | \b (?:concerns?|worries|fears?|objections?|arguments?|doubts?|criticisms?)
        \s+ (?:have \s+ been|has \s+ been|are|is|were|was) \s+
        (?:raised|expressed|voiced|articulated|advanced|made|put) \b
  | \b (?:a|the|one) \s+ (?:common|frequent|recurring|widespread|familiar|
        further|another|second|third) \s+
        (?:concern|worry|fear|objection|argument|criticism|view|claim) \b
  | \b (?:critics|proponents|supporters|opponents|advocates|sceptics|skeptics)
        \s* [:,] |
    \b is \s+ (?:seen|viewed|regarded|understood|framed|described|
        characteri[sz]ed) \s+ (?:by|as) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Sentences about the briefing itself rather than about genome editing. They
# open with a demonstrative, so the scope rule would otherwise let them inherit
# an attribution from the sentence before: "Critics argue X. This report surveys
# the landscape." The second sentence is the essay's own voice about its own
# structure, and belongs in neither category.
META_TEXT = re.compile(
    r"""^ (?:this|the|these) \s+
        (?:report|briefing|document|paper|note|section|summary|overview|
           following|paragraphs?|list|questions?|material)
      | ^ your \s+ task
      | ^ (?:this|it) \s+ is \s+ (?:designed|intended|meant) \s+ to
    """,
    re.IGNORECASE | re.VERBOSE,
)


def classify(sentence: str) -> str:
    """
    Attribution category for a single sentence, ignoring context.

    Returns "named", "framing", "impersonal" or "direct". Checked in order of
    strength, so a sentence that both names an attributor and uses an impersonal
    construction is recorded as the stronger, named case.
    """
    if NAMED_ATTRIBUTION.search(sentence):
        return "named"
    if FRAMING_ATTRIBUTION.search(sentence):
        return "framing"
    if IMPERSONAL_ATTRIBUTION.search(sentence):
        return "impersonal"
    return "direct"


def apply_scope(table: pd.DataFrame) -> pd.Series:
    """
    Extend each attribution to the sentences that continue it.

    "Critics argue X. They also point to Y. This would be irreversible." All
    three sentences are inside the attributed span, but only the first two carry
    a marker of their own. A sentence inherits the previous sentence's status
    when it opens with an unresolved reference and makes no attribution itself.

    Scope is reset at every paragraph boundary, since a new paragraph reopens in
    the essay's own voice often enough that carrying attribution across one
    would over-count badly.
    """
    inherited = table["attribution"].to_numpy(dtype=object).copy()

    previous_key = None
    for position in range(len(table)):
        key = (table["run_id"].iat[position], table["paragraph_index"].iat[position])
        if key != previous_key:
            previous_key = key
            continue  # first sentence of a paragraph inherits nothing

        if (inherited[position] == "direct"
                and table["anaphoric"].iat[position]
                and not META_TEXT.match(table["text"].iat[position].strip())
                and inherited[position - 1] != "direct"):
            inherited[position] = "inherited"

    return pd.Series(inherited, index=table.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    parser.add_argument("--samples", type=int, default=6,
                        help="sentences of each kind to print for inspection")
    args = parser.parse_args()
    prompt_name = config.apply_overrides(args)

    processed = config.PROCESSED_DIR / prompt_name
    sentences = pd.read_csv(processed / "sentences.csv")
    paragraphs = pd.read_csv(processed / "paragraphs.csv")

    discourses = load_discourses()
    all_codes = discourses["code"].tolist()
    codes = [c for c in all_codes if c in config.ACTIVE_DISCOURSES]
    names = dict(zip(discourses["code"], discourses["name"]))

    airtime = np.load(processed / f"sentence_x_discourse_{config.BASELINE}.npy")
    airtime_frame = pd.DataFrame(airtime, columns=all_codes)
    sentences["discourse"] = np.array(codes)[
        airtime_frame[codes].to_numpy().argmax(axis=1)
    ]

    # The dismissal verdict, so directness can be separated from dismissal --
    # two different ways of not letting a discourse speak.
    verdict_path = config.results_dir(prompt_name) / "sentence_scores_depersonalised.csv"
    if verdict_path.exists():
        scores = pd.read_csv(verdict_path)
        # This is a join by row position, which is silent when it is wrong: if
        # the scoring stage ever filters rows, every verdict would attach to the
        # wrong sentence and the per-discourse rates below would be nonsense
        # that still looks plausible. Both tables carry keys, so check them.
        keys = ["run_id", "paragraph_index", "sentence_index"]
        if len(scores) != len(sentences) or not (
                scores[keys].to_numpy() == sentences[keys].to_numpy()).all():
            raise ValueError(
                f"{verdict_path.name} is not row-aligned with sentences.csv "
                f"({len(scores)} vs {len(sentences)} rows). Re-run "
                "src/analyse_sentences.py before this script."
            )
        sentences["verdict"] = scores["verdict"].to_numpy()
    else:
        sentences["verdict"] = "unknown"

    # --- classify ----------------------------------------------------------
    sentences["attribution"] = sentences["text"].map(classify)
    sentences["attribution_scoped"] = apply_scope(sentences)
    sentences["attributed"] = sentences["attribution_scoped"] != "direct"

    # --- corpus level, and the check against the paragraph figure ----------
    paragraph_attributed = paragraphs["text"].map(lambda t: classify(t) != "direct")
    # Named attribution alone, at paragraph level. This is the narrow reading --
    # an explicit attributor plus a reporting verb, nothing else -- and it is
    # the comparable figure to the 53.5% recorded in
    # pre-delib_q_statements_depersonalised.CHANGES.md, which came from a
    # narrower ad-hoc pattern than the one used here.
    paragraph_named = paragraphs["text"].map(
        lambda t: bool(NAMED_ATTRIBUTION.search(t))
    )

    lines = [
        "# Directness: whose voice carries each discourse",
        "",
        f"Baseline: **{config.baseline()['label']}**",
        "",
        f"Prompt: `{prompt_name}`  |  {len(sentences)} sentences, "
        f"{len(paragraphs)} paragraphs",
        "",
        "A discourse can be given airtime and left uncontradicted while still "
        "never being spoken in the essay's own voice. This report separates "
        "**direct** assertion from **attributed** reporting, and breaks the "
        "rate down by discourse.",
        "",
        "## 1. How much of the corpus is reported speech",
        "",
        "| Level | Attributed | Direct |",
        "|---|---|---|",
        f"| Paragraphs, any marker | {pct(paragraph_attributed.mean())} | "
        f"{pct(1 - paragraph_attributed.mean())} |",
        f"| Paragraphs, named attribution only | {pct(paragraph_named.mean())} | "
        f"{pct(1 - paragraph_named.mean())} |",
        f"| Sentences | {pct(sentences['attributed'].mean())} | "
        f"{pct(1 - sentences['attributed'].mean())} |",
        "",
        "The paragraph figures are higher than the sentence figure by "
        "construction: a paragraph counts as attributed if any part of it is, "
        f"and paragraphs here average "
        f"{len(sentences) / max(len(paragraphs), 1):.1f} sentences.",
        "",
        "A figure of **53.5%** is recorded in "
        "`pre-delib_q_statements_depersonalised.CHANGES.md` from a narrower "
        "ad-hoc pattern used before this script existed. It sits in the same "
        "region as the narrow row above, but the script that produced it no "
        "longer exists and the number cannot be regenerated, so it is a "
        "historical note and **not** a validation of this detector. The "
        "detector's precision and recall have never been measured against "
        "hand-labelled sentences; `--samples` prints matched and unmatched "
        "cases for inspection, which catches systematic error but does not "
        "quantify what is left. Read the rates below as comparative between "
        "discourses, not as calibrated levels.",
        "",
        "Breakdown of the sentence-level classification:",
        "",
        "| Category | Sentences | Share | What it looks like |",
        "|---|---|---|---|",
    ]
    descriptions = {
        "named": "an attributor and a reporting verb — *critics argue that*",
        "framing": "ascription without a verb of saying — *on this view*, *for opponents*",
        "impersonal": "distanced but unascribed — *it is often argued*, *there are concerns*",
        "inherited": "continues an attribution from the previous sentence",
        "direct": "asserted in the essay's own voice",
    }
    counts = sentences["attribution_scoped"].value_counts()
    for category in ["named", "framing", "impersonal", "inherited", "direct"]:
        count = int(counts.get(category, 0))
        lines.append(
            f"| `{category}` | {count} | {pct(count / len(sentences))} | "
            f"{descriptions[category]} |"
        )

    # --- 2. by discourse ---------------------------------------------------
    lines += [
        "",
        "## 2. Directness by discourse",
        "",
        "Of the sentences that engage each discourse's themes, the share put in "
        "somebody else's mouth. A high rate means the discourse reaches the "
        "reader as a report of what others think rather than as a claim about "
        "the world.",
        "",
        "| Discourse | Sentences | Attributed | Direct | vs corpus (pp) |",
        "|---|---|---|---|---|",
    ]
    corpus_rate = sentences["attributed"].mean()
    by_discourse = {}
    for code in codes:
        group = sentences[sentences["discourse"] == code]
        rate = float(group["attributed"].mean()) if len(group) else float("nan")
        by_discourse[code] = rate
        lines.append(
            f"| {code} {names.get(code, '')} | {len(group)} | {pct(rate)} | "
            f"{pct(1 - rate)} | {points(rate - corpus_rate)} |"
        )

    ordered = sorted(by_discourse.items(), key=lambda kv: -kv[1])
    most, least = ordered[0], ordered[-1]
    lines += [
        "",
        f"**Spread: {points(most[1] - least[1]).lstrip('+')} percentage points** "
        f"between {most[0]} ({names.get(most[0], '')}, {pct(most[1])} attributed) "
        f"and {least[0]} ({names.get(least[0], '')}, {pct(least[1])}).",
    ]

    # --- 3. by discourse and model ----------------------------------------
    lines += [
        "",
        "### By model",
        "",
        "| Model | " + " | ".join(codes) + " |",
        "|---" * (len(codes) + 1) + "|",
    ]
    for model, group in sentences.groupby("model"):
        cells = []
        for code in codes:
            subset = group[group["discourse"] == code]
            cells.append(pct(subset["attributed"].mean()) if len(subset) else "—")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")

    # --- 4. directness against dismissal -----------------------------------
    if (sentences["verdict"] != "unknown").any():
        lines += [
            "",
            "## 3. Two different ways of not letting a discourse speak",
            "",
            "Attribution and dismissal are separate failures. A discourse can be "
            "reported at length and never contradicted, or asserted directly and "
            "then denied. Reading them together says more than either alone.",
            "",
            "| Discourse | Attributed | Dismissed | Direct *and* not dismissed |",
            "|---|---|---|---|",
        ]
        for code in codes:
            group = sentences[sentences["discourse"] == code]
            if not len(group):
                continue
            dismissed = float((group["verdict"] == "dismissed").mean())
            own_voice = float(
                (~group["attributed"] & (group["verdict"] != "dismissed")).mean()
            )
            lines.append(
                f"| {code} | {pct(group['attributed'].mean())} | "
                f"{pct(dismissed)} | {pct(own_voice)} |"
            )
        lines += [
            "",
            "The last column is the strictest reading of representation "
            "available from the automated measures: the discourse's themes are "
            "engaged, the essay speaks in its own voice, and nothing contradicts "
            "it.",
        ]

    # --- 5. samples for inspection ----------------------------------------
    lines += [
        "",
        "## 4. Samples, for checking the detector",
        "",
        "A pattern-based classifier is exactly the kind of instrument that looks "
        "authoritative while measuring something adjacent to the intended "
        "target. These are drawn at random with a fixed seed; the question to "
        "ask of each is whether the label is right.",
        "",
    ]
    generator = np.random.default_rng(0)
    for category in ["named", "framing", "impersonal", "inherited", "direct"]:
        pool = sentences[sentences["attribution_scoped"] == category]
        if pool.empty:
            continue
        picked = pool.iloc[
            generator.choice(len(pool), size=min(args.samples, len(pool)),
                             replace=False)
        ]
        lines.append(f"**`{category}`**")
        lines.append("")
        for text in picked["text"]:
            lines.append(f"- {text.strip()}")
        lines.append("")

    lines += [
        "## Caveats",
        "",
        "- The detector is lexical. It finds the *markers* of attribution, not "
        "attribution itself, and will miss a view ascribed by context alone.",
        "- `direct` is a residual category, so every false negative lands in it. "
        "The direct rate is therefore an upper bound and the attributed rate a "
        "lower bound.",
        "- Attribution scope is inherited only by sentences that open with an "
        "unresolved reference, and never across a paragraph boundary. Both "
        "choices are conservative, and both push the attributed rate down.",
        "- Which discourse a sentence belongs to comes from the same "
        "cosine argmax used everywhere else, with the same weaknesses — see "
        "`src/validate_baseline.py`. The comparison between discourses is made "
        "under a constant bias, so the ordering is sturdier than the levels.",
        "",
    ]

    out_dir = config.results_dir(prompt_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    sentences.to_csv(out_dir / "attribution.csv", index=False)
    summary = "\n".join(lines)
    (out_dir / "attribution.md").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"\nWritten to {out_dir}/attribution.md")


if __name__ == "__main__":
    main()
