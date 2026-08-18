"""
Validation: can the scorer tell endorsement from rejection?

This is the test that decides whether the whole stance pipeline is worth having.
Each case below is a matched pair: two passages on the same topic, written with
deliberately overlapping vocabulary, one asserting a Q-statement and one denying
it. A measure that tracks agreement must separate them; a measure that tracks
topic cannot.

Cosine similarity fails this test badly. It scored the endorsing passage higher
in only 2 of 6 cases, with a mean gap of -0.039 -- that is, rejecting passages
were on average *closer* to the statement than endorsing ones. The mechanism is
plain once seen: to reject a claim you must first restate it and then argue
against it, so a rejection contains the proposition plus extra on-topic
material, making it lexically denser on the statement than a sympathetic
passage that merely gestures at the idea.

Run with --cosine to reproduce that failure alongside the NLI result.

Usage:
    python src/validate_stance.py
    python src/validate_stance.py --cosine
"""

import argparse

import numpy as np
import pandas as pd

import config
import stance

# Statements are quoted verbatim from the factor array; the passages are written
# to mirror each other's vocabulary as closely as the opposing stance allows.
PAIRS = [
    {
        "statement": "It's a slippery slope from genome editing for medical reasons to using it for other reasons.",
        "endorse": "Once genome editing is permitted for medical reasons it will not stop there. The boundary between treating disease and enhancing traits is porous, and each approved use makes the next one easier to justify. We would slide from curing illness to selecting for height or intelligence without ever taking a decision to do so.",
        "reject": "The idea that medical genome editing leads inevitably to enhancement does not survive scrutiny. The boundary between treating disease and selecting for traits is one regulators draw all the time in other fields, and there is no evidence that permitting therapeutic use creates any slide towards editing for height or intelligence.",
    },
    {
        "statement": "Parents and guardians have a right to edit the genes of their children before they are born.",
        "endorse": "Decisions about a future child's health belong to that child's parents. Families already make far-reaching medical choices on behalf of their children, and choosing to remove a devastating inherited condition before birth is continuous with that established parental authority rather than a departure from it.",
        "reject": "Decisions about a future child's genome are not the parents' to make. Families make medical choices for their children all the time, but altering a child's genes before birth binds a person who cannot consent and who must live with the result, which puts it outside legitimate parental authority.",
    },
    {
        "statement": "I'm concerned that genome editing will lead to a reduction in genetic diversity.",
        "endorse": "If enough families make the same editing choices, the population's genetic variation will narrow. Traits now spread across the community would be steadily removed, and that loss of genetic diversity leaves us more fragile as a species than we are today.",
        "reject": "Worries about genome editing narrowing our genetic variation are overstated. The number of families who would use these techniques is small, the edits are targeted at specific disease alleles, and the effect on genetic diversity across the population as a whole would be negligible.",
    },
    {
        "statement": "Parents want 'designer babies' with certain traits out of vanity.",
        "endorse": "Much of the demand here is simple vanity. Parents will want taller, smarter, better-looking children, and once the technology exists they will use it to design offspring who reflect well on them rather than to prevent any real suffering.",
        "reject": "The caricature of vain parents designing taller or better-looking children bears little relation to who actually seeks these technologies. The families involved are overwhelmingly those facing a devastating inherited condition, not parents shopping for traits that reflect well on them.",
    },
    {
        "statement": "Mitochondrial donation doesn't fundamentally change our genetics should be viewed like any other organ transplant.",
        "endorse": "Mitochondrial donation replaces a faulty cellular component and leaves the nuclear genome that carries our traits untouched. It is far closer to an organ transplant than to genetic engineering, and it should be regulated as the former.",
        "reject": "Calling mitochondrial donation a kind of organ transplant understates what it does. It introduces heritable genetic material from a third person that passes to every future generation, which is not something any transplant does, and treating it as routine obscures a genuine change to our genetics.",
    },
    {
        "statement": "Genome editing is unnatural and interferes with the natural order.",
        "endorse": "There is something about rewriting the human genome that crosses a line we were not meant to cross. Life has an integrity that comes from not being engineered, and interfering at that level puts us in a role we should not occupy.",
        "reject": "The claim that editing the genome offends against some natural order does not hold up. Medicine has always intervened in what nature dealt us, from antibiotics to transplants, and there is no coherent line at which intervention suddenly becomes a violation of an order we were meant to respect.",
    },
]


# --------------------------------------------------------------------------
# Second validation: attitude-report hypotheses against reported speech
# --------------------------------------------------------------------------
#
# Two thirds of the Q-statements are phrased as attitude reports -- "I'm
# concerned that X", "I think X" -- rather than as claims about the world. That
# is a problem here, because the essays are themselves largely reported speech:
# 53.5% of paragraphs contain an attributed-view construction. If NLI reads
# "many Australians worry that X" as entailing "I'm concerned that X" purely
# because a concern is mentioned, the measure collapses back into aboutness and
# buys us nothing over cosine.
#
# The test that matters is therefore not endorse-vs-reject but
# voiced-vs-dismissed: does a passage that raises a concern and lets it stand
# score higher than one that raises the same concern and defuses it? That is
# precisely the distinction the whole pipeline exists to make.
REPORTED_CASES = [
    {
        "statement": "I'm concerned that genome editing will lead to a reduction in genetic diversity.",
        "voiced": "Many Australians are deeply troubled by the prospect of a narrowing gene pool. They see the steady removal of variation from the population as a loss we would not be able to reverse, and one that would leave the species more fragile.",
        "dismissed": "Some Australians raise the prospect of a narrowing gene pool, though geneticists generally regard this as unfounded. The number of families involved would be tiny and the edits highly targeted, so the effect on population-level variation would be undetectable.",
        "absent": "The regulatory framework in Australia is set out across several instruments. Reproductive cloning is prohibited, embryo research is licensed, and mitochondrial donation was permitted under strict conditions in 2022.",
    },
    {
        "statement": "My cultural beliefs make me cautious about genome editing.",
        "voiced": "For many Aboriginal and Torres Strait Islander participants, and for people from a range of faith traditions, this question cannot be separated from cultural and spiritual understandings of the body. Those understandings counsel real caution about intervening in inheritance.",
        "dismissed": "Cultural and religious objections are sometimes raised in this debate, but they tend to soften once people understand what the technology actually involves, and they are not a sound basis for restricting access for everybody else.",
        "absent": "Clinical trials of somatic gene therapy have shown durable responses in sickle cell disease and certain inherited blindnesses, and several such therapies now hold regulatory approval overseas.",
    },
    {
        "statement": "I'm worried that genetic technology will widen the gap between those who can afford it and those who can't.",
        "voiced": "Cost is at the front of many people's minds. If these therapies arrive through private clinics at prices only wealthy families can meet, the result will be a health system that entrenches advantage rather than levelling it.",
        "dismissed": "Concerns about a two-tier system are often voiced, but Australia's universal healthcare arrangements make that outcome unlikely. Where therapies have proved effective they have generally been brought into public subsidy within a few years.",
        "absent": "Genome editing techniques differ considerably in precision. Base editing and prime editing avoid the double-strand breaks associated with earlier CRISPR approaches and carry different off-target profiles.",
    },
]


def run_reported() -> pd.DataFrame:
    """Can the scorer tell a voiced concern from a defused one?"""
    model, tokenizer, device, indices = stance.load_nli()

    kinds = ["voiced", "dismissed", "absent"]
    premises, hypotheses = [], []
    for case in REPORTED_CASES:
        for kind in kinds:
            premises.append(case[kind])
            hypotheses.append(case["statement"])

    scores = stance.score_pairs(model, tokenizer, device, indices,
                                premises, hypotheses, batch_size=16)
    signed = (scores[:, 0] - scores[:, 1]).reshape(len(REPORTED_CASES), len(kinds))

    table = pd.DataFrame(signed, columns=kinds)
    table.insert(0, "statement",
                 [c["statement"][:40] + "..." for c in REPORTED_CASES])
    table["voiced-dismissed"] = table["voiced"] - table["dismissed"]

    print("\n--- Attitude-report statements against reported speech ---\n")
    print(table.round(3).to_string(index=False))
    print(f"\nvoiced scored above dismissed: "
          f"{(table['voiced-dismissed'] > 0).sum()}/{len(table)}")
    print(f"mean voiced-dismissed gap: {table['voiced-dismissed'].mean():+.3f}")
    print(f"mean score on an unrelated passage: {table['absent'].mean():+.3f} "
          "(should sit near zero)")
    return table


def report(name: str, endorse: np.ndarray, reject: np.ndarray) -> pd.DataFrame:
    """Print the endorse/reject comparison for one scoring method."""
    rows = []
    for i, pair in enumerate(PAIRS):
        rows.append({
            "statement": pair["statement"][:44] + "...",
            "endorse": round(float(endorse[i]), 3),
            "reject": round(float(reject[i]), 3),
            "gap": round(float(endorse[i] - reject[i]), 3),
            "ok": "yes" if endorse[i] > reject[i] else "NO",
        })
    table = pd.DataFrame(rows)

    print(f"\n--- {name} ---\n")
    print(table.to_string(index=False))
    gaps = table["gap"].to_numpy()
    print(f"\nendorse scored above reject: {(gaps > 0).sum()}/{len(gaps)}")
    print(f"mean gap: {gaps.mean():+.3f}")
    return table


def run_nli() -> pd.DataFrame:
    """Score every pair with the NLI model."""
    model, tokenizer, device, indices = stance.load_nli()

    premises, hypotheses = [], []
    for pair in PAIRS:
        premises += [pair["endorse"], pair["reject"]]
        hypotheses += [pair["statement"]] * 2

    scores = stance.score_pairs(model, tokenizer, device, indices,
                                premises, hypotheses, batch_size=16)
    signed = scores[:, 0] - scores[:, 1]

    table = report("NLI  P(entail) - P(contradict)", signed[0::2], signed[1::2])
    print(f"endorse scored positive: {(table['endorse'] > 0).sum()}/{len(table)}")
    print(f"reject scored negative : {(table['reject'] < 0).sum()}/{len(table)}")
    return table


def run_cosine() -> pd.DataFrame:
    """Score every pair with the embedding model, to reproduce the failure."""
    import embed

    model = embed.load_model()
    statements = embed.encode(model, [p["statement"] for p in PAIRS], as_query=False)
    endorse = embed.encode(model, [p["endorse"] for p in PAIRS], as_query=False)
    reject = embed.encode(model, [p["reject"] for p in PAIRS], as_query=False)

    return report(
        "Cosine similarity",
        np.einsum("ij,ij->i", endorse, statements),
        np.einsum("ij,ij->i", reject, statements),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosine", action="store_true",
                        help="also score with cosine similarity, for contrast")
    args = parser.parse_args()

    if args.cosine:
        run_cosine()
    run_nli()
    run_reported()

    print("\nA measure fit for the factor arrays must separate these pairs. "
          "Cosine does not; if NLI does not either, the stance pipeline "
          "should be abandoned rather than reported.")


if __name__ == "__main__":
    main()
