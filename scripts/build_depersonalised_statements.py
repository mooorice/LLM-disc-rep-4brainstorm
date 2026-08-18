"""
Build the depersonalised variant of the 46 Q-statements.

Why this exists
---------------
Two thirds of the Q-statements are phrased as attitude reports -- "I'm
concerned that X", "I think X", "it makes me sick to hear X". That phrasing
defeats NLI scoring on this corpus. The essays are largely reported speech, and
"many Australians worry that X" does not entail "*I* am concerned that X": the
model correctly returns neutral, and the measure collapses to nothing. Tested
directly, attitude-report hypotheses gave a mean voiced-minus-dismissed gap of
-0.294; the same cases with the wrapper stripped gave +0.789.

The transformation keeps the proposition and removes the attitude wrapper. That
is defensible because of what the Q-sort measured in the first place: a
participant placing "I'm concerned that X" at +2 is expressing agreement with
the proposition that X is worrying. The wrapper is a survey convention, not part
of the content the z-score attaches to.

It is still an edit to the instrument, so both forms are kept in the output and
every change is classified. See the accompanying CHANGES.md for the full
rationale and the cases where the transformation is not neutral.

Usage:
    python scripts/build_depersonalised_statements.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402

# statement_id -> (depersonalised text, transformation class)
#
#   unchanged  - already a proposition; text is copied verbatim
#   repaired   - grammar fixed only, no change of meaning
#   stripped   - attitude wrapper removed, proposition otherwise intact
#   recast     - wrapper removal required rewording that is NOT meaning-neutral;
#                these are the ones to argue about, and they are listed
#                individually in CHANGES.md
REWRITES = {
    1:  ("Mitochondrial donation doesn't fundamentally change our genetics and should be viewed like any other organ transplant.", "repaired"),
    2:  ("It's a slippery slope from genome editing for medical reasons to using it for other reasons.", "unchanged"),
    3:  ("Parents and guardians have a right to edit the genes of their children before they are born.", "unchanged"),
    4:  ("Parents want 'designer babies' with certain traits out of vanity.", "unchanged"),
    5:  ("Genome editing will lead to a reduction in genetic diversity.", "stripped"),
    6:  ("Being able to choose the way someone behaves would be a good thing.", "recast"),
    7:  ("The idea that gene editing can make perfect people and societies is extremely dangerous.", "unchanged"),
    8:  ("It is acceptable to use genome editing to change someone's appearance.", "stripped"),
    9:  ("Gene editing will be used as a quick techno-fix without actually dealing with real problems.", "stripped"),
    10: ("As we start to tinker with the genome, we are going to accidentally introduce a whole host of other problems.", "stripped"),
    11: ("If this technology is available then it should be available to everyone who needs it, not just those who can afford it.", "unchanged"),
    12: ("Genetic technology will widen the gap between those who can afford it and those who can't.", "stripped"),
    13: ("If we open the box of genome editing we will not know where to draw the line.", "unchanged"),
    14: ("The use of genome editing will advantage one group over another.", "stripped"),
    15: ("Decisions about the use of genome editing are best left to the experts.", "unchanged"),
    16: ("If genome editing works, the future will be like a dystopian movie.", "unchanged"),
    17: ("Genome editing is really cool science.", "unchanged"),
    18: ("The real aim of genome editing will be enhancement for wealthy people.", "unchanged"),
    19: ("Gene editing in humans is fundamentally different from gene editing in animals or plants.", "recast"),
    20: ("Editing genes that will be inherited is problematic because it means making decisions for future people who don't exist yet.", "unchanged"),
    21: ("Cultural beliefs are a reason to be cautious about genome editing.", "recast"),
    22: ("When you change someone's genes you are fundamentally changing who they are.", "unchanged"),
    23: ("Genome editing in agriculture will have the biggest impact on people's health.", "unchanged"),
    24: ("Anything that has been artificially created in a lab is unnatural and unsettling.", "recast"),
    25: ("Something will go terribly wrong with this technology.", "stripped"),
    26: ("It's better to prevent a disease from happening rather than it happening and then having to treat it.", "unchanged"),
    27: ("It would be miraculous and incredible if people with genetic conditions were cured within our lifetimes.", "stripped"),
    28: ("Genome editing technologies give us power over human life itself.", "repaired"),
    29: ("It is offensive to speak of genome editing as a 'cure' for disability.", "recast"),
    30: ("It is acceptable to use genome editing to speed up evolution.", "stripped"),
    31: ("As a society, we have an obligation to help strengthen coming generations by eradicating diseases.", "unchanged"),
    32: ("We find strength when we face illness and adversity. It makes us who we are.", "unchanged"),
    33: ("If genome editing is proven to be safe, there is no reason to oppose it.", "stripped"),
    34: ("It is better to accept our genes the way they are than to tamper with things we don't understand.", "unchanged"),
    35: ("It is acceptable for parents to use genome editing if it could safely give their children an advantage in life.", "recast"),
    36: ("We do not know enough about genetic diseases or the long-term consequences of genome editing.", "stripped"),
    37: ("We should keep an open mind about new discoveries.", "stripped"),
    38: ("It is acceptable for a person to have their own genome edited.", "recast"),
    39: ("Each individual has a right to decide for themselves whether to undergo gene editing.", "unchanged"),
    40: ("Interfering in people's genes is intervening a bit too far in human creation.", "unchanged"),
    41: ("The idea of editing the human genome is frightening.", "stripped"),
    42: ("The use of gene editing for non-medical reasons recalls the Nazis during the Second World War.", "stripped"),
    43: ("Science and technology are controlling our lives.", "unchanged"),
    44: ("Future humans will need genome editing to survive.", "stripped"),
    45: ("We should remain as natural as possible but we should not shy away from furthering medicine and the chance to save lives.", "stripped"),
    46: ("If you can have strong genes, you will have wellbeing.", "unchanged"),
}


def main() -> None:
    source = pd.read_csv(config.FACTOR_ARRAY_FILE, encoding="utf-8-sig")
    source = source.rename(columns={"No": "statement_id", "Item": "item_original"})

    missing = set(source["statement_id"]) - set(REWRITES)
    if missing:
        raise ValueError(f"No rewrite provided for statements {sorted(missing)}")

    source["item_depersonalised"] = source["statement_id"].map(
        lambda i: REWRITES[i][0]
    )
    source["transformation"] = source["statement_id"].map(lambda i: REWRITES[i][1])

    # Curly apostrophes in the source are normalised so the two columns can be
    # compared and tokenised consistently.
    source["item_original"] = source["item_original"].str.replace("’", "'")

    columns = ["statement_id", "item_original", "item_depersonalised",
               "transformation", "A", "B", "C", "D"]
    out_path = config.DEPERSONALISED_FILE
    source[columns].to_csv(out_path, index=False)

    print(f"Wrote {len(source)} statements to {out_path}\n")
    print(source["transformation"].value_counts().to_string())
    print("\nThe 'recast' cases are the ones whose meaning may have shifted; "
          "they are argued individually in the CHANGES file.")
    for row in source[source["transformation"] == "recast"].itertuples():
        print(f"\n  {row.statement_id}. {row.item_original}")
        print(f"   -> {row.item_depersonalised}")


if __name__ == "__main__":
    main()
