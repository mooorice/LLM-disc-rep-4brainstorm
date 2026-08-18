"""
Build the post-deliberative (six discourse) benchmark files.

Two artefacts are produced in data/human_baseline/:

  * post-delib_discourses_clean.txt  -- the six discourse narratives, unwrapped
    and de-jargoned in exactly the way the four pre-deliberative ones were, so
    that the two baselines are comparable as embedding targets.

  * post-delib_factor_array.csv      -- Table 6 of OP12, the z-score of each
    surveyed statement under each of the six post-deliberative discourses.

The factor array is transcribed by hand from a PDF table, which is exactly the
kind of step that fails silently. It is therefore checked against three
independent quantities the report publishes separately, before anything is
written to disk (see validate()). If a check fails the script refuses to write.

Usage:
    python scripts/build_post_delib_baseline.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "data" / "human_baseline"

SOURCE_NARRATIVES = BASELINE_DIR / "post-delib_discourses.txt"
SOURCE_MAPPING_ARRAY = BASELINE_DIR / "pre-delib_discourse_mapping_factor_array.csv"

OUT_NARRATIVES = BASELINE_DIR / "post-delib_discourses_clean.txt"
OUT_ARRAY = BASELINE_DIR / "post-delib_factor_array.csv"

CODES = ["A", "B", "C", "D", "E", "F"]

# --------------------------------------------------------------------------
# Table 6: Factor Scores -- 6-Discourse Map (OP12 p. 142)
# --------------------------------------------------------------------------
# Statement numbering matches Table 1 / Table 2, i.e. the same instrument the
# mapping study used. Statement 46 is absent on purpose: OP12 p. 124 note reads
# "Statement 46 was dropped following the Mapping Study and not used in
# subsequent surveys", so the post-deliberative sorts covered 45 items.
#
# Column order is A B C D E F, confirmed in validate() below.
FACTOR_SCORES = {
    1:  (+0.4, +1.8, -0.9, -0.8, +0.7, +0.4),
    2:  (-0.8, +1.5, +1.3, +0.7, -0.1, +1.1),
    3:  (+0.8, -0.5, -1.5, +0.7, -1.0, +1.5),
    4:  (+0.1, +0.0, +0.6, +0.9, -1.2, +0.4),
    5:  (-0.6, -1.3, +0.6, -0.6, -0.5, -0.8),
    6:  (+0.0, -1.7, -1.3, -2.1, -1.8, +0.0),
    7:  (-0.4, +0.9, +1.4, +0.9, +1.4, +0.0),
    8:  (+0.3, -1.6, -2.0, -0.8, -1.9, +0.0),
    9:  (-0.7, -0.9, +0.7, -0.3, +0.5, +0.8),
    10: (+0.4, +1.1, +1.3, +0.6, -0.3, +1.1),
    11: (+1.2, +1.7, +0.0, +1.4, +2.2, -0.4),
    12: (+0.3, +1.7, +0.7, +0.3, +0.7, -0.8),
    13: (-0.4, -0.2, +1.3, -0.1, -0.6, +0.8),
    14: (+0.2, +1.0, +1.6, +0.0, -0.1, +0.8),
    15: (-0.4, -0.6, -0.9, +1.3, +0.4, +1.9),
    16: (-2.0, -0.6, +0.2, +1.0, +0.6, -0.4),
    17: (+1.5, +1.7, +0.0, +0.9, +1.0, +1.5),
    18: (-0.3, +0.1, +0.6, -0.5, -0.9, +0.8),
    19: (+0.1, -1.1, -1.2, -0.2, -0.4, +0.0),
    20: (-0.2, -0.6, +0.7, -0.7, -0.3, -1.1),
    21: (-1.1, -1.1, +0.1, -0.3, -1.4, -1.9),
    22: (-1.3, -0.8, +0.4, -0.2, -0.9, -1.5),
    23: (-0.2, -1.2, +0.1, -0.9, -0.4, -1.1),
    24: (-1.6, -1.2, -0.3, -0.8, -1.1, -1.9),
    25: (-0.2, +0.6, +1.2, +0.2, -0.2, +0.0),
    26: (+1.4, +0.8, -0.5, +1.8, +1.2, +1.5),
    27: (+1.9, +1.3, +0.3, +1.7, +0.6, +1.9),
    28: (-0.8, +0.0, +0.4, -0.2, -1.4, +0.0),
    29: (+0.2, +0.0, -0.2, -0.5, -1.2, -1.1),
    30: (+0.9, -0.3, -1.4, -0.4, -1.1, -0.8),
    31: (+1.7, +0.6, -0.1, +1.2, +1.3, +1.1),
    32: (+1.0, +1.1, +1.4, -1.2, +1.5, -0.4),
    33: (+1.4, +0.1, -1.0, +1.2, +0.9, -1.1),
    34: (-0.5, -1.3, +0.4, -1.5, -0.5, -1.5),
    35: (-0.3, +0.3, -1.7, -1.2, -1.1, +0.4),
    36: (+0.5, +1.1, +0.5, +1.4, -0.2, +0.0),
    37: (+1.7, +1.4, +0.7, +1.1, +0.7, +0.8),
    38: (+1.4, +0.1, -1.8, -1.0, +0.6, +0.4),
    39: (+0.4, -0.4, -0.2, -0.2, +1.4, +1.1),
    40: (-1.9, -1.3, +0.2, -1.3, +0.3, -1.5),
    41: (-1.4, -0.3, +0.3, -0.3, -0.1, -0.4),
    42: (-1.5, -0.5, -0.4, -0.3, +1.2, -0.4),
    43: (-0.8, -0.7, -0.5, -1.5, +0.4, -0.8),
    44: (-1.0, -0.5, -2.3, -1.2, -0.6, +0.4),
    45: (+0.4, -0.1, +1.3, +1.7, +1.7, -0.8),
}

# --------------------------------------------------------------------------
# Published correlations used as transcription checks
# --------------------------------------------------------------------------

# Table 5: Correlations -- Post-Deliberation (Six) Discourses, OP12 p. 140.
TABLE_5 = {
    ("A", "B"): 0.56, ("A", "C"): -0.10, ("A", "D"): 0.44,
    ("A", "E"): 0.35, ("A", "F"): 0.48,
    ("B", "C"): 0.35, ("B", "D"): 0.53, ("B", "E"): 0.53, ("B", "F"): 0.45,
    ("C", "D"): 0.30, ("C", "E"): 0.27, ("C", "F"): -0.04,
    ("D", "E"): 0.47, ("D", "F"): 0.45,
    ("E", "F"): 0.21,
}

# Table 4: Correlations -- Mapping Study (Four) vs Post-Deliberation, OP12 p. 140.
# Rows are post-deliberative discourses, columns mapping-study discourses.
TABLE_4 = {
    "A": {"A": 0.77, "B": 0.11, "C": 0.12, "D": 0.39},
    "B": {"A": 0.58, "B": 0.50, "C": 0.40, "D": 0.16},
    "C": {"A": -0.13, "B": 0.73, "C": 0.42, "D": 0.22},
    "D": {"A": 0.57, "B": 0.50, "C": 0.25, "D": 0.26},
    "E": {"A": 0.57, "B": 0.47, "C": 0.48, "D": 0.34},
    "F": {"A": 0.59, "B": 0.24, "C": 0.10, "D": 0.23},
}

# Table 3: Correlations -- Mapping Study Discourses, OP12 p. 134. Used as a
# control: it validates the *method* of the check (correlating factor arrays
# reproduces the reported figures) against data we did not transcribe here.
TABLE_3 = {
    ("A", "B"): 0.32, ("A", "C"): 0.22, ("A", "D"): 0.36,
    ("B", "C"): 0.48, ("B", "D"): 0.28, ("C", "D"): 0.34,
}

# Tolerance for the checks. The transcribed array is rounded to one decimal
# place while the report's correlations were computed on unrounded scores, so
# exact agreement is not expected; anything beyond this is a transcription
# error rather than rounding.
CORRELATION_TOLERANCE = 0.08

# --------------------------------------------------------------------------
# Narrative cleaning
# --------------------------------------------------------------------------

# Sentences describing the study's own mechanics rather than the discourse.
# Removed for the same reason the equivalents were removed from the four
# pre-deliberative descriptions: they say nothing about the way of reasoning,
# and they inject sample-size vocabulary into a text used as an embedding
# target. The deleted text is reproduced in the CHANGES file.
DROP_SENTENCES = [
    "Only one Australian Citizens' Jury participant was strongly associated "
    "with this position.",
]


def normalise_quotes(text: str) -> str:
    """Curly quotes and dashes to their straight ASCII equivalents."""
    for curly, straight in [("’", "'"), ("‘", "'"),
                            ("“", '"'), ("”", '"')]:
        text = text.replace(curly, straight)
    return text


def clean_narratives() -> str:
    """
    Unwrap the six narratives into one line per paragraph and drop study
    mechanics, matching the treatment of pre-delib_discourses_clean.txt.

    The header lines read "Position A: ..." in the source. They are rewritten to
    "Discourse A: ..." so that both baselines parse with the same loader; the
    position *names* are untouched.
    """
    raw = normalise_quotes(SOURCE_NARRATIVES.read_text(encoding="utf-8"))
    blocks = [b.strip() for b in raw.split("###") if b.strip()]

    if len(blocks) != 6:
        raise ValueError(f"Expected 6 narrative blocks, found {len(blocks)}")

    cleaned_blocks = []
    for block in blocks:
        header, _, body = block.partition("\n")
        header = header.strip()

        match = re.match(r"^(?:Position|Discourse)\s+([A-F])\s*:\s*(.+)$", header)
        if not match:
            raise ValueError(f"Unrecognised header: {header!r}")
        code, name = match.group(1), match.group(2).strip()

        # Blank lines separate paragraphs; hard wraps inside a paragraph are
        # artefacts of the PDF and are removed.
        paragraphs = []
        for paragraph in re.split(r"\n\s*\n", body):
            text = " ".join(paragraph.split())
            for unwanted in DROP_SENTENCES:
                text = text.replace(unwanted, "").strip()
            text = re.sub(r"\s{2,}", " ", text)
            if text:
                paragraphs.append(text)

        cleaned_blocks.append(
            f"Discourse {code}: {name}\n\n" + "\n\n".join(paragraphs)
        )

    return "\n\n###\n\n".join(cleaned_blocks) + "\n"


# --------------------------------------------------------------------------
# Factor array
# --------------------------------------------------------------------------


def build_array() -> pd.DataFrame:
    """Join the transcribed z-scores to the statement texts."""
    mapping = pd.read_csv(SOURCE_MAPPING_ARRAY)
    mapping = mapping.rename(columns={"No": "statement_id", "Item": "item"})
    mapping["item"] = mapping["item"].map(normalise_quotes)

    ids = sorted(FACTOR_SCORES)
    if ids != list(range(1, 46)):
        raise ValueError("Expected statements 1-45 in the transcribed table.")

    rows = []
    for statement_id in ids:
        item = mapping.loc[mapping["statement_id"] == statement_id, "item"]
        if item.empty:
            raise ValueError(f"No statement text for id {statement_id}")
        rows.append({
            "statement_id": statement_id,
            "item": item.iloc[0],
            **dict(zip(CODES, FACTOR_SCORES[statement_id])),
        })
    return pd.DataFrame(rows)


def correlations(frame: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """Pearson correlation between discourse columns, across statements."""
    return frame[codes].corr()


def validate(post: pd.DataFrame) -> list[str]:
    """
    Check the transcription against the report's own published numbers.

    Returns a list of failure messages; empty means everything agreed. Three
    checks, each able to catch a different kind of error:

      1. Table 3 reproduced from the *pre*-deliberative array. This validates the
         method, using data transcribed in an earlier session, so a failure here
         means the check itself is wrong rather than the new data.
      2. Table 5 reproduced from the new array. Catches wrong values and, more
         importantly, wrong column order -- swapping two discourse columns
         permutes this matrix in a way that will not survive fifteen comparisons.
      3. Table 4 reproduced by correlating the new array against the old one.
         An independent anchor: it ties the six-discourse columns to the four
         pre-deliberative ones, whose identities are already established.
    """
    failures = []

    mapping = pd.read_csv(SOURCE_MAPPING_ARRAY).rename(columns={"No": "statement_id"})
    # The post-deliberative sorts covered statements 1-45 only, so every
    # comparison is made on that common subset.
    common = mapping[mapping["statement_id"].isin(post["statement_id"])]
    common = common.sort_values("statement_id").reset_index(drop=True)
    post = post.sort_values("statement_id").reset_index(drop=True)

    # 1. Control: does the method reproduce Table 3 from the old array?
    pre_corr = correlations(common, ["A", "B", "C", "D"])
    for (left, right), expected in TABLE_3.items():
        actual = pre_corr.loc[left, right]
        if abs(actual - expected) > CORRELATION_TOLERANCE:
            failures.append(
                f"[control, Table 3] MS {left}-{right}: "
                f"report {expected:+.2f}, computed {actual:+.2f}"
            )

    # 2. Table 5: the six discourses against each other.
    post_corr = correlations(post, CODES)
    for (left, right), expected in TABLE_5.items():
        actual = post_corr.loc[left, right]
        if abs(actual - expected) > CORRELATION_TOLERANCE:
            failures.append(
                f"[Table 5] PD {left}-{right}: "
                f"report {expected:+.2f}, computed {actual:+.2f}"
            )

    # 3. Table 4: the six against the four.
    for post_code, expected_row in TABLE_4.items():
        for pre_code, expected in expected_row.items():
            actual = float(np.corrcoef(post[post_code], common[pre_code])[0, 1])
            if abs(actual - expected) > CORRELATION_TOLERANCE:
                failures.append(
                    f"[Table 4] PD-{post_code} vs MS-{pre_code}: "
                    f"report {expected:+.2f}, computed {actual:+.2f}"
                )

    return failures


def main() -> None:
    post = build_array()

    print(f"Transcribed {len(post)} statements x {len(CODES)} discourses.\n")

    failures = validate(post)
    if failures:
        print("VALIDATION FAILED — nothing written.\n")
        for failure in failures:
            print(f"  {failure}")
        sys.exit(1)

    print("Validation passed: Tables 3, 4 and 5 all reproduced within "
          f"{CORRELATION_TOLERANCE:.2f}.\n")

    post.to_csv(OUT_ARRAY, index=False)
    print(f"Wrote {OUT_ARRAY.relative_to(ROOT)}")

    narratives = clean_narratives()
    OUT_NARRATIVES.write_text(narratives, encoding="utf-8")
    print(f"Wrote {OUT_NARRATIVES.relative_to(ROOT)}")

    # Console summary: the strongest agreement and disagreement per discourse is
    # the quickest human check that the columns mean what their labels say.
    print("\nStrongest agreement / disagreement per discourse:")
    for code in CODES:
        top = post.loc[post[code].idxmax()]
        bottom = post.loc[post[code].idxmin()]
        print(f"\n  {code}  (+{top[code]:.1f}) {top['item'][:88]}")
        print(f"     ({bottom[code]:+.1f}) {bottom['item'][:88]}")


if __name__ == "__main__":
    main()
