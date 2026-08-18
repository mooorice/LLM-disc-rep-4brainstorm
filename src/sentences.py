"""
Step 2b: split the analysed paragraphs into sentences.

Stance scoring works on sentences rather than whole paragraphs, for one reason
above all: the pattern we most need to detect lives *inside* a paragraph. A
briefing sentence that raises a concern and a following sentence that defuses it
cancel out when the paragraph is scored as a single premise, and the resulting
score is indistinguishable from a paragraph that never engaged with the concern
at all. Scored separately, the two sentences produce an entailment and a
contradiction, and their co-occurrence is exactly the measurement we want.

Two further reasons: NLI models are trained on single-sentence premises, so a
120-word paragraph is off-distribution; and one relevant sentence in a long
paragraph has its signal diluted by everything around it.

The cost is anaphora. "This would deepen inequality" means nothing once it is
separated from the sentence before it. Rather than hope the effect is small,
every sentence that opens with a bare demonstrative or pronoun is flagged here,
so the share of the corpus at risk is known and a context-window robustness pass
can target them.

Usage:
    python src/sentences.py
    python src/sentences.py --prompt brainstorm_generic
"""

import argparse
import re

import pandas as pd
import pysbd

import config

# Sentences opening with one of these carry a referent they no longer have once
# the sentence stands alone. Matched at the start only: "This would deepen
# inequality" is at risk, "Concerns about this are common" is not.
ANAPHORIC_OPENERS = re.compile(
    r"^(this|that|these|those|it|they|them|such|he|she|his|her|its|their|"
    r"both|either|neither|the former|the latter|doing so|here|there)\b",
    re.IGNORECASE,
)


def split_sentences(text: str, segmenter: pysbd.Segmenter) -> list[str]:
    """
    Split a paragraph into sentences.

    pysbd is used rather than a regex because this corpus is full of the things
    naive splitting breaks on: "e.g.", "Dr.", "2022.", "U.S." and abbreviated
    legislation names all appear in the essays.
    """
    return [s.strip() for s in segmenter.segment(text) if s.strip()]


def is_anaphoric(sentence: str) -> bool:
    """Whether this sentence opens with a reference to something outside itself."""
    return bool(ANAPHORIC_OPENERS.match(sentence.strip()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    prompt_name = config.apply_overrides(parser.parse_args())

    in_dir = config.PROCESSED_DIR / prompt_name
    paragraphs_path = in_dir / "paragraphs.csv"
    if not paragraphs_path.exists():
        raise FileNotFoundError(
            f"No paragraph table at {paragraphs_path}. Run src/segment.py first."
        )

    paragraphs = pd.read_csv(paragraphs_path)
    segmenter = pysbd.Segmenter(language="en", clean=False)

    rows = []
    for paragraph in paragraphs.itertuples():
        sentences = split_sentences(paragraph.text, segmenter)

        for position, sentence in enumerate(sentences):
            n_words = len(sentence.split())
            if n_words < config.MIN_SENTENCE_WORDS:
                # Fragments left by the splitter carry no scoreable claim. They
                # are dropped rather than fed to the NLI model, which would
                # return a confident judgement on almost nothing.
                continue

            rows.append({
                "run_id": paragraph.run_id,
                "model": paragraph.model,
                "repetition": paragraph.repetition,
                "paragraph_index": paragraph.paragraph_index,
                "sentence_index": position,
                "n_words": n_words,
                # The preceding sentence, kept so that the anaphora robustness
                # pass can build a two-sentence premise without re-splitting.
                "context": sentences[position - 1] if position > 0 else "",
                "anaphoric": is_anaphoric(sentence),
                "text": sentence,
            })

    sentence_table = pd.DataFrame(rows)
    out_path = in_dir / "sentences.csv"
    sentence_table.to_csv(out_path, index=False)

    print(f"Paragraphs in:   {len(paragraphs)}")
    print(f"Sentences out:   {len(sentence_table)}")
    print(f"Mean per paragraph: "
          f"{len(sentence_table) / max(len(paragraphs), 1):.1f}")
    print(f"Mean words/sentence: {sentence_table['n_words'].mean():.1f}")

    anaphoric = sentence_table["anaphoric"]
    print(f"\nOpening with an unresolved reference: {anaphoric.sum()} "
          f"({anaphoric.mean():.1%})")
    print("These are the sentences the context-window robustness pass targets.")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
