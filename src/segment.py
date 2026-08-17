"""
Step 2: split the generated essays into paragraphs.

The unit of analysis is the paragraph: each one is later embedded and matched
against the four human discourse descriptions, and the proportions of paragraphs
falling to each discourse are what we compare against the Australian population
weights. This script turns the raw essay JSON files into one tidy table of
paragraphs.

Markdown formatting is stripped, because the models sometimes add emphasis or a
stray heading despite being asked for continuous prose, and those characters
would otherwise end up inside the embedded text. Segments shorter than
config.MIN_PARAGRAPH_WORDS are dropped, but they are written to a separate file
rather than silently discarded, so the loss can be inspected.

Usage:
    python src/segment.py
    python src/segment.py --prompt brainstorm_generic
"""

import argparse
import json
import re

import pandas as pd

import config


def strip_markdown(text: str) -> str:
    """
    Remove the markdown decorations a model may have added, leaving plain prose.

    This deliberately only removes formatting characters; no words are changed,
    so the embedded text is exactly what the model wrote.
    """
    # Leading heading markers: "## Something" -> "Something"
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text)
    # Leading list markers: "- ", "* ", "1. ", "1) ", bullet characters
    text = re.sub(r"^\s{0,3}(?:[-*+•–]|\d+[.)])\s+", "", text)
    # Bold / italic / inline code markers, kept simple on purpose
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Markdown links: "[label](url)" -> "label"
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Horizontal rules become empty
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text)
    # Collapse any run of whitespace (including the essay's own line wrapping)
    return re.sub(r"\s+", " ", text).strip()


def split_paragraphs(essay: str) -> list[str]:
    """Split an essay on blank lines and clean each resulting segment."""
    raw_segments = re.split(r"\n\s*\n", essay)
    cleaned = [strip_markdown(segment) for segment in raw_segments]
    return [segment for segment in cleaned if segment]


def load_essays(prompt_name: str) -> list[dict]:
    """Read every generated essay for one prompt, sorted for stable ordering."""
    directory = config.GENERATED_DIR / prompt_name
    if not directory.exists():
        raise FileNotFoundError(
            f"No generated essays at {directory}. Run src/generate.py first."
        )

    records = []
    for path in sorted(directory.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise FileNotFoundError(f"{directory} contains no essay JSON files.")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    prompt_name = config.apply_overrides(parser.parse_args())

    essays = load_essays(prompt_name)

    kept_rows = []
    dropped_rows = []

    for record in essays:
        model = record["model"]
        repetition = record["repetition"]
        run_id = f"{model.replace('/', '__')}__rep{repetition:02d}"

        # paragraph_index counts only the paragraphs we keep, so indices in the
        # output table are contiguous and reflect position in the analysed text.
        paragraph_index = 0
        for segment in split_paragraphs(record["essay"]):
            n_words = len(segment.split())
            if n_words < config.MIN_PARAGRAPH_WORDS:
                dropped_rows.append({
                    "run_id": run_id, "model": model, "repetition": repetition,
                    "n_words": n_words, "text": segment,
                })
                continue

            kept_rows.append({
                "run_id": run_id,
                "model": model,
                "repetition": repetition,
                "paragraph_index": paragraph_index,
                "n_words": n_words,
                "text": segment,
            })
            paragraph_index += 1

    paragraphs = pd.DataFrame(kept_rows)
    dropped = pd.DataFrame(dropped_rows)

    out_dir = config.PROCESSED_DIR / prompt_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paragraphs.to_csv(out_dir / "paragraphs.csv", index=False)
    dropped.to_csv(out_dir / "dropped_segments.csv", index=False)

    # --- summary for the console ------------------------------------------
    print(f"Essays read:        {len(essays)}")
    print(f"Paragraphs kept:    {len(paragraphs)}")
    print(f"Segments dropped:   {len(dropped)} "
          f"(under {config.MIN_PARAGRAPH_WORDS} words)")
    print(f"\nWritten to {out_dir}/paragraphs.csv\n")

    if not paragraphs.empty:
        per_model = paragraphs.groupby("model").agg(
            essays=("run_id", "nunique"),
            paragraphs=("text", "size"),
            mean_words=("n_words", "mean"),
        )
        per_model["paragraphs_per_essay"] = (
            per_model["paragraphs"] / per_model["essays"]
        )
        print("Per model:")
        print(per_model.round(1).to_string())


if __name__ == "__main__":
    main()
