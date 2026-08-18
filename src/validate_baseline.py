"""
Does the embedding baseline measure discourse, or does it measure topic?

The four-discourse run produced a result that looked substantive and was not:
discourse C dominated because its description was the only one containing the
word "mitochondrial", and every paragraph about mitochondrial donation went to it
regardless of what the paragraph argued. The six-discourse baseline reproduces
the same shape -- C takes roughly two fifths of all sentences -- so it has to
face the same test before anything is claimed.

Three checks:

  1. STRUCTURE. The report publishes correlations between its discourses,
     computed from the factor arrays: how similarly two discourses respond to the
     45 surveyed statements. If the embedding is capturing ways of reasoning, the
     cosine between two discourse descriptions should track those correlations.
     If it is capturing topic, it will not.

  2. LENGTH AND LEXICON. A description that is longer, or that uses more of the
     corpus's own vocabulary, is generically closer to everything. If argmax
     share is predicted by description length, the winner is an artefact of how
     much the researchers wrote about each position.

  3. DISTINCTIVE TERMS. For each discourse, the words that appear in its
     description and in no other. These are the keyword hooks: if the sentences
     assigned to a discourse are the ones containing its unique vocabulary, the
     measure is a term-matcher wearing a semantic coat.

Usage:
    python src/validate_baseline.py
    python src/validate_baseline.py --baseline pre
"""

import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd

import config
from embed import load_discourses

# Published inter-discourse correlations, computed by the report from the factor
# arrays. Keyed by baseline.
PUBLISHED_CORRELATIONS = {
    # OP12 Table 5, p. 140 -- the six post-deliberative discourses.
    "post": {
        ("A", "B"): 0.56, ("A", "C"): -0.10, ("A", "D"): 0.44,
        ("A", "E"): 0.35, ("A", "F"): 0.48,
        ("B", "C"): 0.35, ("B", "D"): 0.53, ("B", "E"): 0.53, ("B", "F"): 0.45,
        ("C", "D"): 0.30, ("C", "E"): 0.27, ("C", "F"): -0.04,
        ("D", "E"): 0.47, ("D", "F"): 0.45,
        ("E", "F"): 0.21,
    },
    # OP12 Table 3, p. 134 -- the four mapping-study discourses.
    "pre": {
        ("A", "B"): 0.32, ("A", "C"): 0.22, ("A", "D"): 0.36,
        ("B", "C"): 0.48, ("B", "D"): 0.28, ("C", "D"): 0.34,
    },
}

# Words too common to be diagnostic of anything.
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself me more most my myself no nor
not of off on once only or other ought our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why with would you your yours yourself yourselves also
this position discourse people associated tend tends likely other others
compared particularly strongly relatively nevertheless however although
""".split())


def tokenise(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) > 3 and w not in STOPWORDS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    prompt_name = config.apply_overrides(parser.parse_args())

    discourses = load_discourses()
    codes = discourses["code"].tolist()
    names = dict(zip(discourses["code"], discourses["name"]))

    embeddings = np.load(config.discourse_embeddings_path())
    similarity = pd.DataFrame(embeddings @ embeddings.T, index=codes, columns=codes)

    processed = config.PROCESSED_DIR / prompt_name
    sentences = pd.read_csv(processed / "sentences.csv")
    airtime = pd.DataFrame(
        np.load(processed / f"sentence_x_discourse_{config.BASELINE}.npy"),
        columns=codes,
    )
    active = [c for c in codes if c in config.ACTIVE_DISCOURSES]
    winner = pd.Series(np.array(active)[airtime[active].to_numpy().argmax(axis=1)])

    print(f"Baseline: {config.baseline()['label']}")
    print(f"Prompt:   {prompt_name}   ({len(sentences)} sentences)\n")

    # --- 1. structure ------------------------------------------------------
    print("=" * 72)
    print("1. Does embedding similarity track the report's discourse correlations?")
    print("=" * 72)

    published = PUBLISHED_CORRELATIONS[config.BASELINE]
    rows = []
    for (left, right), reported in sorted(published.items()):
        rows.append({
            "pair": f"{left}-{right}",
            "report_r": reported,
            "embedding_cos": similarity.loc[left, right],
        })
    pairs = pd.DataFrame(rows).sort_values("report_r")
    print(pairs.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    agreement = float(np.corrcoef(pairs["report_r"], pairs["embedding_cos"])[0, 1])
    print(f"\n  Correlation between the two columns: r = {agreement:+.3f}")
    if agreement < 0.3:
        print("  -> The embedding does NOT recover the structure of the human")
        print("     discourse space. Cosine to a discourse description is a")
        print("     measure of shared topic, not of shared reasoning.")
    else:
        print("  -> The embedding partially recovers the human structure.")

    # The sharpest version of the same point: the pair the report calls most
    # different, versus what the embedding calls most different.
    most_distinct_human = pairs.iloc[0]["pair"]
    most_distinct_embed = pairs.sort_values("embedding_cos").iloc[0]["pair"]
    print(f"\n  Most distinct pair per the report:     {most_distinct_human}")
    print(f"  Most distinct pair per the embedding:  {most_distinct_embed}")

    # --- 2. length and attractiveness -------------------------------------
    print("\n" + "=" * 72)
    print("2. Is the winner predicted by how much was written about it?")
    print("=" * 72)

    profile = pd.DataFrame({
        "name": [names[c] for c in codes],
        "words": [len(d.split()) for d in discourses["text"]],
        "mean_cosine": airtime.mean().reindex(codes).to_numpy(),
        "argmax_share": [float((winner == c).mean()) for c in codes],
    }, index=codes)
    print(profile.to_string(float_format=lambda v: f"{v:.4f}"))

    for column in ["mean_cosine", "argmax_share"]:
        r = float(np.corrcoef(profile["words"], profile[column])[0, 1])
        print(f"\n  words vs {column}: r = {r:+.3f}")
    print("\n  A strong positive correlation here means longer descriptions win,")
    print("  which is a property of the baseline file rather than of the essays.")

    # --- 3. distinctive vocabulary ----------------------------------------
    print("\n" + "=" * 72)
    print("3. Distinctive vocabulary, and whether it drives the assignment")
    print("=" * 72)

    vocabularies = {c: set(tokenise(t)) for c, t in zip(codes, discourses["text"])}
    corpus_counts = Counter()
    for text in sentences["text"]:
        corpus_counts.update(set(tokenise(text)))

    for code in codes:
        others = set().union(*(v for c, v in vocabularies.items() if c != code))
        unique = vocabularies[code] - others
        # Only terms that actually occur in the essays can influence anything.
        occurring = sorted(
            (w for w in unique if corpus_counts[w] > 0),
            key=lambda w: -corpus_counts[w],
        )

        assigned = winner == code
        print(f"\n  {code} ({names[code]}) — {int(assigned.sum())} sentences")
        if not occurring:
            print("     no unique terms occurring in the corpus")
            continue

        # For the top unique terms: how much more likely is a sentence
        # containing the term to be assigned to this discourse?
        lines = []
        for word in occurring[:6]:
            contains = sentences["text"].str.contains(
                rf"\b{word}\b", case=False, regex=True
            ).to_numpy()
            if contains.sum() < 3:
                continue
            rate_with = float(assigned[contains].mean())
            rate_without = float(assigned[~contains].mean())
            lift = rate_with / rate_without if rate_without > 0 else float("inf")
            lines.append(
                f"     {word:<18} in {int(contains.sum()):>4} sentences  "
                f"assigned {rate_with:>6.1%} vs {rate_without:>6.1%}  "
                f"lift x{lift:.1f}"
            )
        print("\n".join(lines) if lines else "     unique terms too rare to test")

    print("\n  A large lift means sentences are being routed by keyword. The")
    print("  question to ask of any such term is whether it names the discourse's")
    print("  reasoning or merely its subject matter.")

    # --- 4. what the top-scoring sentences actually say -------------------
    print("\n" + "=" * 72)
    print("4. The most confidently assigned sentence per discourse")
    print("=" * 72)
    print("  Read these against the discourse names above. If a sentence is")
    print("  neutral exposition rather than an expression of that position, the")
    print("  argmax is tracking subject matter.\n")

    ordered = np.sort(airtime[active].to_numpy(), axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    for code in active:
        mask = (winner == code).to_numpy()
        if not mask.any():
            continue
        best = int(np.argmax(np.where(mask, margin, -np.inf)))
        print(f"  {code} (margin {margin[best]:.4f}):")
        print(f"     {sentences['text'].iloc[best][:300]}\n")


if __name__ == "__main__":
    main()
