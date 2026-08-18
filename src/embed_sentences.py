"""
Step 3c: embed sentences and score cosine similarity at sentence level.

This is the *airtime* measure. Cosine similarity has been tested directly on
this corpus and shown to track what a text is about rather than what it claims
(`src/validate_stance.py`: passages rejecting a statement scored on average
closer to it than passages endorsing it). Rather than treat that as a defect, it
is used here for the one thing it does reliably — detecting whether a theme is
engaged at all — and the stance question is left to the NLI and judge stages.

Three similarity matrices are produced for each sentence:

  * against the four discourse descriptions, so the existing paragraph-level
    result can be reproduced at finer granularity;
  * against the 46 Q-statements in their original phrasing;
  * against the 46 Q-statements depersonalised.

Both statement forms are scored because neither is obviously correct: the
original is the instrument as administered, the depersonalised form is the one
the stance stage can read. Cosine should be relatively insensitive to the
difference, since it responds to topic rather than framing -- which makes this a
useful control on how much the rewriting perturbed things.

Usage:
    python src/embed_sentences.py
    python src/embed_sentences.py --prompt brainstorm_generic
"""

import argparse
import json

import numpy as np
import pandas as pd

import config
from embed import encode, load_discourses, load_model


def load_statement_forms() -> pd.DataFrame:
    """Load the 46 statements with both phrasings and their factor scores."""
    if not config.DEPERSONALISED_FILE.exists():
        raise FileNotFoundError(
            f"No depersonalised statements at {config.DEPERSONALISED_FILE}. "
            "Run scripts/build_depersonalised_statements.py first."
        )
    return pd.read_csv(config.DEPERSONALISED_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    prompt_name = config.apply_overrides(parser.parse_args())

    out_dir = config.PROCESSED_DIR / prompt_name
    sentences_path = out_dir / "sentences.csv"
    if not sentences_path.exists():
        raise FileNotFoundError(
            f"No sentence table at {sentences_path}. Run src/sentences.py first."
        )

    sentences = pd.read_csv(sentences_path)
    discourses = load_discourses()
    statements = load_statement_forms()

    print(f"Sentences: {len(sentences)}   Discourses: {len(discourses)}   "
          f"Statements: {len(statements)}")

    model = load_model()

    print("\nEncoding sentences ...")
    sentence_embeddings = encode(
        model, sentences["text"].tolist(),
        as_query=config.USE_ASYMMETRIC_PROMPTS,
    )

    print("\nEncoding discourse descriptions ...")
    discourse_embeddings = encode(model, discourses["text"].tolist(), as_query=False)

    print("\nEncoding statements (original phrasing) ...")
    original_embeddings = encode(
        model, statements["item_original"].tolist(), as_query=False
    )

    print("\nEncoding statements (depersonalised) ...")
    depersonalised_embeddings = encode(
        model, statements["item_depersonalised"].tolist(), as_query=False
    )

    # All embeddings are unit length, so a dot product is the cosine.
    #
    # The discourse matrix is suffixed by baseline because its columns are the
    # four pre-deliberative or the six post-deliberative discourses. The two
    # statement matrices are not: the surveyed statements are the same
    # instrument in both maps, so those matrices serve either baseline.
    similarities = {
        f"sentence_x_discourse_{config.BASELINE}":
            sentence_embeddings @ discourse_embeddings.T,
        "sentence_x_statement_original": sentence_embeddings @ original_embeddings.T,
        "sentence_x_statement_depersonalised":
            sentence_embeddings @ depersonalised_embeddings.T,
    }

    np.save(out_dir / "sentence_embeddings.npy", sentence_embeddings)
    for name, matrix in similarities.items():
        np.save(out_dir / f"{name}.npy", matrix)

    settings = {
        "embedding_model": config.EMBEDDING_MODEL,
        "compression_ratio": config.COMPRESSION_RATIO,
        "use_asymmetric_prompts": config.USE_ASYMMETRIC_PROMPTS,
        "baseline": config.BASELINE,
        "n_sentences": len(sentences),
        "n_discourses": len(discourses),
        "n_statements": len(statements),
        "shapes": {name: list(m.shape) for name, m in similarities.items()},
    }
    (out_dir / "sentence_embedding_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    # --- console diagnostics ---------------------------------------------
    discourse_sim = similarities[f"sentence_x_discourse_{config.BASELINE}"]
    codes = discourses["code"].tolist()

    print(f"\nSaved to {out_dir}/")
    print("\nMean sentence-to-discourse cosine, and how often each discourse wins:")
    winners = pd.Series(np.array(codes)[discourse_sim.argmax(axis=1)])
    summary = pd.DataFrame({
        "mean_cosine": discourse_sim.mean(axis=0).round(4),
        "argmax_share": [winners.eq(c).mean() for c in codes],
    }, index=codes)
    summary["argmax_share"] = summary["argmax_share"].map(lambda v: f"{v:.1%}")
    print(summary.to_string())

    # The margin is the thing to watch: at paragraph level the winner led the
    # runner-up by only a few thousandths, which is what made the whole
    # assignment unstable.
    top2 = np.sort(discourse_sim, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    print(f"\nWinner's lead over runner-up: mean {margin.mean():.4f}, "
          f"median {np.median(margin):.4f}")

    # How much did depersonalising perturb the topical signal? If cosine really
    # is stance-blind, the two statement forms should agree closely.
    a = similarities["sentence_x_statement_original"]
    b = similarities["sentence_x_statement_depersonalised"]
    per_statement = [np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(a.shape[1])]
    print(f"\nCorrelation between the two statement forms, across sentences:")
    print(f"  mean r over the 46 statements: {np.mean(per_statement):.3f}")
    print(f"  lowest: {np.min(per_statement):.3f} "
          f"(statement {int(np.argmin(per_statement)) + 1})")
    print("  A high correlation means the rewriting left the topical signal "
          "intact, as expected if cosine responds to topic rather than framing.")


if __name__ == "__main__":
    main()
