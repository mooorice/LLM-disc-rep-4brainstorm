"""
Step 3b: score each sentence's stance towards each Q-statement, using NLI.

Cosine similarity measures what a text is *about*. It cannot tell endorsement
from rejection -- tested directly on this corpus, passages rejecting a statement
scored on average *closer* to it than passages endorsing it, because rejecting a
claim means restating it and then arguing. That makes similarity unusable for
the question the factor arrays actually pose, which is agreement.

Natural language inference asks the right question. With the sentence as premise
and a Q-statement as hypothesis, an NLI model returns probabilities for
entailment, neutrality and contradiction. The signed score

    stance = P(entail) - P(contradict)

runs from -1 (the sentence denies the statement) through 0 (silent or
irrelevant) to +1 (the sentence asserts it), which is directly commensurable
with the signed z-scores of the human factor array.

Output is a (sentences x statements) matrix for each of the two components,
kept separate rather than pre-combined: a paragraph that both entails and
contradicts the same statement is raising a concern and then defusing it, and
that pattern is a finding in its own right.

Usage:
    python src/stance.py
    python src/stance.py --prompt brainstorm_generic
    python src/stance.py --context   # anaphora robustness pass
"""

import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


def load_statements(form: str) -> pd.DataFrame:
    """
    Load the 46 Q-statements in one of the two phrasings, with their z-scores.

    "original" is the instrument as administered. Two thirds of those items are
    attitude reports ("I'm concerned that X"), which NLI cannot usefully score
    against reported speech -- "many Australians worry that X" does not entail
    "*I* am concerned that X", so the model correctly returns neutral and the
    measurement says nothing. "depersonalised" strips the wrapper and leaves the
    proposition. Both are scored; see the CHANGES file beside the data.
    """
    if form not in config.STATEMENT_FORMS:
        raise ValueError(f"Unknown form {form!r}; expected one of "
                         f"{config.STATEMENT_FORMS}")

    statements = pd.read_csv(config.DEPERSONALISED_FILE)
    statements["text"] = statements[f"item_{form}"]
    return statements[["statement_id", "text", "transformation", "A", "B", "C", "D"]]


def load_nli():
    """Load the NLI model and work out which output index means what."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {config.NLI_MODEL} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.NLI_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    # Never hard-code the label order. Different NLI checkpoints use different
    # index orders, and silently swapping entailment for contradiction would
    # invert every result while looking entirely plausible.
    labels = {name.lower(): index for index, name in model.config.id2label.items()}
    try:
        indices = (labels["entailment"], labels["contradiction"])
    except KeyError as error:
        raise RuntimeError(
            f"Cannot find entailment/contradiction in {model.config.id2label}"
        ) from error

    print(f"Label map: {model.config.id2label}")
    return model, tokenizer, device, indices


@torch.inference_mode()
def score_pairs(model, tokenizer, device, indices, premises: list[str],
                hypotheses: list[str], batch_size: int) -> np.ndarray:
    """
    Score a flat list of (premise, hypothesis) pairs.

    Returns an (n, 2) array of [P(entail), P(contradict)].
    """
    entail_index, contradict_index = indices
    out = np.zeros((len(premises), 2), dtype=np.float32)

    for start in range(0, len(premises), batch_size):
        stop = min(start + batch_size, len(premises))
        batch = tokenizer(
            premises[start:stop], hypotheses[start:stop],
            return_tensors="pt", padding=True, truncation=True, max_length=256,
        ).to(device)

        probabilities = model(**batch).logits.float().softmax(dim=-1)
        out[start:stop, 0] = probabilities[:, entail_index].cpu().numpy()
        out[start:stop, 1] = probabilities[:, contradict_index].cpu().numpy()

        if start % (batch_size * 50) == 0:
            print(f"  {stop}/{len(premises)} pairs", flush=True)

    return out


def score_matrix(model, tokenizer, device, indices, texts: list[str],
                 statements: list[str], batch_size: int) -> np.ndarray:
    """
    Score every text against every statement.

    Returns an (n_texts, n_statements, 2) array, last axis [entail, contradict].
    """
    premises, hypotheses = [], []
    for text in texts:
        for statement in statements:
            premises.append(text)
            hypotheses.append(statement)

    flat = score_pairs(model, tokenizer, device, indices,
                       premises, hypotheses, batch_size)
    return flat.reshape(len(texts), len(statements), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    parser.add_argument(
        "--context", action="store_true",
        help="prepend the preceding sentence to anaphoric sentences "
             "(robustness pass for unresolved references)",
    )
    parser.add_argument(
        "--form", choices=config.STATEMENT_FORMS, default="depersonalised",
        help="which phrasing of the Q-statements to score against "
             "(default: depersonalised)",
    )
    args = parser.parse_args()
    prompt_name = config.apply_overrides(args)

    out_dir = config.PROCESSED_DIR / prompt_name
    sentences_path = out_dir / "sentences.csv"
    if not sentences_path.exists():
        raise FileNotFoundError(
            f"No sentence table at {sentences_path}. Run src/sentences.py first."
        )

    sentences = pd.read_csv(sentences_path).fillna({"context": ""})
    statements = load_statements(args.form)

    premises = sentences["text"].tolist()
    if args.context:
        # Only the flagged sentences get their predecessor prepended. Giving
        # every sentence a context window would reintroduce the dilution that
        # sentence-level scoring exists to avoid.
        premises = [
            f"{context} {text}".strip() if anaphoric and context else text
            for text, context, anaphoric in zip(
                sentences["text"], sentences["context"], sentences["anaphoric"]
            )
        ]
        n_extended = int(sentences["anaphoric"].sum())
        print(f"Context pass: {n_extended} anaphoric sentences extended.")

    print(f"Sentences: {len(premises)}   Statements: {len(statements)}   "
          f"Pairs: {len(premises) * len(statements):,}")

    model, tokenizer, device, indices = load_nli()
    scores = score_matrix(
        model, tokenizer, device, indices,
        premises, statements["text"].tolist(), config.NLI_BATCH_SIZE,
    )

    suffix = f"_{args.form}" + ("_context" if args.context else "")
    np.save(out_dir / f"sentence_stance{suffix}.npy", scores)
    statements.to_csv(config.PROCESSED_DIR / f"statements_{args.form}.csv",
                      index=False)

    stance = scores[:, :, 0] - scores[:, :, 1]
    print(f"\nSaved {scores.shape} to {out_dir}/sentence_stance{suffix}.npy")
    print(f"\nStance score distribution over all {stance.size:,} pairs:")
    print(f"  asserting  (> +0.5): {(stance > 0.5).mean():.2%}")
    print(f"  denying    (< -0.5): {(stance < -0.5).mean():.2%}")
    print(f"  silent  (|s| < 0.1): {(np.abs(stance) < 0.1).mean():.2%}")
    print(f"  mean |stance|      : {np.abs(stance).mean():.3f}")


if __name__ == "__main__":
    main()
