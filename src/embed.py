"""
Step 3: embed the discourse baselines and the generated paragraphs.

Both sides of the comparison are encoded with the same model and the same
settings, and embeddings are L2-normalised so that a dot product between any two
of them is their cosine similarity.

Two choices worth knowing about:

  * The four baseline texts are the *cleaned* discourse descriptions from the
    Australian mapping study. Only the description body is embedded; the
    "Discourse A: Scientific Progress" header line is dropped, because the label
    is a name the researchers gave the factor, not something the participants
    said.

  * Jasper offers an asymmetric query/document prompt pair meant for retrieval.
    Our comparison is symmetric, so by default neither side gets the query
    prompt. See config.USE_ASYMMETRIC_PROMPTS to switch.

Usage:
    python src/embed.py
    python src/embed.py --prompt brainstorm_generic
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import config


def load_discourses(path=None) -> pd.DataFrame:
    """
    Parse the cleaned discourse file for the active baseline into one row per
    discourse.

    The file holds one block per discourse separated by lines containing '###',
    each opening with a "Discourse X: Name" header followed by the description
    body. Both baseline files use this format: four blocks for the
    pre-deliberative mapping study, six for the post-deliberative jury map.
    """
    path = path or config.DISCOURSE_FILE
    blocks = [b.strip() for b in path.read_text(encoding="utf-8").split("###")]
    blocks = [b for b in blocks if b]

    rows = []
    for block in blocks:
        header, _, body = block.partition("\n")
        header = header.strip()

        # "Discourse A: Scientific Progress" -> code "A", name "Scientific Progress"
        label = header.removeprefix("Discourse").strip()
        code, _, name = label.partition(":")

        rows.append({
            "code": code.strip(),
            "name": name.strip(),
            # Blank lines inside the body separate its paragraphs; the whole body
            # is embedded as a single text, since a discourse is characterised by
            # all of it together.
            "text": " ".join(body.split()),
        })

    discourses = pd.DataFrame(rows).sort_values("code").reset_index(drop=True)

    # The baseline declares which codes it should contain. Checking against that
    # rather than a bare count catches a half-parsed file, a stray separator, or
    # the wrong baseline file being pointed at.
    expected = config.baseline()["codes"]
    if list(discourses["code"]) != expected:
        raise ValueError(
            f"Baseline {config.BASELINE!r} expects discourses {expected} in "
            f"{path}, parsed {list(discourses['code'])}"
        )
    return discourses


def load_model() -> SentenceTransformer:
    """Load the embedding model onto the GPU if one is available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {config.EMBEDDING_MODEL} on {device} ...")
    return SentenceTransformer(
        config.EMBEDDING_MODEL,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
        },
        trust_remote_code=True,
        tokenizer_kwargs={"padding_side": "left"},
        device=device,
    )


def encode(model: SentenceTransformer, texts: list[str], as_query: bool) -> np.ndarray:
    """
    Encode texts to unit-length vectors.

    `as_query` applies Jasper's retrieval query prompt; it is only ever True when
    config.USE_ASYMMETRIC_PROMPTS has been switched on.
    """
    kwargs = {
        "normalize_embeddings": True,
        "compression_ratio": config.COMPRESSION_RATIO,
        "batch_size": config.EMBEDDING_BATCH_SIZE,
        "show_progress_bar": True,
    }
    if as_query:
        kwargs["prompt_name"] = "query"

    embeddings = np.asarray(model.encode(texts, **kwargs), dtype=np.float32)

    # The model normalises internally in bfloat16, which leaves norms off by up
    # to ~0.5%. Re-normalising in float32 makes the dot product an exact cosine,
    # which matters here because the differences we care about are small.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    prompt_name = config.apply_overrides(parser.parse_args())

    out_dir = config.PROCESSED_DIR / prompt_name

    paragraphs_path = out_dir / "paragraphs.csv"
    if not paragraphs_path.exists():
        raise FileNotFoundError(
            f"No paragraph table at {paragraphs_path}. Run src/segment.py first."
        )
    paragraphs = pd.read_csv(paragraphs_path)
    discourses = load_discourses()

    print(f"Discourses: {len(discourses)}   Paragraphs: {len(paragraphs)}")
    model = load_model()

    # Paragraphs take the query role only in the asymmetric setting; the four
    # discourse descriptions are always encoded as plain documents.
    print("\nEncoding discourse baselines ...")
    discourse_embeddings = encode(model, discourses["text"].tolist(), as_query=False)

    print("\nEncoding paragraphs ...")
    paragraph_embeddings = encode(
        model, paragraphs["text"].tolist(), as_query=config.USE_ASYMMETRIC_PROMPTS
    )

    # The discourse embeddings depend only on the baseline file and the encoder
    # settings, so they live outside the per-prompt directory -- but they are
    # suffixed by baseline, so the four and the six coexist.
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.discourse_embeddings_path(), discourse_embeddings)
    discourses.to_csv(config.discourse_table_path(), index=False)
    np.save(out_dir / "paragraph_embeddings.npy", paragraph_embeddings)

    # Record the encoder settings next to the vectors, so a stale embedding file
    # can never be silently reused under different settings.
    settings = {
        "embedding_model": config.EMBEDDING_MODEL,
        "compression_ratio": config.COMPRESSION_RATIO,
        "use_asymmetric_prompts": config.USE_ASYMMETRIC_PROMPTS,
        "baseline": config.BASELINE,
        "n_discourses": len(discourses),
        "n_paragraphs": len(paragraphs),
        "embedding_dim": int(paragraph_embeddings.shape[1]),
    }
    (out_dir / "embedding_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    print(f"\nEmbedding dimension: {paragraph_embeddings.shape[1]}")
    print(f"Saved to {out_dir}/paragraph_embeddings.npy")

    # A quick sanity check: how similar are the baselines to each other? The
    # report publishes the human correlations between discourses (Table 3 for
    # the four, Table 5 for the six), so this ordering is worth eyeballing
    # against them -- where embedding similarity and human correlation disagree,
    # the embedding is responding to shared topic rather than shared reasoning.
    similarity = discourse_embeddings @ discourse_embeddings.T
    print("\nBaseline-to-baseline cosine similarity:")
    print(pd.DataFrame(
        similarity, index=discourses["code"], columns=discourses["code"]
    ).round(3).to_string())


if __name__ == "__main__":
    main()
