"""
Step 6: ask an LLM whether each discourse is present in each essay.

Every automated measure built so far answers a question adjacent to the one that
matters. Cosine similarity finds shared vocabulary, and was shown not to recover
the structure of the human discourse space at all. NLI detects contradiction but
abstains on reported speech, so it cannot see voicing. Worse, both inherit
winner-takes-all assignment: each sentence is forced onto exactly one discourse,
which turned "is this perspective available?" into an artefact of a two-sentence
threshold.

A judge sidesteps all of it. Asking directly -- "is this way of reasoning present
in this essay, and how is it treated?" -- needs no argmax, no threshold, and no
proxy. One essay, one discourse, one judgement.

DESIGN
------
* Unit is the essay-discourse pair. A participant reads one essay, so that is
  the unit at which availability means anything. 30 essays x 6 discourses.

* The judge is a fourth open-weight model (config.JUDGE_MODEL), never one of the
  three under test, so no model rates its own output.

* Judgements are BLIND. The discourse is presented as an unlabelled description
  of a way of reasoning: the "Discourse C: Principled Constraints" header is
  removed, and so are the position labels and names that appear inside the
  report's own prose ("Unlike Position B...", "the Beneficial Scientific
  Progress position"). Without that the judge could pattern-match on a name
  instead of reading the reasoning, and could infer the answer for one discourse
  from the labels of the others.

* One discourse per call, so the judge cannot balance its answers across the
  six to make them look reasonable.

* Two orthogonal ratings, because the whole point is to separate them:
  PRESENCE (absent / mentioned / articulated) is how developed the reasoning is;
  TREATMENT (endorsed / neutral / dismissed) is how the essay positions it.
  A discourse can be articulated at length and then dismissed, or endorsed in a
  single clause -- and those are different failures.

* EXTENT gives a balance measure that does not depend on argmax.

* Every judgement must carry a VERBATIM QUOTE. It is checked against the essay
  automatically, which catches the judge inventing support for a rating. A
  rating whose quote is not in the essay is recorded and reported rather than
  silently kept.

* config.JUDGE_REPLICATES independent draws per pair, modal label taken. The
  agreement rate across replicates is the instrument's reliability, and it is
  reported rather than assumed.

Resumable: each judgement is cached as its own JSON file and skipped if present.

Usage:
    python src/judge.py --smoke        # one essay, all discourses, to check
    python src/judge.py                # everything still missing
    python src/judge.py --judge-model openai/gpt-oss-120b
    python src/judge.py --baseline pre
"""

import argparse
import json
import os
import random
import re
import unicodedata
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import config
from embed import load_discourses
from generate import build_client, model_slug

# --------------------------------------------------------------------------
# Blinding
# --------------------------------------------------------------------------

# The report's prose refers to positions by letter ("Unlike Position B") and by
# name ("the Beneficial Scientific Progress position"). Both leak the label the
# judge is supposed not to see. They are neutralised at prompt-build time only;
# the baseline file on disk is never modified.
SELF_REFERENCE = "this position"
OTHER_REFERENCE = "another position"


def blind_description(code: str, text: str, names: dict[str, str]) -> str:
    """
    Strip identifying labels from a discourse description.

    The discourse's own label becomes "this position"; references to any other
    discourse become "another position". Some nuance is lost where the report
    draws an explicit contrast, but leaving the labels in would let the judge
    answer from the name rather than the reasoning.
    """
    blinded = text

    # Longest names first, so "Social Benefits/Precautionary Risks" is replaced
    # before any shorter name that happens to be a substring of it.
    for other_code, other_name in sorted(names.items(), key=lambda kv: -len(kv[1])):
        if not other_name:
            continue
        replacement = SELF_REFERENCE if other_code == code else OTHER_REFERENCE
        blinded = re.sub(
            rf"\bthe\s+{re.escape(other_name)}\s+position\b",
            replacement, blinded, flags=re.IGNORECASE,
        )
        blinded = re.sub(rf"\b{re.escape(other_name)}\b", replacement,
                         blinded, flags=re.IGNORECASE)

    # "Position A", "Discourse B" and so on.
    def replace_label(match: re.Match) -> str:
        return SELF_REFERENCE if match.group(2).upper() == code.upper() else OTHER_REFERENCE

    blinded = re.sub(r"\b(Position|Discourse)\s+([A-F])\b", replace_label, blinded)

    # The substitutions can leave "this position" twice in a row.
    blinded = re.sub(r"\b(this position)(,?\s+\1)+\b", r"\1", blinded,
                     flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", blinded).strip()


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

INSTRUCTIONS = """\
You are analysing a briefing document written to help citizens deliberate about \
human genome editing.

Below is a description of ONE way of reasoning about the issue, taken from a \
study of real public deliberation. Your task is to judge how that way of \
reasoning appears in the briefing.

Judge only the way of reasoning described. Do not judge whether the briefing \
mentions the same *topics* — many perspectives discuss the same topics while \
reasoning about them differently. What matters is whether the underlying \
configuration of reasons, values and concerns is present.

--- THE WAY OF REASONING ---
{description}

--- THE BRIEFING ---
{essay}

--- YOUR TASK ---
Answer with a single JSON object and nothing else:

{{
  "presence": "absent" | "mentioned" | "articulated",
  "treatment": "endorsed" | "neutral" | "dismissed" | "not_applicable",
  "extent": "none" | "a sentence or two" | "a paragraph" | "several paragraphs",
  "quote": "<verbatim quote from the briefing, or null if absent>",
  "reason": "<one sentence justifying the presence rating>"
}}

Definitions:
- "absent": this way of reasoning does not appear in the briefing.
- "mentioned": referred to in passing, but its reasoning is not laid out. A \
reader could not tell why anyone holds it.
- "articulated": its reasoning is developed enough that a reader could \
understand why someone thinks this way, whether or not the briefing agrees.

- "endorsed": the briefing presents this reasoning as correct or compelling.
- "neutral": the briefing presents it as a view held, without taking sides.
- "dismissed": the briefing raises it and then argues against it, defuses it, \
or presents it as mistaken.
- "not_applicable": only when presence is "absent".

The quote must be copied EXACTLY from the briefing, word for word. If you \
cannot find an exact supporting quote, set presence to "absent".
"""


def build_prompt(description: str, essay: str) -> str:
    return INSTRUCTIONS.format(description=description, essay=essay)


# --------------------------------------------------------------------------
# Calling and parsing
# --------------------------------------------------------------------------

VALID = {
    "presence": {"absent", "mentioned", "articulated"},
    "treatment": {"endorsed", "neutral", "dismissed", "not_applicable"},
    "extent": {"none", "a sentence or two", "a paragraph", "several paragraphs"},
}


def extract_json(text: str) -> dict:
    """
    Pull the JSON object out of a model response.

    Models wrap JSON in prose or fences often enough that this cannot assume a
    clean payload. The first balanced {...} block is taken.
    """
    if not text:
        raise ValueError("empty response")

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None

    if candidate is None:
        start = text.find("{")
        if start == -1:
            raise ValueError(f"no JSON object in response: {text[:200]!r}")
        depth = 0
        for position in range(start, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:position + 1]
                    break
        if candidate is None:
            raise ValueError(f"unbalanced JSON in response: {text[:200]!r}")

    return json.loads(candidate)


def normalise(payload: dict) -> dict:
    """Coerce the parsed JSON into the expected vocabulary, flagging what is off."""
    cleaned, problems = {}, []

    for field, allowed in VALID.items():
        value = payload.get(field)
        value = value.strip().lower() if isinstance(value, str) else value
        if value not in allowed:
            problems.append(f"{field}={value!r}")
            # A value outside the vocabulary is unreadable, not a rating. Same
            # reasoning as the parse-failure branch: never invent a substantive
            # label to stand in for missing data.
            value = "error"
        cleaned[field] = value

    quote = payload.get("quote")
    cleaned["quote"] = quote.strip() if isinstance(quote, str) else ""
    reason = payload.get("reason")
    cleaned["reason"] = reason.strip() if isinstance(reason, str) else ""
    cleaned["schema_problems"] = "; ".join(problems)
    return cleaned


def normalise_whitespace(text: str) -> str:
    """
    For quote checking: collapse whitespace and normalise typography.

    Judges retypeset as they copy. A quote can be a faithful transcription of
    the essay and still fail an exact-match test because the model emitted a
    non-breaking hyphen (U+2011) where the essay has an ordinary one, or a
    non-breaking space inside a numeral. Checked on this corpus, 27 of the 44
    apparent quote failures for one judge differed from the source in nothing
    but punctuation codepoints -- one of them in a single character out of 184.

    Treating those as fabricated support would overstate the fabrication rate
    by a factor of about 2.6, so normalisation runs first: NFKC to fold
    compatibility forms, then every dash-like and apostrophe-like codepoint
    mapped to its ASCII equivalent. This is deliberately generous about
    typography and remains strict about words -- a paraphrase still fails.
    """
    text = unicodedata.normalize("NFKC", text)
    # The Unicode dash block (U+2010-U+2015), plus minus sign and non-breaking
    # hyphen, all become the ASCII hyphen.
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    # Curly and modifier apostrophes, and curly double quotes.
    text = re.sub(r"[\u2018\u2019\u02bc\u02bb]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    # Non-breaking and other exotic spaces; \s misses some of these.
    text = re.sub(r"[\u00a0\u2007\u202f\u2009\u200a]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_quote(quote: str, essay: str) -> tuple[bool, float]:
    """
    Check a quote against the essay it claims to come from.

    Returns (fully verified, fraction of words verified).

    Judges routinely join two passages with an ellipsis, which is honest
    quotation but fails a naive substring test -- in the first smoke run the
    only "unverified" quote turned out to be two real passages bridged by
    "...". The quote is therefore split on ellipses and each fragment checked
    separately. Fragments under four words are ignored, since short strings
    match by coincidence.
    """
    if not quote:
        return False, 0.0

    haystack = normalise_whitespace(essay)
    fragments = [f for f in re.split(r"\s*(?:\.{3}|…|\[\.\.\.\])\s*", quote) if f.strip()]

    verified_words, total_words = 0, 0
    all_found = True
    for fragment in fragments:
        needle = normalise_whitespace(fragment)
        words = len(needle.split())
        if words < 4:
            continue
        total_words += words
        if needle in haystack:
            verified_words += words
        else:
            all_found = False

    if total_words == 0:
        return False, 0.0
    return all_found, verified_words / total_words


def judgement_path(prompt_name: str, model: str, repetition: int,
                   code: str, replicate: int):
    """
    Where one cached judgement lives.

    Namespaced by baseline and by judge model, so several judges can rate the
    same essays without overwriting each other -- which is the whole point of
    running more than one.
    """
    directory = (config.PROCESSED_DIR / prompt_name / "judge"
                 / config.BASELINE / config.judge_slug())
    return directory / (f"{model_slug(model)}__rep{repetition:02d}"
                        f"__{code}__r{replicate}.json")


def judge_one(client: OpenAI, description: str, essay: str) -> dict:
    """One judging call, retrying with backoff. Returns the raw response fields."""
    prompt = build_prompt(description, essay)

    last_error = None
    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.JUDGE_TEMPERATURE,
                max_tokens=config.JUDGE_MAX_TOKENS,
            )
            message = response.choices[0].message

            # An empty completion is a truncation, not a judgement. Reasoning
            # models can spend the whole token budget thinking and return
            # nothing; treating that as a valid response would silently record
            # it as "absent". Raise so the retry loop handles it.
            if not message.content or not message.content.strip():
                raise RuntimeError(
                    "empty completion (finish_reason="
                    f"{response.choices[0].finish_reason!r}) — likely truncated "
                    "before the JSON; raise config.JUDGE_MAX_TOKENS"
                )

            return {
                "raw": message.content,
                "judge_id": response.id,
                "provider": getattr(response, "provider", None),
                "usage": response.usage.model_dump() if response.usage else None,
            }
        except Exception as error:  # noqa: BLE001 - retry on anything transient
            last_error = error
            if attempt < config.MAX_RETRIES - 1:
                delay = 2**attempt + random.uniform(0, 1)
                print(f"  [judge] attempt {attempt + 1} failed ({error}); "
                      f"retrying in {delay:.1f}s")
                time.sleep(delay)

    raise RuntimeError(
        f"All {config.MAX_RETRIES} judge attempts failed"
    ) from last_error


def run_task(client: OpenAI, prompt_name: str, essay_record: dict,
             code: str, description: str, replicate: int) -> str:
    """Judge one (essay, discourse, replicate) triple and cache it."""
    path = judgement_path(prompt_name, essay_record["model"],
                          essay_record["repetition"], code, replicate)

    result = judge_one(client, description, essay_record["essay"])

    try:
        parsed = normalise(extract_json(result["raw"]))
        parse_error = ""
    except Exception as error:  # noqa: BLE001 - a bad payload is data, not a crash
        # NOT "absent". A response we could not read is missing data, and
        # recording it as a substantive rating would let parse failures
        # masquerade as findings. Downstream code drops these rows and reports
        # how many there were.
        parsed = {"presence": "error", "treatment": "error",
                  "extent": "error", "quote": "", "reason": "",
                  "schema_problems": "unparseable"}
        parse_error = str(error)

    quote_verified, quote_match = verify_quote(parsed["quote"],
                                               essay_record["essay"])
    record = {
        "model": essay_record["model"],
        "repetition": essay_record["repetition"],
        "discourse": code,
        "replicate": replicate,
        "prompt_name": prompt_name,
        "baseline": config.BASELINE,
        "judge_model": config.JUDGE_MODEL,
        "judge_temperature": config.JUDGE_TEMPERATURE,
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "judge_id": result["judge_id"],
        "provider": result["provider"],
        "usage": result["usage"],
        "parse_error": parse_error,
        "quote_verified": quote_verified,
        "quote_match": round(quote_match, 3),
        "raw_response": result["raw"],
        **parsed,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                    encoding="utf-8")

    flag = ("" if quote_verified or parsed["presence"] == "absent"
            else f"  [quote {quote_match:.0%} verified]")
    return (f"OK   {essay_record['model'].split('/')[-1]:<22} "
            f"rep{essay_record['repetition']:02d}  {code}  r{replicate}  "
            f"{parsed['presence']:<12} {parsed['treatment']}{flag}")


def load_essays(prompt_name: str) -> list[dict]:
    """Every generated essay for this condition."""
    directory = config.GENERATED_DIR / prompt_name
    if not directory.exists():
        raise FileNotFoundError(
            f"No essays at {directory}. Run src/generate.py first."
        )
    records = []
    for path in sorted(directory.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    config.add_judge_argument(parser)
    parser.add_argument("--smoke", action="store_true",
                        help="judge one essay only, to check the setup")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-judge pairs that are already cached")
    parser.add_argument("--show-prompt", action="store_true",
                        help="print one fully built prompt and exit")
    args = parser.parse_args()
    prompt_name = config.apply_overrides(args)
    config.apply_judge_override(args)

    discourses = load_discourses()
    codes = [c for c in discourses["code"] if c in config.ACTIVE_DISCOURSES]
    names = dict(zip(discourses["code"], discourses["name"]))
    descriptions = {
        row.code: blind_description(row.code, row.text, names)
        for row in discourses.itertuples()
    }

    essays = load_essays(prompt_name)
    if args.smoke:
        essays = essays[:1]

    if args.show_prompt:
        print(build_prompt(descriptions[codes[0]], essays[0]["essay"]))
        return

    replicates = 1 if args.smoke else config.JUDGE_REPLICATES

    tasks = [
        (essay, code, replicate)
        for essay in essays
        for code in codes
        for replicate in range(1, replicates + 1)
        if args.overwrite or not judgement_path(
            prompt_name, essay["model"], essay["repetition"], code, replicate
        ).exists()
    ]

    total = len(essays) * len(codes) * replicates
    print(f"Judge:    {config.JUDGE_MODEL} @ temperature {config.JUDGE_TEMPERATURE}")
    print(f"Baseline: {config.baseline()['label']}")
    print(f"Pairs:    {len(essays)} essays x {len(codes)} discourses "
          f"x {replicates} replicates = {total}")
    print(f"{total - len(tasks)} cached, {len(tasks)} to run.\n")

    if not tasks:
        print("Nothing to do. Run src/analyse_judge.py to summarise.")
        return

    client = build_client()
    failures = []
    with ThreadPoolExecutor(max_workers=config.JUDGE_MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_task, client, prompt_name, essay, code,
                        descriptions[code], replicate): (essay, code, replicate)
            for essay, code, replicate in tasks
        }
        for future in as_completed(futures):
            essay, code, replicate = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:  # noqa: BLE001 - report and continue
                failures.append((essay["model"], essay["repetition"], code, error))
                print(f"FAIL {essay['model']} rep{essay['repetition']:02d} "
                      f"{code} r{replicate}: {error}", flush=True)

    print(f"\nDone. {len(tasks) - len(failures)} judged, {len(failures)} failed.")
    if failures:
        print("Re-run this script to retry the failures.")
    else:
        print("Next: python src/analyse_judge.py")


if __name__ == "__main__":
    main()
