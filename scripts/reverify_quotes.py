"""
Recompute quote verification on already-cached judgements.

The judge stores `quote_verified` and `quote_match` at call time, so fixing the
verifier does not by itself fix the stored results. This walks the cache and
recomputes both fields from the stored quote and the source essay. No API calls
are made and no rating is touched -- presence, treatment and extent are exactly
as the judge returned them.

Run after any change to judge.verify_quote or judge.normalise_whitespace.

Usage:
    python scripts/reverify_quotes.py --dry-run     # report, change nothing
    python scripts/reverify_quotes.py               # rewrite the cache
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config
from judge import verify_quote


def load_essays(prompt_name: str) -> dict[str, str]:
    """Reassemble each essay from the paragraph table, keyed by run_id."""
    paragraphs = pd.read_csv(config.PROCESSED_DIR / prompt_name / "paragraphs.csv")
    return {run_id: " ".join(group["text"])
            for run_id, group in paragraphs.groupby("run_id")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_prompt_argument(parser)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()
    prompt_name = config.apply_overrides(args)

    essays = load_essays(prompt_name)
    judge_root = config.PROCESSED_DIR / prompt_name / "judge" / config.BASELINE
    if not judge_root.exists():
        raise FileNotFoundError(f"No judge cache at {judge_root}")

    for judge_dir in sorted(p for p in judge_root.iterdir() if p.is_dir()):
        files = sorted(judge_dir.glob("*.json"))
        changed = gained = lost = 0

        for path in files:
            record = json.loads(path.read_text())

            # Absent judgements carry no quote and are not verified either way.
            if record.get("presence") in (None, "absent") or record.get("parse_error"):
                continue

            run_id = f"{record['model'].replace('/', '__')}__rep{record['repetition']:02d}"
            essay = essays.get(run_id)
            if essay is None:
                raise KeyError(f"No essay for {run_id}")

            was = bool(record.get("quote_verified"))
            now, match = verify_quote(record.get("quote", ""), essay)

            if now == was and round(match, 3) == record.get("quote_match"):
                continue

            changed += 1
            gained += int(now and not was)
            lost += int(was and not now)

            if not args.dry_run:
                record["quote_verified"] = now
                record["quote_match"] = round(match, 3)
                path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

        verb = "would change" if args.dry_run else "changed"
        print(f"{judge_dir.name}: {len(files)} cached, {verb} {changed} "
              f"(newly verified {gained}, newly failed {lost})")


if __name__ == "__main__":
    main()
