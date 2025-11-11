#!/usr/bin/env python3
"""Compute answer-in-context statistics for HotpotQA prediction files."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentragdrop.answer_cleaning import clean_answer


def iter_files(patterns: Iterable[str]) -> Iterable[Path]:
    for pat in patterns:
        path = Path(pat)
        if path.is_file():
            yield path
            continue
        for candidate in sorted(Path().glob(pat)):
            if candidate.is_file():
                yield candidate


def answer_in_context_stats(path: Path, limit: Optional[int], lowercase: bool) -> dict:
    total = 0
    hits = 0
    missing_context = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj.get("question", "")
            gold = obj.get("gold_answer", "")
            contexts = obj.get("retrieved_context") or obj.get("evidence") or []

            if not contexts:
                missing_context += 1
                continue

            if isinstance(contexts, str):
                contexts_list = [contexts]
            else:
                contexts_list = [str(c) for c in contexts]

            joined = " \n".join(contexts_list)
            gold_clean = clean_answer(gold, question)

            if lowercase:
                joined = joined.lower()
                gold_clean = gold_clean.lower()

            total += 1
            if gold_clean and gold_clean in joined:
                hits += 1

            if limit and total >= limit:
                break

    return {
        "file": str(path),
        "examples": total,
        "hits": hits,
        "missing_context": missing_context,
        "rate": (hits / total) if total else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure how often the gold answer appears in retrieved contexts.")
    parser.add_argument("inputs", nargs="+", help="Prediction JSONL files or glob patterns.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of examples per file (0 = no limit).")
    parser.add_argument("--case-sensitive", action="store_true", help="Do not lowercase text before matching.")

    args = parser.parse_args(argv)
    limit = args.limit if args.limit > 0 else None
    lowercase = not args.case_sensitive

    if not args.inputs:
        parser.error("No input files provided")

    reports = []
    for file_path in iter_files(args.inputs):
        stats = answer_in_context_stats(file_path, limit, lowercase)
        reports.append(stats)

    print("Answer-in-context rates:")
    for info in reports:
        pct = info["rate"] * 100.0 if info["examples"] else 0.0
        miss = info["missing_context"]
        print(
            f"  {info['file']}: {pct:.1f}% ({info['hits']}/{info['examples']})",
            f"missing_context={miss}" if miss else ""
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
