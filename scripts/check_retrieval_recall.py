#!/usr/bin/env python3
"""Compute retrieval recall statistics from HotpotQA prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence


def _ensure_sequence(value) -> Sequence[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    return [str(value)]


def answer_in_context_rate(path: Path, limit: int | None) -> dict:
    total = 0
    em_sum = 0.0
    f1_sum = 0.0
    ctx_total = 0
    ctx_hits = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1

            em_sum += float(obj.get("em", 0.0))
            f1_sum += float(obj.get("f1", 0.0))

            gold = str(obj.get("gold_answer", "")).strip().lower()
            contexts = _ensure_sequence(
                obj.get("contexts")
                or obj.get("retrieved_context")
                or obj.get("evidence")
                or obj.get("context")
            )
            if gold:
                ctx_total += 1
                joined = " \n".join(contexts).lower()
                if gold in joined:
                    ctx_hits += 1

            if limit and total >= limit:
                break

    metrics = {
        "examples": total,
        "em_mean": (em_sum / total) if total else 0.0,
        "f1_mean": (f1_sum / total) if total else 0.0,
        "answer_in_context": (ctx_hits / ctx_total) if ctx_total else 0.0,
        "answer_in_context_hits": ctx_hits,
        "answer_in_context_total": ctx_total,
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect retrieval recall quality for HotpotQA predictions.")
    parser.add_argument("results", help="Prediction JSONL file to analyze.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on the number of examples to read.")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    path = Path(args.results)
    if not path.is_file():
        parser.error(f"Results file not found: {path}")

    metrics = answer_in_context_rate(path, limit)

    print(f"Examples: {metrics['examples']}")
    print(f"EM (mean): {metrics['em_mean']:.3f}")
    print(f"F1 (mean): {metrics['f1_mean']:.3f}")
    hits = metrics["answer_in_context_hits"]
    total = metrics["answer_in_context_total"]
    if total:
        print(f"Answer-in-context rate: {metrics['answer_in_context']:.3f} ({hits}/{total})")
    else:
        print("Answer-in-context rate: N/A (no gold answers available)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
