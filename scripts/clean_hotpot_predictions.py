#!/usr/bin/env python3
"""Post-process HotpotQA prediction files to normalize answer strings."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentragdrop.answer_cleaning import clean_answer


def derive_output_path(path: Path, suffix: str) -> Path:
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.with_name(f"{path.name}{suffix}")


def process_file(
    input_path: Path,
    output_path: Path,
    question_field: str,
    answer_field: str,
    raw_field: Optional[str],
    max_tokens: int,
) -> dict:
    changed = 0
    total = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_answer = obj.get(answer_field, "")
            question = obj.get(question_field, "")
            cleaned = clean_answer(raw_answer, question, max_tokens=max_tokens)

            if raw_field:
                obj.setdefault(raw_field, raw_answer)
            obj[answer_field] = cleaned

            if cleaned != raw_answer:
                changed += 1
            total += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "examples": total,
        "changed": changed,
    }


def iter_inputs(paths: Iterable[str]) -> Iterable[Path]:
    for pattern in paths:
        p = Path(pattern)
        if p.is_file():
            yield p
            continue
        for candidate in sorted(Path().glob(pattern)):
            if candidate.is_file():
                yield candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean HotpotQA prediction outputs using heuristic normalization.")
    parser.add_argument("inputs", nargs="+", help="Input JSONL files or glob patterns to clean.")
    parser.add_argument("--output-suffix", default="clean", help="Suffix to append to cleaned files (default: 'clean').")
    parser.add_argument("--question-field", default="question", help="JSON field containing the question text.")
    parser.add_argument("--answer-field", default="pred_answer", help="JSON field containing the predicted answer.")
    parser.add_argument("--raw-field", default="raw_pred_answer", help="Field to store the original answer (blank to disable).")
    parser.add_argument("--max-tokens", type=int, default=6, help="Maximum tokens to keep after cleaning (default: 6).")

    args = parser.parse_args(argv)

    outputs = []
    suffix = args.output_suffix

    for input_path in iter_inputs(args.inputs):
        out_path = derive_output_path(input_path, suffix)
        raw_field = args.raw_field or None
        stats = process_file(
            input_path,
            out_path,
            question_field=args.question_field,
            answer_field=args.answer_field,
            raw_field=raw_field,
            max_tokens=args.max_tokens,
        )
        outputs.append(stats)

    print("Cleaned files:")
    for info in outputs:
        pct = (info["changed"] / info["examples"] * 100.0) if info["examples"] else 0.0
        print(f"  {info['input']} -> {info['output']} (changed {info['changed']}/{info['examples']} = {pct:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
