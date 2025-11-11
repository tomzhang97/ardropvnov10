#!/usr/bin/env python3
import argparse
import json


def clean_answer(raw: str, question: str) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()

    # 1) Use explicit answer markers if present (last one wins)
    markers = ["Final answer:", "Answer:", "The answer is"]
    lower_text = text.lower()
    for m in markers:
        m_lower = m.lower()
        if m_lower in lower_text:
            idx = lower_text.rfind(m_lower)
            text = text[idx + len(m):].strip()
            lower_text = text.lower()
            break

    # 2) Heuristic yes/no detection from question
    lower_q = (question or "").lower()
    lower_t = text.lower()
    yesno_starts = (
        "is ", "are ", "was ", "were ",
        "do ", "does ", "did ",
        "can ", "could ", "will ", "would ",
        "has ", "have ", "had "
    )
    if lower_q.startswith(yesno_starts) or "yes or no" in lower_q:
        has_yes = "yes" in lower_t
        has_no = "no" in lower_t
        if has_yes and not has_no:
            return "yes"
        if has_no and not has_yes:
            return "no"

    # 3) Cut at first sentence boundary
    for sep in [".", "?", "!"]:
        if sep in text:
            text = text.split(sep)[0]
            break

    # 4) Strip quotes and spaces
    text = text.strip().strip('"').strip("'")

    # 5) Truncate to at most 6 tokens (Hotpot answers are short)
    tokens = text.split()
    if len(tokens) > 6:
        text = " ".join(tokens[:6])

    return text


def get_field(obj, *candidates, default=""):
    for k in candidates:
        if k in obj:
            return obj[k]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input JSONL results")
    ap.add_argument("--output", required=True, help="output JSONL with cleaned predictions")
    ap.add_argument("--pred_key", default=None, help="field name for prediction (default: auto-detect)")
    ap.add_argument("--q_key", default=None, help="field name for question (default: auto-detect)")
    args = ap.parse_args()

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            obj = json.loads(line)

            q_key = args.q_key or None
            p_key = args.pred_key or None

            question = obj.get(q_key) if q_key \
                else get_field(obj, "question", "q", "query")
            raw_pred = obj.get(p_key) if p_key \
                else get_field(obj, "prediction", "pred", "answer")

            cleaned = clean_answer(raw_pred, question)

            # Overwrite the field used by your evaluator.
            # If evaluator reads "prediction", update that:
            obj["prediction"] = cleaned

            # Optionally also keep original:
            # obj["prediction_raw"] = raw_pred

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
