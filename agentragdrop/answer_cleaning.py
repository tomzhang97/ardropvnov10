"""Utilities for normalizing HotpotQA answers."""

from __future__ import annotations

import re
from typing import Optional

__all__ = ["clean_answer"]


_ANSWER_MARKERS = [
    "answer:",
    "final answer:",
    "the answer is",
]

_YES_NO_PREFIXES = (
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "will ",
    "would ",
)


def _extract_after_markers(text: str) -> str:
    lowered = text.lower()
    for marker in _ANSWER_MARKERS:
        idx = lowered.rfind(marker)
        if idx != -1:
            start = idx + len(marker)
            return text[start:]
    return text


def _is_yes_no_question(question: str) -> bool:
    question = question.strip().lower()
    if any(question.startswith(prefix) for prefix in _YES_NO_PREFIXES):
        return True
    return "yes or no" in question


def clean_answer(raw: str, question: Optional[str] = None, *, max_tokens: int = 8) -> str:
    """Normalize a free-form model answer into a short canonical span.

    Args:
        raw: Original model output string.
        question: The associated question text (used for yes/no heuristics).
        max_tokens: Maximum number of tokens to keep in the final answer.

    Returns:
        Cleaned answer string suitable for HotpotQA evaluation.
    """
    if raw is None:
        return "unknown"

    text = str(raw).strip()
    if not text:
        return "unknown"

    # Replace obviously empty placeholders
    if re.fullmatch(r"[_\W]+", text):
        return "unknown"

    # Collapse internal whitespace early
    text = re.sub(r"\s+", " ", text)

    # 1) Explicit answer markers
    extracted = _extract_after_markers(text)
    if extracted != text:
        text = extracted.strip()

    # 2) Strip quotes and surrounding punctuation
    text = text.strip().strip('"').strip("'")

    # 3) Handle yes/no questions
    if question:
        lower_text = text.lower()
        if _is_yes_no_question(question):
            if "yes" in lower_text and "no" not in lower_text:
                return "yes"
            if "no" in lower_text and "yes" not in lower_text:
                return "no"

    # 4) Truncate at the first sentence boundary
    for sep in [".", "?", "!"]:
        if sep in text:
            text = text.split(sep)[0]
            break

    # 5) Remove trailing punctuation again after truncation
    text = text.strip().strip('"').strip("'")
    text = text.rstrip(".,;:!?")

    # 6) Collapse whitespace once more
    text = re.sub(r"\s+", " ", text).strip()

    # 7) Enforce max token length
    tokens = text.split()
    if len(tokens) > max_tokens:
        text = " ".join(tokens[:max_tokens])

    if not text:
        return "unknown"

    return text
