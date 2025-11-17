from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


PROHIBITED_KEYWORDS = {
    "nudity",
    "gore",
    "blood",
    "weapon",
    "violence",
    "hate",
    "self harm",
}


@dataclass
class ContentFilterResult:
    allowed: bool
    reason: str | None = None
    blocked_terms: list[str] | None = None


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def check_prompt(prompt: str, extra_banned: Iterable[str] | None = None) -> ContentFilterResult:
    """Basic lexical filter for inappropriate prompts."""

    normalized = normalize(prompt)
    banned = set(PROHIBITED_KEYWORDS)
    if extra_banned:
        banned.update(normalize(term) for term in extra_banned if term)

    found = sorted({term for term in banned if term and term in normalized})

    if found:
        return ContentFilterResult(
            allowed=False,
            reason="Prompt contains blocked content",
            blocked_terms=found,
        )

    return ContentFilterResult(allowed=True)

