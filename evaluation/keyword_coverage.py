"""Task-related vocabulary coverage.

Engineering definition: the score is the fraction of unique configured vocabulary
terms found in the agent conversation (case-insensitive substring/token matching).
Phrase terms are supported.  This is a transparent proxy, not the paper's
original proprietary evaluation setup.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


def _text(conversation: str | Sequence[Mapping[str, Any]]) -> str:
    if isinstance(conversation, str):
        return conversation.lower()
    return " ".join(str(item.get("content", "")) for item in conversation).lower()


def task_related_coverage(
    conversation: str | Sequence[Mapping[str, Any]], vocabulary: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    text = _text(conversation)
    categories: dict[str, float] = {}
    found: list[str] = []
    found_labels: list[str] = []
    all_terms: list[str] = []
    for category, terms in vocabulary.items():
        normalized = [str(term).strip().lower() for term in terms if str(term).strip()]
        hits = [term for term in normalized if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)]
        all_terms.extend(normalized)
        found.extend(hits)
        found_labels.extend(f"{category}:{term}" for term in hits)
        categories[category] = len(hits) / len(normalized) if normalized else 0.0
    denominator = len(set(all_terms))
    return {
        "score": len(set(found)) / denominator if denominator else 0.0,
        "category_scores": categories,
        "found_terms": sorted(set(found_labels)),
        "vocabulary_size": denominator,
    }
