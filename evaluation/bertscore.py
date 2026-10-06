"""Optional BERTScore metric with a deterministic lexical fallback.

The fallback exists for laptop/offline environments and is explicitly labeled in
the result. It uses token F1 and should not be interpreted as neural BERTScore.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower()))


def lexical_f1(candidate: str, reference: str) -> float:
    candidate_tokens, reference_tokens = _tokens(candidate), _tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = len(candidate_tokens & reference_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bertscore(
    candidates: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = "distilbert-base-uncased",
    device: str = "cpu",
    use_neural: bool = False,
) -> dict[str, float | str]:
    if not candidates or not references:
        return {"score": 0.0, "backend": "empty"}
    if use_neural:
        try:
            from bert_score import score

            _, _, f1 = score(list(candidates), list(references), model_type=model_type, device=device, verbose=False)
            return {"score": float(f1.mean().item()), "backend": "bert-score", "model": model_type}
        except Exception:
            pass
    pairs = zip(candidates, references)
    values = [lexical_f1(candidate, reference) for candidate, reference in pairs]
    return {"score": sum(values) / len(values), "backend": "lexical_f1_fallback"}
