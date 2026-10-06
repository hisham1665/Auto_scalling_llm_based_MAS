"""A dependency-free implementation of the MTLD lexical-diversity measure."""

from __future__ import annotations

import re


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())


def _factor_length(tokens: list[str], threshold: float = 0.72) -> float:
    if not tokens:
        return 0.0
    types: set[str] = set()
    factors = 0
    start = 0
    for index, token in enumerate(tokens, 1):
        types.add(token)
        ttr = len(types) / index
        if ttr <= threshold:
            factors += 1
            start = index
            types = set()
    remainder = len(tokens) - start
    if remainder:
        residual_ttr = len(types) / remainder
        if residual_ttr != 1.0:
            factors += (1.0 - residual_ttr) / (1.0 - threshold)
        else:
            factors += 1.0
    return len(tokens) / factors if factors else float(len(tokens))


def mtld(text: str, threshold: float = 0.72) -> float:
    tokens = _tokens(text)
    if len(tokens) < 2:
        return float(len(tokens))
    forward = _factor_length(tokens, threshold)
    backward = _factor_length(list(reversed(tokens)), threshold)
    return (forward + backward) / 2.0
