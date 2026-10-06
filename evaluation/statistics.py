"""Small-sample statistics used by experiment comparisons."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(slots=True)
class StatisticalTest:
    metric: str
    n_a: int
    n_b: int
    p_value: float
    effect_size: float
    interpretation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for position in range(index, end):
            ranks[indexed[position][0]] = rank
        index = end
    return ranks


def mann_whitney_u(a: Sequence[float], b: Sequence[float], metric: str = "metric") -> StatisticalTest:
    if not a or not b:
        return StatisticalTest(metric, len(a), len(b), 1.0, 0.0, "insufficient samples")
    combined = list(a) + list(b)
    ranks = _ranks(combined)
    n_a, n_b = len(a), len(b)
    u_a = sum(ranks[:n_a]) - n_a * (n_a + 1) / 2
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)
    mean = n_a * n_b / 2
    tie_counts = {}
    for value in combined:
        tie_counts[value] = tie_counts.get(value, 0) + 1
    tie_term = sum(count**3 - count for count in tie_counts.values())
    variance = n_a * n_b / 12 * (n_a + n_b + 1 - tie_term / ((n_a + n_b) * (n_a + n_b - 1)))
    z = (u - mean + 0.5) / math.sqrt(variance) if variance else 0.0
    p_value = math.erfc(abs(z) / math.sqrt(2))
    delta = cliffs_delta(a, b)
    interpretation = "statistically notable at p<0.05" if p_value < 0.05 else "not statistically notable at p<0.05"
    return StatisticalTest(metric, n_a, n_b, p_value, delta, interpretation)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    greater = sum(x > y for x in a for y in b)
    lesser = sum(x < y for x in a for y in b)
    return (greater - lesser) / (len(a) * len(b))


def pearson_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return numerator / denominator if denominator else 0.0
