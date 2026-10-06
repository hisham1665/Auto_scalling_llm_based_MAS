"""The nine fixed architecture/selection configurations."""

from __future__ import annotations

CONFIGURATIONS = tuple(
    {"approach": approach, "selection_strategy": selection}
    for approach in ("static", "iaag", "drtag")
    for selection in ("llm", "round_robin", "random")
)


def all_configurations() -> list[dict[str, str]]:
    return [dict(item) for item in CONFIGURATIONS]
