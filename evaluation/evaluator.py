"""Metric orchestration for saved runs or in-memory conversations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.bertscore import bertscore
from evaluation.keyword_coverage import task_related_coverage
from evaluation.mtld import mtld
from evaluation.tfidf import tfidf_richness


def evaluate_conversation(
    task: str,
    conversation: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, Sequence[str]],
    *,
    evaluation_model: str = "distilbert-base-uncased",
    use_neural_bertscore: bool = False,
    generated_agent_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    agent_messages = [item for item in conversation if item.get("role") == "agent"]
    responses = [str(item.get("content", "")) for item in agent_messages]
    full_text = " ".join(responses)
    all_terms = [term for terms in vocabulary.values() for term in terms]
    coverage = task_related_coverage(agent_messages, vocabulary)
    generated_names = {name.casefold() for name in (generated_agent_names or [])}
    generated_messages = [
        item for item in agent_messages if str(item.get("agent", "")).casefold() in generated_names
    ]
    generated_coverage = task_related_coverage(generated_messages, vocabulary) if generated_names else {
        "score": 0.0,
        "category_scores": {},
        "found_terms": [],
        "vocabulary_size": len({term for terms in vocabulary.values() for term in terms}),
    }
    thematic = bertscore(
        [full_text],
        [task],
        model_type=evaluation_model,
        use_neural=use_neural_bertscore,
    )
    pairwise = bertscore(
        responses[1:], responses[:-1],
        model_type=evaluation_model,
        use_neural=use_neural_bertscore,
    ) if len(responses) > 1 else {"score": 0.0, "backend": "insufficient_messages"}
    return {
        "task_coverage": coverage["score"],
        "generated_agent_coverage": generated_coverage["score"],
        "coverage_by_category": coverage["category_scores"],
        "covered_terms": coverage["found_terms"],
        "tfidf": tfidf_richness(responses or [""], all_terms),
        "mtld": mtld(full_text),
        "bertscore": pairwise["score"],
        "pairwise_topical_consistency": pairwise,
        "thematic_relevance": thematic,
        "evaluation_backend": thematic.get("backend"),
    }
