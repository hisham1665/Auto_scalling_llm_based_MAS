from evaluation.evaluator import evaluate_conversation
from evaluation.keyword_coverage import task_related_coverage
from evaluation.mtld import mtld
from evaluation.statistics import cliffs_delta, mann_whitney_u, pearson_correlation


def test_evaluation_metrics_are_computed_without_optional_packages() -> None:
    conversation = [{"role": "agent", "content": "The doctor reviews fever and abdominal pain with imaging and monitoring."}]
    vocabulary = {"symptoms": ("fever", "abdominal pain"), "tests": ("imaging",), "safety": ("monitoring",)}
    metrics = evaluate_conversation("Review fever and abdominal pain", conversation, vocabulary)
    assert metrics["task_coverage"] == 1.0
    assert metrics["tfidf"] >= 0.0
    assert metrics["mtld"] > 0
    assert 0.0 <= metrics["thematic_relevance"]["score"] <= 1.0


def test_statistics_helpers() -> None:
    assert task_related_coverage("fever imaging", {"x": ("fever", "imaging")})["score"] == 1.0
    assert mtld("one two three four five six") > 0
    assert cliffs_delta([3, 4], [1, 2]) == 1.0
    assert mann_whitney_u([3, 4], [1, 2]).effect_size == 1.0
    assert pearson_correlation([1, 2, 3], [1, 2, 3]) == 1.0
