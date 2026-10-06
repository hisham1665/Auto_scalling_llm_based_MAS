"""Optional matplotlib plots for saved experiment JSON files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def generate_plots(results: list[dict[str, Any]], output_dir: str | Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completed = [item for item in results if item.get("status") == "completed"]
    if not completed:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in completed:
        grouped[item.get("approach", "unknown")].append(item)
    paths: list[str] = []
    for metric, label in (("agent_count", "Agent count"), ("task_coverage", "Task-related coverage"), ("tfidf", "TF-IDF richness"), ("mtld", "MTLD"), ("bertscore", "BERTScore / fallback")):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for approach, values in sorted(grouped.items()):
            if metric == "agent_count":
                series = [len(item.get("final_agents", [])) for item in values]
            else:
                series = [float(item.get("metrics", {}).get(metric, 0.0)) for item in values]
            axis.plot(range(1, len(series) + 1), series, marker="o", label=approach)
        axis.set_title(label)
        axis.set_xlabel("Run index")
        axis.set_ylabel(label)
        axis.legend()
        figure.tight_layout()
        path = output / f"{metric}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(str(path))
    figure, axis = plt.subplots(figsize=(7, 5))
    for approach, values in sorted(grouped.items()):
        axis.scatter(
            [len(item.get("final_agents", [])) for item in values],
            [float(item.get("metrics", {}).get("task_coverage", 0.0)) for item in values],
            label=approach,
        )
    axis.set_title("Agent count vs task-related coverage")
    axis.set_xlabel("Final agent count")
    axis.set_ylabel("Task-related coverage")
    axis.legend()
    figure.tight_layout()
    path = output / "agent_count_vs_coverage.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths.append(str(path))

    figure, axis = plt.subplots(figsize=(9, 5))
    approaches = sorted(grouped)
    metric_names = ["task_coverage", "tfidf", "mtld", "bertscore"]
    x_positions = list(range(len(approaches)))
    width = 0.18
    for offset, metric in enumerate(metric_names):
        values = [
            sum(float(item.get("metrics", {}).get(metric, 0.0)) for item in grouped[approach]) / len(grouped[approach])
            for approach in approaches
        ]
        axis.bar([position + (offset - 1.5) * width for position in x_positions], values, width, label=metric)
    axis.set_title("Static vs IAAG vs DRTAG metric comparison")
    axis.set_xticks(x_positions, approaches)
    axis.set_ylabel("Mean metric value")
    axis.legend()
    figure.tight_layout()
    path = output / "approach_comparison.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths.append(str(path))
    return paths
