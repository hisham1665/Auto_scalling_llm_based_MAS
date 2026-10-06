"""CLI for demos, individual runs, experiments, and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config.settings import Settings
from evaluation.evaluator import evaluate_conversation
from evaluation.statistics import mann_whitney_u, pearson_correlation
from experiments.experiment_runner import ExperimentRunner
from experiments.plots import generate_plots
from experiments.scenarios import SCENARIOS, get_scenario


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IAAG/DRTAG dynamic multi-agent research runner")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the real Ollama DRTAG medical demonstration")
    demo.add_argument("--scenario", default="medical", choices=sorted(SCENARIOS))
    run = sub.add_parser("run", help="run one configuration")
    run.add_argument("--approach", choices=("static", "iaag", "drtag"), required=True)
    run.add_argument("--selection", choices=("llm", "round_robin", "random"), required=True)
    run.add_argument("--scenario", default="medical", choices=sorted(SCENARIOS))
    run.add_argument("--max-turns", type=int)
    run.add_argument("--random-seed", type=int)
    experiment = sub.add_parser("experiment", help="run experiments")
    experiment.add_argument("--all", action="store_true", help="execute all nine configurations")
    experiment.add_argument("--scenario", default="medical", choices=sorted(SCENARIOS))
    experiment.add_argument("--runs", type=int)
    experiment.add_argument("--max-turns", type=int)
    evaluate = sub.add_parser("evaluate", help="recompute metrics for saved JSON results")
    evaluate.add_argument("--results", default="results")
    compare = sub.add_parser("compare", help="summarize saved results and create plots")
    compare.add_argument("--results", default="results")
    agents = sub.add_parser("list-agents", help="list scenario seed and static agents")
    agents.add_argument("--scenario", default="medical", choices=sorted(SCENARIOS))
    return parser


def _json_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if "failed" not in path.parts)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    settings = Settings.from_env()
    _configure_logging(settings)
    if args.command == "list-agents":
        scenario = get_scenario(args.scenario)
        print("Seed agents:")
        for name, role, _ in scenario.seed_agent_specs:
            print(f"  - {name}: {role}")
        print("Static baseline agents:")
        for name, role, _ in scenario.static_agent_specs:
            print(f"  - {name}: {role}")
        return
    runner = ExperimentRunner(settings)
    if args.command == "demo":
        payload = runner.run_experiment(approach="drtag", selection="llm", scenario=args.scenario, max_turns=settings.max_turns)
        print(json.dumps(payload, indent=2))
        return
    if args.command == "run":
        payload = runner.run_experiment(
            approach=args.approach,
            selection=args.selection,
            scenario=args.scenario,
            random_seed=args.random_seed,
            max_turns=args.max_turns,
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "experiment":
        if not args.all:
            raise SystemExit("Use --all to execute the nine configurations")
        payloads = runner.run_all(scenario=args.scenario, runs_per_configuration=args.runs, max_turns=args.max_turns)
        print(json.dumps({"count": len(payloads), "results": payloads}, indent=2))
        return
    if args.command in {"evaluate", "compare"}:
        root = Path(args.results)
        files = _json_files(root)
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        for payload in payloads:
            if payload.get("status") != "completed":
                continue
            scenario = get_scenario(payload["scenario"])
            payload["metrics"] = evaluate_conversation(
                payload["task"],
                payload["conversation"],
                scenario.vocabulary,
                generated_agent_names=[item["name"] for item in payload.get("generated_agents", [])],
            )
        if args.command == "evaluate":
            print(json.dumps(payloads, indent=2))
        else:
            summary = [
                {"experiment_id": p.get("experiment_id"), "approach": p.get("approach"), "selection": p.get("selection_strategy"), "agents": len(p.get("final_agents", [])), "turns": p.get("turn_count"), "metrics": p.get("metrics", {})}
                for p in payloads
            ]
            completed = [p for p in payloads if p.get("status") == "completed"]
            comparisons = []
            for metric in ("task_coverage", "tfidf", "mtld", "bertscore"):
                static = [float(p.get("metrics", {}).get(metric, 0.0)) for p in completed if p.get("approach") == "static"]
                iaag = [float(p.get("metrics", {}).get(metric, 0.0)) for p in completed if p.get("approach") == "iaag"]
                drtag = [float(p.get("metrics", {}).get(metric, 0.0)) for p in completed if p.get("approach") == "drtag"]
                comparisons.extend([
                    mann_whitney_u(static, iaag, f"{metric}: static vs iaag").to_dict(),
                    mann_whitney_u(static, drtag, f"{metric}: static vs drtag").to_dict(),
                    mann_whitney_u(iaag, drtag, f"{metric}: iaag vs drtag").to_dict(),
                ])
            coverage = [float(p.get("metrics", {}).get("task_coverage", 0.0)) for p in completed]
            agent_counts = [float(len(p.get("final_agents", []))) for p in completed]
            print(json.dumps({"summary": summary, "statistical_comparisons": comparisons, "agent_count_coverage_pearson": pearson_correlation(agent_counts, coverage)}, indent=2))
            print("plots:", generate_plots(payloads, root / "plots"))
