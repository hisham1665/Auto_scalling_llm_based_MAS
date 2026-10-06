"""Reusable experiment execution, persistence, and nine-configuration runs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from approaches.drtag import DRTAG
from approaches.iaag import IAAG
from approaches.static_baseline import StaticBaseline
from config.settings import Settings
from evaluation.evaluator import evaluate_conversation
from experiments.configurations import all_configurations
from experiments.scenarios import Scenario, get_scenario
from llm.base_llm import BaseLLM
from llm.ollama_client import OllamaLLM

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


class ExperimentRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: BaseLLM | None = None,
        result_dir: str | Path | None = None,
        llm_factory: Callable[[Settings], BaseLLM] | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.llm = llm
        self.llm_factory = llm_factory or (lambda current: OllamaLLM(current))
        self.result_dir = Path(result_dir or self.settings.results_dir)

    def _provider(self) -> BaseLLM:
        if self.llm is None:
            candidate = self.llm_factory(self.settings)
            try:
                candidate.validate_availability()
            except Exception:
                self.llm = None
                raise
            self.llm = candidate
        return self.llm

    def _save(self, payload: dict[str, Any], *, failed: bool = False) -> Path:
        date_dir = self.result_dir / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target_dir = date_dir / "failed" if failed else date_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        experiment_id = payload["experiment_id"]
        path = target_dir / f"{experiment_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path

    def run_experiment(
        self,
        *,
        approach: str,
        selection: str,
        scenario: str | Scenario,
        random_seed: int | None = None,
        max_turns: int | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        current_settings = replace(
            self.settings,
            random_seed=self.settings.random_seed if random_seed is None else random_seed,
            max_turns=self.settings.max_turns if max_turns is None else max_turns,
        )
        selected_scenario = get_scenario(scenario) if isinstance(scenario, str) else scenario
        experiment_id = f"experiment_{datetime.now(timezone.utc).strftime('%H%M%S')}_{uuid4().hex[:8]}"
        started = time.monotonic()
        llm: BaseLLM | None = None
        logger.info("experiment started: %s %s %s", experiment_id, approach, selection)
        try:
            llm = self._provider()
            initial_agents = selected_scenario.create_agents(llm, static=approach.lower() == "static")
            approach_runner = {"static": StaticBaseline(), "iaag": IAAG(), "drtag": DRTAG()}.get(approach.lower())
            if approach_runner is None:
                raise ValueError(f"Unknown approach {approach!r}")
            run_result = approach_runner.run(
                selected_scenario.task,
                initial_agents,
                llm,
                settings=current_settings,
                selection_strategy=selection,
            )
            metrics = evaluate_conversation(
                selected_scenario.task,
                run_result.conversation,
                selected_scenario.vocabulary,
                evaluation_model=current_settings.evaluation_model,
                use_neural_bertscore=current_settings.enable_bertscore,
                generated_agent_names=[item["name"] for item in run_result.generated_agents],
            )
            payload = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": llm.model,
                "model_name": llm.model,
                "model_configuration": {
                    "provider": "ollama",
                    "base_url": getattr(llm, "base_url", None),
                    "temperature": current_settings.temperature,
                    "max_response_tokens": current_settings.max_response_tokens,
                },
                "scenario": selected_scenario.name,
                "task": selected_scenario.task,
                "approach": approach.lower(),
                "selection_strategy": selection,
                "random_seed": current_settings.random_seed,
                "max_turns": current_settings.max_turns,
                "initial_agents": [agent["name"] for agent in run_result.initial_agents],
                "generated_agents": run_result.generated_agents,
                "final_agents": [agent["name"] for agent in run_result.final_agents],
                "turn_count": run_result.turn_count,
                "termination_reason": run_result.termination_reason,
                "duration_seconds": round(time.monotonic() - started, 4),
                "token_information": run_result.token_usage,
                "configuration": current_settings.as_dict(),
                "conversation": run_result.conversation,
                "agents": run_result.final_agents,
                "events": run_result.events,
                "metrics": metrics,
                "errors": run_result.errors,
                "retries": run_result.retries,
                "status": "completed",
            }
        except Exception as error:
            logger.error("experiment failed: %s: %s", experiment_id, error)
            payload = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": getattr(llm, "model", current_settings.ollama_model),
                "model_name": getattr(llm, "model", current_settings.ollama_model),
                "scenario": selected_scenario.name,
                "approach": approach.lower(),
                "selection_strategy": selection,
                "random_seed": current_settings.random_seed,
                "max_turns": current_settings.max_turns,
                "duration_seconds": round(time.monotonic() - started, 4),
                "status": "failed",
                "error": str(error),
            }
            if save:
                path = self._save(payload, failed=True)
                payload["result_path"] = str(path)
            return payload
        if save:
            path = self._save(payload)
            payload["result_path"] = str(path)
        logger.info("experiment completed: %s", experiment_id)
        return payload

    def run_all(
        self,
        *,
        scenario: str = "medical",
        runs_per_configuration: int | None = None,
        max_turns: int | None = None,
    ) -> list[dict[str, Any]]:
        runs = runs_per_configuration or self.settings.runs_per_configuration
        results = []
        for run_index in range(runs):
            for config in all_configurations():
                results.append(
                    self.run_experiment(
                        approach=config["approach"],
                        selection=config["selection_strategy"],
                        scenario=scenario,
                        random_seed=self.settings.random_seed + run_index,
                        max_turns=max_turns,
                    )
                )
        return results
