"""Small optional API over the same research engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as error:  # pragma: no cover - exercised only without optional dependency
    FastAPI = None  # type: ignore[assignment]
    _IMPORT_ERROR = error

if FastAPI is not None:
    from config.settings import Settings
    from experiments.experiment_runner import ExperimentRunner

    app = FastAPI(title="IAAG/DRTAG Research Engine")
    runner = ExperimentRunner(Settings.from_env())

    class RunRequest(BaseModel):
        approach: str = Field(pattern="^(static|iaag|drtag)$")
        selection: str = Field(pattern="^(llm|round_robin|random)$")
        scenario: str = "medical"
        max_turns: int | None = None
        random_seed: int | None = None

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run")
    def run(request: RunRequest) -> dict[str, Any]:
        return runner.run_experiment(
            approach=request.approach,
            selection=request.selection,
            scenario=request.scenario,
            max_turns=request.max_turns,
            random_seed=request.random_seed,
        )

    @app.get("/experiments")
    def experiments() -> list[dict[str, Any]]:
        root = Path(runner.result_dir)
        return [json.loads(path.read_text(encoding="utf-8")) for path in root.rglob("*.json")]

    @app.get("/experiments/{experiment_id}")
    def experiment(experiment_id: str) -> dict[str, Any]:
        matches = list(Path(runner.result_dir).rglob(f"{experiment_id}.json"))
        if not matches:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return json.loads(matches[0].read_text(encoding="utf-8"))

    @app.get("/agents")
    def agents() -> dict[str, Any]:
        return {"message": "Agents are run-scoped; inspect agents in experiment results."}
else:
    def _missing_fastapi() -> None:
        raise RuntimeError(f"Install optional API dependencies to use api.main: {_IMPORT_ERROR}")

    app = None
