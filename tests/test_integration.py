import json

from config.settings import Settings
from experiments.experiment_runner import ExperimentRunner
from tests.conftest import MockLLM


def test_drtag_integration_saves_dynamic_participation(tmp_path) -> None:
    runner = ExperimentRunner(
        Settings(max_turns=4, max_agents=4, results_dir=str(tmp_path)),
        llm=MockLLM(),
        result_dir=tmp_path,
    )
    payload = runner.run_experiment(approach="drtag", selection="llm", scenario="medical")
    assert payload["status"] == "completed"
    assert payload["result_path"]
    saved = json.loads(open(payload["result_path"], encoding="utf-8").read())
    assert any(agent["name"] == "Surgeon" for agent in saved["generated_agents"])
    assert any(item.get("agent") == "Surgeon" for item in saved["conversation"] if item["role"] == "agent")
    assert any(event["type"] == "AGENT_CREATED" for event in saved["events"])
