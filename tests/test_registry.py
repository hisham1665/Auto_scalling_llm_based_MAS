import pytest

from agents.agent import Agent
from agents.agent_registry import AgentRegistry, DuplicateAgentError


def _agent(name: str, role: str) -> Agent:
    return Agent(name=name, role=role, system_prompt="A focused role prompt.", model="mock")


def test_registry_is_authoritative_and_rejects_duplicate_name_or_role() -> None:
    registry = AgentRegistry()
    doctor = registry.register(_agent("Doctor", "physician"))
    assert registry.get(doctor.agent_id) is doctor
    assert registry.get("doctor") is doctor
    with pytest.raises(DuplicateAgentError):
        registry.register(_agent("DOCTOR", "other role"))
    with pytest.raises(DuplicateAgentError):
        registry.register(_agent("Other", "physician"))
    assert len(registry.list_agents()) == 1
