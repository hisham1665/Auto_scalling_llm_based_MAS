from agents.agent import Agent


def test_agent_has_required_metadata_and_serializes_without_provider() -> None:
    agent = Agent(name="Doctor", role="physician", system_prompt="Assess the case carefully.", model="mock")
    data = agent.to_dict(include_runtime=True)
    assert data["name"] == "Doctor"
    assert data["created_dynamically"] is False
    assert "_llm" not in data
