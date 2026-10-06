from agents.agent import Agent
from config.settings import Settings
from manager.conversation_manager import ConversationManager
from tests.conftest import MockLLM


def test_drtag_dynamic_agent_is_created_then_participates() -> None:
    llm = MockLLM()
    result = ConversationManager(llm, settings=Settings(max_turns=4, max_agents=4)).run(
        "A task requiring a missing specialist",
        [Agent("Doctor", "physician", "Assess the task.", "mock", _llm=llm), Agent("Nurse", "nurse", "Observe the task.", "mock", _llm=llm)],
        approach="drtag",
        selection_strategy="llm",
    )
    events = result.events
    created = next(i for i, event in enumerate(events) if event["type"] == "AGENT_CREATED")
    registered = next(i for i, event in enumerate(events) if event["type"] == "AGENT_REGISTERED" and event["agent"] == "Surgeon")
    selected = next(i for i, event in enumerate(events) if event["type"] == "AGENT_SELECTED" and event["agent"] == "Surgeon")
    response = next(i for i, event in enumerate(events) if event["type"] == "RESPONSE_RECEIVED" and event["agent"] == "Surgeon")
    assert created < registered < selected < response
    assert "Surgeon" in [item["agent"] for item in result.conversation if item["role"] == "agent"]
    assert result.termination_reason.startswith("The required")
