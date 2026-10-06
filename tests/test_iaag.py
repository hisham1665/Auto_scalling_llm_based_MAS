from agents.agent import Agent
from config.settings import Settings
from manager.conversation_manager import ConversationManager
from tests.conftest import MockLLM


def test_iaag_generates_before_first_agent_turn() -> None:
    llm = MockLLM()
    result = ConversationManager(llm, settings=Settings(max_turns=3, max_agents=4)).run(
        "A task needing multiple perspectives",
        [Agent("Doctor", "physician", "Assess the task.", "mock", _llm=llm), Agent("Nurse", "nurse", "Observe the task.", "mock", _llm=llm)],
        approach="iaag",
        selection_strategy="llm",
    )
    first_response_index = next(i for i, item in enumerate(result.events) if item["type"] == "RESPONSE_RECEIVED")
    created_index = next(i for i, item in enumerate(result.events) if item["type"] == "AGENT_CREATED")
    assert created_index < first_response_index
    assert result.generated_agents[0]["name"] == "Surgeon"
