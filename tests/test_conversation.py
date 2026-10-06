from agents.agent import Agent
from config.settings import Settings
from manager.conversation_manager import ConversationManager
from tests.conftest import MockLLM


def test_conversation_stops_on_manager_decision() -> None:
    llm = MockLLM(create_on_generation=False)
    settings = Settings(max_turns=5, max_agents=4, max_agent_generations_per_run=2)
    result = ConversationManager(llm, settings=settings).run(
        "Discuss a task", [Agent("Doctor", "physician", "Assess this task carefully.", "mock", _llm=llm)], approach="static", selection_strategy="llm"
    )
    assert result.turn_count == 1
    assert result.termination_reason.startswith("The required")
    assert any(event["type"] == "TERMINATION" for event in result.events)
