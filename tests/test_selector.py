import random

from agents.agent import Agent
from agents.agent_registry import AgentRegistry
from config.settings import Settings
from conversation.memory import GlobalConversation
from manager.selector import AgentSelector
from tests.conftest import MockLLM


def test_round_robin_and_random_are_reproducible() -> None:
    llm = MockLLM()
    registry = AgentRegistry()
    for name, role in (("Doctor", "doctor"), ("Nurse", "nurse"), ("Surgeon", "surgeon")):
        registry.register(Agent(name=name, role=role, system_prompt="A role prompt.", model="mock", _llm=llm))
    conversation = GlobalConversation()
    round_robin = AgentSelector("round_robin", llm, Settings(max_turns=2), rng=random.Random(2))
    assert [round_robin.select("task", registry, conversation) for _ in range(4)] == ["Doctor", "Nurse", "Surgeon", "Doctor"]
    first = AgentSelector("random", llm, Settings(max_turns=2), rng=random.Random(42))
    second = AgentSelector("random", llm, Settings(max_turns=2), rng=random.Random(42))
    assert [first.select("task", registry, conversation) for _ in range(5)] == [second.select("task", registry, conversation) for _ in range(5)]


def test_llm_selection_is_validated_against_registry() -> None:
    llm = MockLLM()
    registry = AgentRegistry()
    registry.register(Agent(name="Doctor", role="doctor", system_prompt="A role prompt.", model="mock", _llm=llm))
    assert AgentSelector("llm", llm, Settings(max_turns=2)).select("task", registry, GlobalConversation()) == "Doctor"
