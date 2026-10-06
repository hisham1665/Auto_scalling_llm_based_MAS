"""Agent-selection strategies from the paper."""

from __future__ import annotations

import logging
import random
from enum import Enum

from agents.agent_registry import AgentRegistry
from config.settings import Settings
from conversation.context_manager import ContextManager
from conversation.memory import GlobalConversation
from llm.base_llm import BaseLLM
from manager.agent_generator import SelectionDecision, PromptLibrary

logger = logging.getLogger(__name__)


class SelectionStrategy(str, Enum):
    LLM = "llm"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


class AgentSelector:
    def __init__(
        self,
        strategy: SelectionStrategy | str,
        llm: BaseLLM,
        settings: Settings | None = None,
        context_manager: ContextManager | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.strategy = SelectionStrategy(strategy)
        self.llm = llm
        self.settings = settings or Settings.from_env()
        self.context_manager = context_manager or ContextManager(self.settings.context_max_chars)
        self.rng = rng or random.Random(self.settings.random_seed)
        self._round_robin_cursor = 0
        self.prompts = PromptLibrary(self.settings.prompts_dir)

    def _llm_select(self, task: str, registry: AgentRegistry, conversation: GlobalConversation) -> str:
        fallback = """You select the next agent in a multi-agent conversation.
Choose only from the available agents and return only JSON with selected_agent and a short reason.
"""
        template = self.prompts.get("agent_selector.txt", fallback)
        prompt = (
            f"{template}\n\nTASK:\n{task}\n\nAVAILABLE AGENTS:\n"
            f"{self.context_manager.agents_snapshot(registry)}\n\nRECENT CONVERSATION:\n"
            f"{self.context_manager.conversation_snapshot(conversation)}\n\n"
            'Return {"selected_agent":"AgentName","reason":"short reason"}. '
            "Never invent an unavailable name."
        )
        try:
            decision = self.llm.generate_structured(
                [{"role": "system", "content": prompt}],
                SelectionDecision,
                temperature=0.0,
                max_tokens=192,
                retries=1,
            )
            return registry.get(decision.selected_agent).name
        except Exception as error:
            logger.warning("LLM selection failed; using deterministic fallback: %s", error)
            return registry.list_agents()[self._round_robin_cursor % len(registry)].name

    def select(self, task: str, registry: AgentRegistry, conversation: GlobalConversation) -> str:
        agents = registry.list_agents()
        if not agents:
            raise ValueError("Cannot select an agent from an empty registry")
        if self.strategy is SelectionStrategy.LLM:
            selected = self._llm_select(task, registry, conversation)
        elif self.strategy is SelectionStrategy.ROUND_ROBIN:
            selected = agents[self._round_robin_cursor % len(agents)].name
        else:
            selected = self.rng.choice(agents).name
        self._round_robin_cursor += 1
        return selected
