"""Initial Automatic Agent Generation (IAAG)."""

from __future__ import annotations

from agents.agent import Agent
from config.settings import Settings
from llm.base_llm import BaseLLM
from manager.conversation_manager import ConversationManager, RunResult


class IAAG:
    name = "iaag"

    def run(
        self,
        task: str,
        agents: list[Agent],
        llm: BaseLLM,
        *,
        settings: Settings,
        selection_strategy: str,
    ) -> RunResult:
        manager = ConversationManager(llm, settings=settings)
        return manager.run(task, agents, approach=self.name, selection_strategy=selection_strategy)
