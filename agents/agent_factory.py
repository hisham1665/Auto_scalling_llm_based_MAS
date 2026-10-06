"""Validated construction of executable agents."""

from __future__ import annotations

import re

from agents.agent import STANDARD_AGENT_INSTRUCTIONS, Agent
from agents.agent_registry import AgentRegistry
from llm.base_llm import BaseLLM
from manager.agent_generator import AgentSpec


class AgentFactory:
    def __init__(self, llm: BaseLLM, model_name: str | None = None) -> None:
        self.llm = llm
        self.model_name = model_name or llm.model

    def create(
        self,
        spec: AgentSpec,
        registry: AgentRegistry,
        *,
        created_dynamically: bool,
    ) -> Agent:
        name = spec.name.strip()
        role = spec.role.strip()
        prompt = spec.system_prompt.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{1,63}", name):
            raise ValueError("Agent name must be 2-64 characters and contain no JSON/control syntax")
        if len(prompt) < 12 or len(prompt) > 4000:
            raise ValueError("Agent system prompt must contain 12-4000 characters")
        if registry.has(name) or any(agent.role.casefold() == role.casefold() for agent in registry.list_agents()):
            raise ValueError(f"Duplicate agent or role: {name}")
        full_prompt = STANDARD_AGENT_INSTRUCTIONS.format(role=role) + "\nRole-specific instructions:\n" + prompt
        return Agent(
            name=name,
            role=role,
            system_prompt=full_prompt,
            model=self.model_name,
            created_dynamically=created_dynamically,
            generation_reason=spec.reason,
            _llm=self.llm,
        )
