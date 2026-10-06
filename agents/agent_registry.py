"""Authoritative in-memory registry for active agents."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from agents.agent import Agent


class DuplicateAgentError(ValueError):
    pass


class AgentNotFoundError(KeyError):
    pass


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: OrderedDict[str, Agent] = OrderedDict()

    @staticmethod
    def _key(value: str) -> str:
        return " ".join(value.casefold().split())

    def register(self, agent: Agent, *, allow_duplicate: bool = False) -> Agent:
        if not allow_duplicate:
            if any(self._key(existing.name) == self._key(agent.name) for existing in self._agents.values()):
                raise DuplicateAgentError(f"Agent named {agent.name!r} is already registered")
            if any(self._key(existing.role) == self._key(agent.role) for existing in self._agents.values()):
                raise DuplicateAgentError(f"Agent role {agent.role!r} is already registered")
            if agent.agent_id in self._agents:
                raise DuplicateAgentError(f"Agent id {agent.agent_id!r} is already registered")
        self._agents[agent.agent_id] = agent
        return agent

    def get(self, identifier: str) -> Agent:
        if identifier in self._agents:
            return self._agents[identifier]
        key = self._key(identifier)
        for agent in self._agents.values():
            if self._key(agent.name) == key:
                return agent
        raise AgentNotFoundError(identifier)

    def has(self, identifier: str) -> bool:
        try:
            self.get(identifier)
            return True
        except AgentNotFoundError:
            return False

    def list_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def remove(self, identifier: str) -> Agent:
        agent = self.get(identifier)
        return self._agents.pop(agent.agent_id)

    def __len__(self) -> int:
        return len(self._agents)

    def export_state(self) -> list[dict[str, Any]]:
        return [agent.to_dict(include_runtime=True) for agent in self._agents.values()]
