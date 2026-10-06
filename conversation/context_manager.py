"""Bounded context snapshots for local-model inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.agent_registry import AgentRegistry
    from conversation.memory import GlobalConversation


class ContextManager:
    def __init__(self, max_chars: int = 12000, recent_messages: int = 12) -> None:
        self.max_chars = max_chars
        self.recent_messages = recent_messages

    def conversation_snapshot(self, conversation: "GlobalConversation") -> str:
        text = conversation.format(conversation.recent(self.recent_messages))
        if len(text) <= self.max_chars:
            return text or "(no conversation yet)"
        return "[earlier context truncated]\n" + text[-self.max_chars :]

    def agents_snapshot(self, registry: "AgentRegistry") -> str:
        agents = registry.list_agents()
        if not agents:
            return "(no active agents)"
        return "\n".join(
            f"- {agent.name} | role: {agent.role} | prompt: {agent.system_prompt}"
            for agent in agents
        )

    def build(self, task: str, registry: "AgentRegistry", conversation: "GlobalConversation") -> str:
        return (
            f"TASK:\n{task}\n\nAVAILABLE AGENTS:\n{self.agents_snapshot(registry)}\n\n"
            f"RECENT CONVERSATION:\n{self.conversation_snapshot(conversation)}"
        )
