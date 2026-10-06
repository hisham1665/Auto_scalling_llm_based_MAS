"""Executable agent object."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agents.base_agent import BaseAgent
from llm.base_llm import BaseLLM, LLMResponse


STANDARD_AGENT_INSTRUCTIONS = """You are the {role} agent.

Your responsibility is {role}.
Your expertise should complement the other agents.
Use the conversation context provided.
Do not impersonate other agents.
Do not unnecessarily repeat previous information.
Provide concise, task-relevant contributions.
If uncertainty exists, state it clearly.
"""


@dataclass(slots=True)
class Agent(BaseAgent):
    name: str
    role: str
    system_prompt: str
    model: str
    agent_id: str = field(default_factory=lambda: f"agent_{uuid4().hex[:8]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_dynamically: bool = False
    generation_reason: str | None = None
    _llm: BaseLLM | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.role = self.role.strip()
        self.system_prompt = self.system_prompt.strip()
        if not self.name or not self.role or not self.system_prompt:
            raise ValueError("Agent name, role, and system_prompt must be non-empty")

    def respond(self, task: str, context: str) -> LLMResponse:
        if self._llm is None:
            raise RuntimeError(f"Agent {self.name} has no LLM provider")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"TASK:\n{task}\n\nCONVERSATION CONTEXT:\n{context}\n\n"
                    f"As {self.name}, provide one concise contribution."
                ),
            },
        ]
        return self._llm.generate(messages)

    def to_dict(self, include_runtime: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_llm", None)
        if not include_runtime:
            data.pop("model", None)
        return data
