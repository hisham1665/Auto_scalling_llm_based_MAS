"""Global conversation memory."""

from __future__ import annotations

from typing import Any, Iterable

from conversation.message import Message


class GlobalConversation:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> Message:
        self._messages.append(message)
        return message

    def add_user(self, content: str, *, metadata: dict[str, Any] | None = None) -> Message:
        return self.add(Message(role="user", content=content, message_type="task", metadata=metadata or {}))

    def add_agent(
        self,
        agent_name: str,
        agent_id: str,
        content: str,
        turn_number: int,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        return self.add(
            Message(
                role="agent",
                agent=agent_name,
                agent_id=agent_id,
                content=content,
                turn_number=turn_number,
                metadata=metadata or {},
            )
        )

    def recent(self, limit: int = 8) -> list[Message]:
        return self._messages[-max(0, limit) :]

    def all(self) -> list[Message]:
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def to_dict(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self._messages]

    def format(self, messages: Iterable[Message] | None = None) -> str:
        selected = self._messages if messages is None else messages
        lines = []
        for message in selected:
            speaker = message.agent or message.role
            lines.append(f"[{speaker}, turn {message.turn_number}] {message.content}")
        return "\n".join(lines)
