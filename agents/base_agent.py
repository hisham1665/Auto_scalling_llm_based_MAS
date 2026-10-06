"""Abstract agent behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm.base_llm import LLMResponse


class BaseAgent(ABC):
    @abstractmethod
    def respond(self, task: str, context: str) -> LLMResponse:
        raise NotImplementedError
