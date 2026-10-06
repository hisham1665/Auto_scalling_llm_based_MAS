"""Provider-neutral LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

from llm.response_parser import parse_structured_response

T = TypeVar("T")


@dataclass(slots=True)
class LLMResponse:
    content: str
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ns: int | None = None
    raw: dict[str, Any] | None = None

    def token_info(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class LLMError(RuntimeError):
    """Base error for model-provider failures."""


class BaseLLM(ABC):
    """Minimal interface needed by agents, managers, and experiments."""

    model: str

    def __init__(self) -> None:
        self.last_structured_retries = 0

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        correction_message: str | None = None,
        retries: int = 1,
    ) -> T:
        """Generate and validate a Pydantic response, retrying once if needed."""
        self.last_structured_retries = 0
        response = self.generate(messages, temperature=temperature, max_tokens=max_tokens)
        try:
            return parse_structured_response(response.content, schema)
        except Exception as first_error:
            if retries <= 0:
                raise
            self.last_structured_retries = 1
            correction = correction_message or (
                "Return only valid JSON matching the requested schema. "
                f"Correct the previous response. Validation error: {first_error}"
            )
            retry_messages = [*messages, {"role": "user", "content": correction}]
            retry_response = self.generate(retry_messages, temperature=0.0, max_tokens=max_tokens)
            return parse_structured_response(retry_response.content, schema)

    def validate_availability(self) -> None:
        """Provider-specific startup validation."""
