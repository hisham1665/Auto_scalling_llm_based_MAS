from __future__ import annotations

from typing import Any

from llm.base_llm import BaseLLM, LLMResponse
from manager.agent_generator import AgentSpec, SelectionDecision, TerminationDecision


class MockLLM(BaseLLM):
    """Deterministic provider used only in tests; runtime remains Ollama-only."""

    def __init__(self, *, create_on_generation: bool = True) -> None:
        self.model = "mock-model"
        self.create_on_generation = create_on_generation
        self.generation_calls = 0
        self.structured_calls: list[str] = []
        self.responses: list[tuple[str, str]] = []

    def generate(self, messages: list[dict[str, str]], **_: Any) -> LLMResponse:
        user_text = messages[-1].get("content", "")
        if "As Surgeon" in user_text:
            content = "The specialist contribution addresses the missing surgical expertise and clarifies the next review step."
            speaker = "Surgeon"
        elif "As Nurse" in user_text:
            content = "Nursing observations support monitoring symptoms, escalation, and clear communication."
            speaker = "Nurse"
        else:
            content = "The general assessment organizes the task and identifies information still needed."
            speaker = "Doctor"
        self.responses.append((speaker, content))
        return LLMResponse(content=content, model=self.model, prompt_tokens=10, completion_tokens=8)

    def generate_structured(self, messages: list[dict[str, str]], schema: type[Any], **_: Any) -> Any:
        name = getattr(schema, "__name__", "")
        self.structured_calls.append(name)
        if name == "AgentSpec":
            self.generation_calls += 1
            prompt = messages[-1].get("content", "")
            should_create = (
                self.create_on_generation
                and ("IAAG initial" in prompt or "DRTAG" in prompt)
                and "Surgeon" not in prompt
            )
            if should_create:
                return AgentSpec(
                    create_agent=True,
                    name="Surgeon",
                    role="surgical specialist",
                    system_prompt="Review whether surgical expertise is relevant and state its limits.",
                    reason="The current discussion needs a distinct surgical perspective.",
                )
            return AgentSpec(create_agent=False, reason="Current agents cover the remaining discussion.")
        if name == "SelectionDecision":
            prompt = messages[-1].get("content", "")
            selected = "Surgeon" if "Surgeon" in prompt else "Doctor"
            return SelectionDecision(selected_agent=selected, reason="best current expertise")
        if name == "TerminationDecision":
            return TerminationDecision(end_conversation=True, reason="The required perspectives have been covered.")
        raise AssertionError(f"Unexpected schema {name}")
