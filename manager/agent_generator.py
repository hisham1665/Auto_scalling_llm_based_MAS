"""IAAG/DRTAG agent-generation prompt and validation logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config.settings import Settings
from conversation.context_manager import ContextManager
from conversation.memory import GlobalConversation
from llm.base_llm import BaseLLM

if TYPE_CHECKING:
    from agents.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    create_agent: bool = False
    name: str = ""
    role: str = ""
    system_prompt: str = ""
    reason: str = ""

    @field_validator("name", "role", "system_prompt", "reason", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        return str(value or "").strip()


class SelectionDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    selected_agent: str
    reason: str = ""


class TerminationDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    end_conversation: bool
    reason: str = ""


def _model_dump(value: BaseModel) -> dict[str, Any]:
    return value.model_dump() if hasattr(value, "model_dump") else value.dict()


class PromptLibrary:
    def __init__(self, directory: str = "prompts") -> None:
        self.directory = Path(directory)

    def get(self, filename: str, fallback: str) -> str:
        path = self.directory / filename
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return fallback


class AgentGenerator:
    def __init__(
        self,
        llm: BaseLLM,
        settings: Settings | None = None,
        context_manager: ContextManager | None = None,
        prompts: PromptLibrary | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or Settings.from_env()
        self.context_manager = context_manager or ContextManager(self.settings.context_max_chars)
        self.prompts = prompts or PromptLibrary(self.settings.prompts_dir)

    def _prompt(
        self,
        task: str,
        registry: "AgentRegistry",
        conversation: GlobalConversation,
        generation_context: str,
    ) -> str:
        fallback = """You are an expert agent designer and the Conversation Manager.
Analyze the task, available agents, and recent conversation in a concise controlled sequence:
1. Analyze the task. 2. Review agents. 3. Review conversation.
4. Identify missing expertise. 5. Decide whether a new specialist is required.
6. If required, choose a distinct persona and executable system prompt.
Do not expose hidden chain-of-thought. Return only JSON.
        """
        template = self.prompts.get("agent_generator.txt", fallback)
        manager_template = self.prompts.get("conversation_manager.txt", "")
        examples = self.prompts.get("few_shot_examples.txt", "")
        example_lines = examples.split("\n\n")[: max(0, self.settings.few_shot_examples)]
        few_shot = "\n\n".join(example_lines)
        return (
            f"{manager_template}\n\n{template}\n\nFEW-SHOT EXAMPLES:\n{few_shot}\n\n"
            f"CURRENT TASK:\n{task}\n\nAVAILABLE AGENTS:\n"
            f"{self.context_manager.agents_snapshot(registry)}\n\n"
            f"RECENT CONVERSATION:\n{self.context_manager.conversation_snapshot(conversation)}\n\n"
            f"GENERATION CONTEXT:\n{generation_context}\n\n"
            "Return exactly this JSON shape: {\"create_agent\": false} or "
            "{\"create_agent\": true, \"name\": \"...\", \"role\": \"...\", "
            "\"system_prompt\": \"...\", \"reason\": \"...\"}."
        )

    def generate(
        self,
        task: str,
        registry: "AgentRegistry",
        conversation: GlobalConversation,
        *,
        generation_context: str,
    ) -> AgentSpec:
        prompt = self._prompt(task, registry, conversation, generation_context)
        messages = [{"role": "system", "content": prompt}]
        correction = self.prompts.get(
            "correction.txt",
            "Previous output was invalid. Return only a JSON object matching the requested schema.",
        )
        try:
            result = self.llm.generate_structured(
                messages,
                AgentSpec,
                temperature=0.0,
                max_tokens=self.settings.max_response_tokens,
                correction_message=correction,
                retries=1,
            )
            if result.create_agent and (not result.name or not result.role or not result.system_prompt):
                raise ValueError("create_agent=true requires name, role, and system_prompt")
            return result
        except Exception as error:
            logger.warning("Agent generation failed; continuing with current registry: %s", error)
            return AgentSpec(create_agent=False, reason=f"generation failure: {error}")

    @staticmethod
    def as_dict(spec: AgentSpec) -> dict[str, Any]:
        return _model_dump(spec)
