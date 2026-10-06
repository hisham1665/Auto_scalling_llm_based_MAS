"""Central orchestration loop for static, IAAG, and DRTAG runs."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.agent import Agent
from agents.agent_factory import AgentFactory
from agents.agent_registry import AgentRegistry, DuplicateAgentError
from config.settings import Settings
from conversation.context_manager import ContextManager
from conversation.memory import GlobalConversation
from llm.base_llm import BaseLLM
from manager.agent_generator import AgentGenerator, TerminationDecision
from manager.selector import AgentSelector, SelectionStrategy

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunResult:
    approach: str
    selection_strategy: str
    task: str
    initial_agents: list[dict[str, Any]]
    generated_agents: list[dict[str, Any]]
    final_agents: list[dict[str, Any]]
    conversation: list[dict[str, Any]]
    events: list[dict[str, Any]]
    turn_count: int
    termination_reason: str
    duration_seconds: float
    token_usage: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "approach": self.approach,
            "selection_strategy": self.selection_strategy,
            "task": self.task,
            "initial_agents": self.initial_agents,
            "generated_agents": self.generated_agents,
            "final_agents": self.final_agents,
            "conversation": self.conversation,
            "events": self.events,
            "turn_count": self.turn_count,
            "termination_reason": self.termination_reason,
            "duration_seconds": self.duration_seconds,
            "token_usage": self.token_usage,
            "errors": self.errors,
            "retries": self.retries,
        }


class ConversationManager:
    def __init__(
        self,
        llm: BaseLLM,
        *,
        settings: Settings | None = None,
        registry: AgentRegistry | None = None,
        conversation: GlobalConversation | None = None,
        generator: AgentGenerator | None = None,
        factory: AgentFactory | None = None,
        context_manager: ContextManager | None = None,
        selector: AgentSelector | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or Settings.from_env()
        self.registry = registry or AgentRegistry()
        self.conversation = conversation or GlobalConversation()
        self.context_manager = context_manager or ContextManager(self.settings.context_max_chars)
        self.generator = generator or AgentGenerator(llm, self.settings, self.context_manager)
        self.factory = factory or AgentFactory(llm, llm.model)
        self.selector = selector
        self._events: list[dict[str, Any]] = []
        self._run_started = time.monotonic()
        self._errors: list[str] = []
        self._retries = 0
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._generation_count = 0

    def _event(self, event_type: str, **details: Any) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - self._run_started, 4),
            "type": event_type,
            **details,
        }
        self._events.append(event)
        logger.info("%s %s", event_type, details)

    def _register_initial(self, agents: list[Agent]) -> None:
        if not agents:
            raise ValueError("At least one seed or user-defined agent is required")
        if len(agents) > self.settings.max_agents:
            raise RuntimeError("Initial agent count exceeds MAX_AGENTS")
        for agent in agents:
            self.registry.register(agent)
        self._event("INITIAL_AGENTS_REGISTERED", agents=[a.name for a in agents])

    def _generate_agent(self, task: str, context: str, *, phase: str) -> Agent | None:
        if len(self.registry) >= self.settings.max_agents:
            self._event("GENERATION_BLOCKED", reason="max_agents_reached")
            return None
        if self._generation_count >= self.settings.max_agent_generations_per_run:
            self._event("GENERATION_BLOCKED", reason="max_agent_generations_reached")
            return None
        self._generation_count += 1
        spec = self.generator.generate(task, self.registry, self.conversation, generation_context=context)
        self._retries += int(getattr(self.llm, "last_structured_retries", 0))
        self._event("MANAGER_ANALYSIS", phase=phase, create_agent=spec.create_agent, reason=spec.reason)
        if not spec.create_agent:
            return None
        try:
            agent = self.factory.create(spec, self.registry, created_dynamically=True)
            self.registry.register(agent)
            self._event(
                "AGENT_CREATED",
                agent=agent.name,
                agent_id=agent.agent_id,
                role=agent.role,
                generation_reason=agent.generation_reason,
                phase=phase,
            )
            self._event("AGENT_REGISTERED", agent=agent.name, agent_id=agent.agent_id)
            return agent
        except (ValueError, DuplicateAgentError) as error:
            self._errors.append(str(error))
            self._event("GENERATION_FAILURE", error=str(error))
            logger.warning("Generated agent rejected: %s", error)
            return None

    def _should_end(self, task: str) -> TerminationDecision:
        fallback = TerminationDecision(end_conversation=False, reason="continue until a guard is reached")
        termination_template = self.generator.prompts.get(
            "termination.txt",
            "Based on the task and current conversation, decide whether additional discussion is necessary.",
        )
        prompt = (
            f"{termination_template}\n\nYou are a concise conversation termination reviewer. "
            "Return only JSON with "
            "end_conversation (boolean) and a short reason.\n\n"
            f"TASK:\n{task}\n\nCONVERSATION:\n"
            f"{self.context_manager.conversation_snapshot(self.conversation)}"
        )
        try:
            return self.llm.generate_structured(
                [{"role": "system", "content": prompt}],
                TerminationDecision,
                temperature=0.0,
                max_tokens=160,
                retries=1,
            )
        except Exception as error:
            logger.warning("Termination decision failed; continuing safely: %s", error)
            self._errors.append(f"termination decision: {error}")
            return fallback

    def run(self, task: str, initial_agents: list[Agent], *, approach: str, selection_strategy: str) -> RunResult:
        self._run_started = time.monotonic()
        self._generation_count = 0
        self._events = []
        self._errors = []
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._retries = 0
        self._event("TASK_RECEIVED", task=task)
        self.conversation.add_user(task)
        self._register_initial(initial_agents)
        initial_snapshot = [agent.to_dict(include_runtime=True) for agent in initial_agents]
        self.selector = self.selector or AgentSelector(
            selection_strategy, self.llm, self.settings, self.context_manager
        )
        generated: list[dict[str, Any]] = []
        normalized_approach = approach.lower()
        if normalized_approach == "iaag":
            while len(self.registry) < self.settings.max_agents and self._generation_count < self.settings.max_agent_generations_per_run:
                agent = self._generate_agent(
                    task,
                    "IAAG initial analysis: identify expertise required before turn one.",
                    phase="initial",
                )
                if not agent:
                    break
                generated.append(agent.to_dict(include_runtime=True))
        elif normalized_approach not in {"static", "drtag"}:
            raise ValueError(f"Unknown approach: {approach}")

        termination_reason = "max_turns_reached"
        pending_dynamic_agent: Agent | None = None
        for turn in range(1, self.settings.max_turns + 1):
            if not self.registry.list_agents():
                termination_reason = "no_active_agents"
                break
            if pending_dynamic_agent is not None:
                selected = pending_dynamic_agent
                pending_dynamic_agent = None
            else:
                selected_name = self.selector.select(task, self.registry, self.conversation)
                self._retries += int(getattr(self.llm, "last_structured_retries", 0))
                selected = self.registry.get(selected_name)
            self._event("AGENT_SELECTED", agent=selected.name, agent_id=selected.agent_id, turn=turn)
            try:
                response = selected.respond(
                    task,
                    self.context_manager.conversation_snapshot(self.conversation),
                )
                if response.prompt_tokens:
                    self._token_usage["prompt_tokens"] += response.prompt_tokens
                if response.completion_tokens:
                    self._token_usage["completion_tokens"] += response.completion_tokens
                content = response.content.strip()
            except Exception as error:
                self._errors.append(f"agent {selected.name}: {error}")
                self._event("AGENT_RESPONSE_FAILURE", agent=selected.name, error=str(error), turn=turn)
                content = f"{selected.name} could not provide a response because the model call failed."
            self.conversation.add_agent(selected.name, selected.agent_id, content, turn)
            self._event("RESPONSE_RECEIVED", agent=selected.name, turn=turn)
            dynamically_added = False
            if normalized_approach == "drtag" and turn < self.settings.max_turns:
                agent = self._generate_agent(
                    task,
                    "DRTAG real-time review after the preceding interaction; create only genuinely missing expertise.",
                    phase="dynamic",
                )
                if agent:
                    generated.append(agent.to_dict(include_runtime=True))
                    pending_dynamic_agent = agent
                    dynamically_added = True
            if dynamically_added:
                self._event("DYNAMIC_AGENT_PARTICIPATION_GUARD", reason="new agent must receive a turn")
                continue
            decision = self._should_end(task)
            self._retries += int(getattr(self.llm, "last_structured_retries", 0))
            if decision.end_conversation:
                termination_reason = decision.reason or "manager_end_conversation"
                self._event("TERMINATION", reason=termination_reason, turn=turn)
                break
        else:
            self._event("TERMINATION", reason=termination_reason, turn=self.settings.max_turns)
        self._event("EXPERIMENT_COMPLETE", turns=len([m for m in self.conversation.all() if m.role == "agent"]))
        return RunResult(
            approach=normalized_approach,
            selection_strategy=SelectionStrategy(selection_strategy).value,
            task=task,
            initial_agents=initial_snapshot,
            generated_agents=generated,
            final_agents=self.registry.export_state(),
            conversation=self.conversation.to_dict(),
            events=list(self._events),
            turn_count=len([m for m in self.conversation.all() if m.role == "agent"]),
            termination_reason=termination_reason,
            duration_seconds=round(time.monotonic() - self._run_started, 4),
            token_usage=self._token_usage,
            errors=self._errors,
            retries=self._retries,
        )
