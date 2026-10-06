"""Environment-backed settings for the local research runner."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_dotenv(path: str = ".env") -> None:
    """Load a minimal .env file without requiring python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    """Runtime settings.

    Defaults are intentionally conservative for a laptop running a single
    Ollama model.  All model calls are serialized by the experiment runner.
    """

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "spark-x2.5-4b"
    ollama_timeout: float = 120.0
    ollama_retries: int = 1
    temperature: float = 0.2
    max_response_tokens: int = 384
    max_turns: int = 10
    max_agents: int = 8
    max_agent_generations_per_run: int = 4
    context_max_chars: int = 12000
    random_seed: int = 42
    runs_per_configuration: int = 1
    few_shot_examples: int = 2
    results_dir: str = "results"
    prompts_dir: str = "prompts"
    evaluation_model: str = "distilbert-base-uncased"
    enable_bertscore: bool = False
    log_level: str = "INFO"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, dotenv_path: str = ".env") -> "Settings":
        _load_dotenv(dotenv_path)
        defaults = cls()
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", defaults.ollama_base_url),
            ollama_model=os.getenv("OLLAMA_MODEL", defaults.ollama_model),
            ollama_timeout=_float("OLLAMA_TIMEOUT", defaults.ollama_timeout),
            ollama_retries=_int("OLLAMA_RETRIES", defaults.ollama_retries),
            temperature=_float("LLM_TEMPERATURE", defaults.temperature),
            max_response_tokens=_int("MAX_RESPONSE_TOKENS", defaults.max_response_tokens),
            max_turns=_int("MAX_TURNS", defaults.max_turns),
            max_agents=_int("MAX_AGENTS", defaults.max_agents),
            max_agent_generations_per_run=_int(
                "MAX_AGENT_GENERATIONS_PER_RUN", defaults.max_agent_generations_per_run
            ),
            context_max_chars=_int("CONTEXT_MAX_CHARS", defaults.context_max_chars),
            random_seed=_int("RANDOM_SEED", defaults.random_seed),
            runs_per_configuration=_int("RUNS_PER_CONFIGURATION", defaults.runs_per_configuration),
            few_shot_examples=_int("FEW_SHOT_EXAMPLES", defaults.few_shot_examples),
            results_dir=os.getenv("RESULTS_DIR", defaults.results_dir),
            prompts_dir=os.getenv("PROMPTS_DIR", defaults.prompts_dir),
            evaluation_model=os.getenv("EVALUATION_MODEL", defaults.evaluation_model),
            enable_bertscore=_bool("ENABLE_BERTSCORE", defaults.enable_bertscore),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "ollama_timeout": self.ollama_timeout,
            "ollama_retries": self.ollama_retries,
            "temperature": self.temperature,
            "max_response_tokens": self.max_response_tokens,
            "max_turns": self.max_turns,
            "max_agents": self.max_agents,
            "max_agent_generations_per_run": self.max_agent_generations_per_run,
            "context_max_chars": self.context_max_chars,
            "random_seed": self.random_seed,
            "runs_per_configuration": self.runs_per_configuration,
            "few_shot_examples": self.few_shot_examples,
            "results_dir": self.results_dir,
            "prompts_dir": self.prompts_dir,
            "evaluation_model": self.evaluation_model,
            "enable_bertscore": self.enable_bertscore,
            "log_level": self.log_level,
        }
