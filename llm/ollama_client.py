"""Ollama HTTP client used by the actual local runtime."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config.settings import Settings
from llm.base_llm import BaseLLM, LLMError, LLMResponse

logger = logging.getLogger(__name__)


class OllamaUnavailableError(LLMError):
    pass


class OllamaModelNotFoundError(LLMError):
    pass


class OllamaLLM(BaseLLM):
    """Serialized, non-streaming Ollama client suitable for a laptop."""

    def __init__(self, settings: Settings | None = None, **kwargs: Any) -> None:
        settings = settings or Settings.from_env()
        super().__init__()
        self.base_url = kwargs.get("base_url", settings.ollama_base_url).rstrip("/")
        self.model = kwargs.get("model", settings.ollama_model)
        self.timeout = float(kwargs.get("timeout", settings.ollama_timeout))
        self.retries = int(kwargs.get("retries", settings.ollama_retries))
        self.default_temperature = float(kwargs.get("temperature", settings.temperature))
        self.default_max_tokens = int(kwargs.get("max_tokens", settings.max_response_tokens))
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        attempts = max(0, self.retries) + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                last_error = error
                if attempt + 1 < attempts:
                    logger.warning("Ollama request failed; retrying (%s/%s)", attempt + 1, attempts - 1)
                    time.sleep(min(2**attempt, 4))
        raise OllamaUnavailableError(f"Ollama is not reachable at {self.base_url}: {last_error}") from last_error

    def validate_availability(self) -> None:
        try:
            payload = self._request("GET", "/api/tags").json()
        except OllamaUnavailableError:
            raise
        try:
            names = {str(item.get("name", "")) for item in payload.get("models", [])}
            base_model = self.model.split(":", 1)[0]
            if self.model not in names and base_model not in {name.split(":", 1)[0] for name in names}:
                raise OllamaModelNotFoundError(
                    f"Model {self.model!r} is not installed. Run: ollama pull {self.model}"
                )
        except ValueError as error:
            raise OllamaUnavailableError("Ollama returned invalid model metadata") from error

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.default_temperature if temperature is None else temperature,
                "num_predict": self.default_max_tokens if max_tokens is None else max_tokens,
            },
        }
        response = self._request("POST", "/api/chat", json=payload)
        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as error:
            raise LLMError(f"Ollama returned an invalid chat response: {error}") from error
        return LLMResponse(
            content=str(content),
            model=str(data.get("model", self.model)),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_duration_ns=data.get("total_duration"),
            raw=data,
        )
