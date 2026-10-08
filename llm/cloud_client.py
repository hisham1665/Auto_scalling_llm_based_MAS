"""OpenAI-compatible cloud clients with NVIDIA-first Groq failover."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config.settings import Settings
from llm.base_llm import BaseLLM, LLMError, LLMResponse

logger = logging.getLogger(__name__)


class CloudConfigurationError(LLMError):
    """Raised when neither configured cloud provider can be used."""


class CloudProviderError(LLMError):
    """Raised when a cloud provider rejects or cannot process a request."""


class OpenAICompatibleLLM(BaseLLM):
    """Synchronous client for NVIDIA NIM and other compatible chat APIs."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 90.0,
        retries: int = 0,
        temperature: float = 0.2,
        max_tokens: int = 384,
        reasoning_format: str | None = "hidden",
        session: requests.Session | None = None,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.default_temperature = float(temperature)
        self.default_max_tokens = int(max_tokens)
        self.reasoning_format = reasoning_format
        self.session = session or requests.Session()

    def _uses_nemotron_lightning_template(self) -> bool:
        model = self.model.lower().replace("_", "-")
        return "nemotron-3.5" in model

    def validate_availability(self) -> None:
        if not self.api_key:
            raise CloudConfigurationError(f"{self.provider} API key is not configured")
        if not self.model:
            raise CloudConfigurationError(f"{self.provider} model is not configured")

    def _error_detail(self, response: requests.Response) -> str:
        try:
            payload = response.json()
            detail = payload.get("error", payload) if isinstance(payload, dict) else payload
            if isinstance(detail, dict):
                detail = detail.get("message", detail.get("code", "request rejected"))
            text = str(detail)
        except (ValueError, TypeError):
            text = response.text
        text = text.replace(self.api_key, "[REDACTED]").strip()
        return text[:300] or "request rejected"

    def _request(self, payload: dict[str, Any]) -> requests.Response:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        attempts = self.retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if not response.ok:
                    raise CloudProviderError(
                        f"{self.provider} request failed with HTTP {response.status_code}: "
                        f"{self._error_detail(response)}"
                    )
                return response
            except CloudProviderError:
                raise
            except requests.RequestException as error:
                last_error = error
                if attempt + 1 < attempts:
                    logger.warning(
                        "%s request failed; retrying (%s/%s)",
                        self.provider,
                        attempt + 1,
                        self.retries,
                    )
                    time.sleep(min(2**attempt, 4))
        raise CloudProviderError(
            f"{self.provider} endpoint request failed: {last_error}"
        ) from last_error

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.validate_availability()
        request_messages = [dict(message) for message in messages]
        nvidia_hidden_reasoning = (
            self.provider.lower() == "nvidia" and self.reasoning_format == "hidden"
        )
        if nvidia_hidden_reasoning and not self._uses_nemotron_lightning_template():
            for message in request_messages:
                if message.get("role") == "system":
                    content = message.get("content", "")
                    if not content.startswith("/no_think"):
                        message["content"] = f"/no_think\n{content}"
                    break
        payload = {
            "model": self.model,
            "messages": request_messages,
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if nvidia_hidden_reasoning and self._uses_nemotron_lightning_template():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if self.provider.lower() == "groq" and self.reasoning_format:
            payload["reasoning_format"] = self.reasoning_format
        response = self._request(payload)
        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise CloudProviderError(
                f"{self.provider} returned an invalid chat response: {error}"
            ) from error
        if content is None or not str(content).strip():
            raise CloudProviderError(f"{self.provider} returned an empty chat response")
        return LLMResponse(
            content=str(content),
            model=str(data.get("model", self.model)),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw=data,
        )


class CloudFailoverLLM(BaseLLM):
    """Use NVIDIA NIM first and permanently fail over to Groq on provider errors."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        primary: OpenAICompatibleLLM | None = None,
        fallback: OpenAICompatibleLLM | None = None,
    ) -> None:
        settings = settings or Settings.from_env()
        super().__init__()
        self.primary = primary or self._make_provider(
            provider="NVIDIA",
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            model=settings.nvidia_model,
            timeout=settings.nvidia_timeout,
            retries=settings.nvidia_retries,
            settings=settings,
        )
        self.fallback = fallback or self._make_provider(
            provider="Groq",
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            model=settings.groq_model,
            timeout=settings.groq_timeout,
            retries=settings.groq_retries,
            settings=settings,
        )
        self._active: OpenAICompatibleLLM | None = self.primary or self.fallback
        self.model = self._active.model if self._active else ""
        self.provider = self._active.provider if self._active else ""
        self.base_url = self._active.base_url if self._active else ""
        self.failover_reason: str | None = None

    @staticmethod
    def _make_provider(
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        retries: int,
        settings: Settings,
    ) -> OpenAICompatibleLLM | None:
        if not api_key.strip():
            return None
        return OpenAICompatibleLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            retries=retries,
            temperature=settings.temperature,
            max_tokens=settings.max_response_tokens,
            reasoning_format=settings.cloud_reasoning_format,
        )

    def validate_availability(self) -> None:
        if self._active is None:
            raise CloudConfigurationError(
                "Configure NVIDIA_API_KEY (or Nvidia) and/or GROQ_API_KEY (or Groq) in .env"
            )
        self._active.validate_availability()

    def _set_active(self, provider: OpenAICompatibleLLM) -> None:
        self._active = provider
        self.model = provider.model
        self.provider = provider.provider
        self.base_url = provider.base_url

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if self._active is None:
            self.validate_availability()
        assert self._active is not None
        try:
            response = self._active.generate(
                messages, temperature=temperature, max_tokens=max_tokens
            )
            self.last_structured_retries = self._active.last_structured_retries
            return response
        except LLMError as primary_error:
            if self._active is not self.primary or self.fallback is None:
                raise
            logger.warning(
                "NVIDIA request failed; switching to Groq fallback: %s",
                primary_error,
            )
            self.failover_reason = str(primary_error)
            self._set_active(self.fallback)
            try:
                response = self._active.generate(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
                self.last_structured_retries = self._active.last_structured_retries
                return response
            except LLMError as fallback_error:
                raise CloudProviderError(
                    f"NVIDIA failed and Groq fallback failed: {fallback_error}"
                ) from fallback_error
