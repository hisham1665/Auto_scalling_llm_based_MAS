from __future__ import annotations

from typing import Any

import pytest

from config.settings import Settings
from llm.cloud_client import (
    CloudConfigurationError,
    CloudFailoverLLM,
    OpenAICompatibleLLM,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs["json"], kwargs["headers"]))
        return self.response


def _response(content: str = "cloud response") -> FakeResponse:
    return FakeResponse(
        {
            "model": "cloud-model",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3},
        }
    )


def test_openai_compatible_client_builds_chat_completion_request() -> None:
    session = FakeSession(_response())
    client = OpenAICompatibleLLM(
        provider="NVIDIA",
        api_key="nvidia-secret",
        base_url="https://nvidia.test/v1",
        model="nemotron",
        session=session,
    )

    result = client.generate(
        [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ],
        max_tokens=20,
    )

    assert result.content == "cloud response"
    assert result.prompt_tokens == 4
    assert session.calls[0][0] == "https://nvidia.test/v1/chat/completions"
    assert session.calls[0][1]["model"] == "nemotron"
    assert session.calls[0][1]["messages"][0]["content"] == "/no_think\nbe concise"
    assert session.calls[0][2]["Authorization"] == "Bearer nvidia-secret"


def test_failover_switches_from_nvidia_to_groq() -> None:
    nvidia = OpenAICompatibleLLM(
        provider="NVIDIA",
        api_key="nvidia-secret",
        base_url="https://nvidia.test/v1",
        model="nemotron",
        session=FakeSession(FakeResponse({"error": {"message": "rate limit"}}, 429)),
    )
    groq = OpenAICompatibleLLM(
        provider="Groq",
        api_key="groq-secret",
        base_url="https://groq.test/openai/v1",
        model="qwen/qwen3-32b",
        session=FakeSession(_response("fallback response")),
    )
    client = CloudFailoverLLM(Settings(), primary=nvidia, fallback=groq)

    result = client.generate([{"role": "user", "content": "hello"}])

    assert result.content == "fallback response"
    assert client.provider == "Groq"
    assert client.model == "qwen/qwen3-32b"
    assert client.failover_reason and "HTTP 429" in client.failover_reason
    assert groq.session.calls[0][1]["reasoning_format"] == "hidden"


def test_failover_requires_at_least_one_cloud_key() -> None:
    client = CloudFailoverLLM(Settings(nvidia_api_key="", groq_api_key=""))

    with pytest.raises(CloudConfigurationError, match="NVIDIA_API_KEY"):
        client.validate_availability()


def test_nemotron_lightning_disables_reasoning_with_template_parameter() -> None:
    session = FakeSession(_response())
    client = OpenAICompatibleLLM(
        provider="NVIDIA",
        api_key="nvidia-secret",
        base_url="https://nvidia.test/v1",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        session=session,
    )

    client.generate([{"role": "user", "content": "hello"}], max_tokens=20)

    assert session.calls[0][1]["chat_template_kwargs"] == {"enable_thinking": False}
