from __future__ import annotations

from typing import Any

from config.settings import Settings
from llm.ollama_client import OllamaLLM


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if url.endswith("/api/tags"):
            return FakeResponse({"models": [{"name": "spark-x2.5-4b"}]})
        return FakeResponse({
            "model": "spark-x2.5-4b",
            "message": {"role": "assistant", "content": "local response"},
            "prompt_eval_count": 3,
            "eval_count": 2,
        })


def test_ollama_client_validates_model_and_uses_chat_endpoint() -> None:
    client = OllamaLLM(Settings(ollama_base_url="http://ollama.test", ollama_model="spark-x2.5-4b"))
    session = FakeSession()
    client.session = session
    client.validate_availability()
    response = client.generate([{"role": "user", "content": "hello"}])
    assert response.content == "local response"
    assert response.prompt_tokens == 3
    assert any(url.endswith("/api/chat") for _, url, _ in session.calls)
