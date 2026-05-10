"""Tests for the OpenAI-compatible answer generator."""

from dataclasses import dataclass
from email.message import Message
from io import BytesIO
from urllib import error

import pytest

from labelrag.generation.openai_compatible import (
    OpenAICompatibleAnswerGenerator,
    OpenAICompatibleConfig,
)


@dataclass
class StubOpenAICompatibleAnswerGenerator(OpenAICompatibleAnswerGenerator):
    """Test helper that injects a deterministic JSON response."""

    response: dict[str, object] | None = None

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        response: dict[str, object] | None = None,
    ) -> None:
        super().__init__(config)
        self.response = response

    def _post_json(self, *, url: str, payload: dict[str, object]) -> dict[str, object]:
        """Return the preconfigured response instead of performing a network request."""

        assert url.endswith("/chat/completions")
        assert payload["model"] == self.config.model
        if self.response is None:
            raise RuntimeError("Stub response is not configured.")
        return self.response

    def resolve_api_key_for_test(self) -> str:
        """Expose API key resolution for tests without touching protected methods."""

        return self._resolve_api_key()


def test_openai_compatible_generator_builds_text_answer() -> None:
    """The generator should parse a standard chat-completions response."""

    generator = StubOpenAICompatibleAnswerGenerator(
        OpenAICompatibleConfig(
            model="test-model",
            api_key="secret",
            base_url="https://example.com/v1",
        ),
        response={
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "content": "Answer text.",
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        },
    )
    result = generator.generate("What happened?", "Context here.")

    assert result.text == "Answer text."
    assert result.metadata["model"] == "test-model"
    assert result.metadata["usage"]["total_tokens"] == 14


def test_openai_compatible_generator_accepts_full_endpoint_url() -> None:
    """A full chat-completions endpoint URL should be used as-is."""

    generator = StubOpenAICompatibleAnswerGenerator(
        OpenAICompatibleConfig(
            model="test-model",
            api_key="secret",
            base_url="https://example.com/v1/chat/completions",
        ),
        response={"choices": [{"message": {"content": "Answer text."}}]},
    )

    result = generator.generate("What happened?", "Context here.")

    assert result.text == "Answer text."


def test_openai_compatible_generator_supports_content_part_lists() -> None:
    """The generator should join text parts from compatible structured responses."""

    generator = StubOpenAICompatibleAnswerGenerator(
        OpenAICompatibleConfig(model="test-model", api_key="secret"),
        response={
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": " world"},
                        ]
                    }
                }
            ]
        },
    )
    result = generator.generate("What happened?", "Context here.")

    assert result.text == "Hello world"


def test_openai_compatible_generator_requires_api_key() -> None:
    """The generator should fail clearly when no API key is available."""

    generator = StubOpenAICompatibleAnswerGenerator(
        OpenAICompatibleConfig(model="test-model", api_key=None, api_key_env_var="MISSING_API_KEY")
    )

    with pytest.raises(RuntimeError, match="Missing API key"):
        generator.resolve_api_key_for_test()


def test_openai_compatible_generator_formats_http_errors() -> None:
    """HTTP errors should raise a useful runtime error."""

    generator = OpenAICompatibleAnswerGenerator(
        OpenAICompatibleConfig(
            model="test-model", api_key="secret", base_url="https://example.com/v1"
        )
    )
    headers = Message()

    def fake_urlopen(request_obj: object, timeout: float) -> object:
        del request_obj, timeout
        raise error.HTTPError(
            url="https://example.com/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=headers,
            fp=BytesIO(b'{"error":"bad request"}'),
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("labelrag.generation.openai_compatible.request.urlopen", fake_urlopen)
    try:
        with pytest.raises(RuntimeError, match="HTTP 400"):
            generator.generate("What happened?", "Context here.")
    finally:
        monkeypatch.undo()


def test_openai_compatible_generator_uses_environment_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generator should resolve API keys from the configured environment variable."""

    monkeypatch.setenv("TEST_OPENAI_API_KEY", "env-secret")
    generator = StubOpenAICompatibleAnswerGenerator(
        OpenAICompatibleConfig(
            model="test-model",
            api_key=None,
            api_key_env_var="TEST_OPENAI_API_KEY",
        )
    )

    assert generator.resolve_api_key_for_test() == "env-secret"
