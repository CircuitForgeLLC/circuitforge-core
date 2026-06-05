# circuitforge_core/text/backends/mock.py — synthetic text backend
#
# MIT licensed. No model file, no GPU, no extras required.
# Used in dev, CI, and free-tier nodes below the minimum VRAM threshold.
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterator

from circuitforge_core.text.backends.base import ChatMessage, GenerateResult

_MOCK_RESPONSE = (
    "This is a synthetic response from MockTextBackend. "
    "Install a real backend (llama-cpp-python or transformers) and provide a model path "
    "to generate real text."
)


class MockTextBackend:
    """
    Deterministic synthetic text backend for development and CI.

    Always returns the same fixed response so tests are reproducible without
    a GPU or model file. Streaming emits the response word-by-word with a
    configurable delay so UI streaming paths can be exercised.
    """

    def __init__(
        self,
        model_name: str = "mock",
        token_delay_s: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._token_delay_s = token_delay_s

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vram_mb(self) -> int:
        return 0

    def _response_for(self, prompt_or_messages: str) -> str:
        return _MOCK_RESPONSE

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerateResult:
        text = self._response_for(prompt)
        return GenerateResult(text=text, tokens_used=len(text.split()), model=self._model_name)

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        import time
        for word in self._response_for(prompt).split():
            yield word + " "
            if self._token_delay_s:
                time.sleep(self._token_delay_s)

    async def generate_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerateResult:
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)

    async def generate_stream_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        for word in self._response_for(prompt).split():
            yield word + " "
            if self._token_delay_s:
                await asyncio.sleep(self._token_delay_s)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> GenerateResult:
        # Format messages into a simple prompt for the mock response
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)


# Synthetic PII spans injected by MockClassifierBackend — predictable in tests.
_MOCK_SPANS = [
    {
        "entity_group": "NAME",
        "score": 0.99,
        "word": "Jane Doe",
        "start": 0,
        "end": 8,
    },
    {
        "entity_group": "EMAIL",
        "score": 0.97,
        "word": "jane@example.com",
        "start": 18,
        "end": 34,
    },
]


class MockClassifierBackend:
    """
    Deterministic mock classifier backend for development and CI.

    Always returns the same two synthetic PII spans regardless of input.
    Allows filter.py logic (redaction, span conversion) to be tested without
    a real model or GPU.
    """

    def __init__(self, model_name: str = "mock-classifier") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vram_mb(self) -> int:
        return 0

    def classify(self, text: str) -> list[dict]:
        return list(_MOCK_SPANS)

    async def classify_async(self, text: str) -> list[dict]:
        return self.classify(text)
