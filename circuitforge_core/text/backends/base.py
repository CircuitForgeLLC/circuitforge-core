# circuitforge_core/text/backends/base.py — TextBackend Protocol + factory
#
# MIT licensed. The Protocol and mock backend are always importable.
# Real backends (LlamaCppBackend, TransformersBackend) require optional extras.
from __future__ import annotations

import os
from typing import AsyncIterator, Iterator, Protocol, runtime_checkable


# ── Shared result types ───────────────────────────────────────────────────────


class GenerateResult:
    """Result from a single non-streaming generate() call."""

    def __init__(self, text: str, tokens_used: int = 0, model: str = "") -> None:
        self.text = text
        self.tokens_used = tokens_used
        self.model = model

    def __repr__(self) -> str:
        return f"GenerateResult(text={self.text!r:.40}, tokens={self.tokens_used})"


class ChatMessage:
    """A single message in a chat conversation.

    ``content`` is either a plain string or a list of OpenAI-format content
    blocks (dicts with ``type: "text"`` or ``type: "image_url"``).  Backends
    that do not support images should call ``text_only`` to get the string
    form before passing to the model.
    """

    def __init__(self, role: str, content: "str | list") -> None:
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Invalid role {role!r}. Must be system, user, or assistant.")
        self.role = role
        self.content: "str | list" = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @property
    def has_images(self) -> bool:
        """True when at least one content block is an image_url block."""
        if isinstance(self.content, str):
            return False
        return any(
            isinstance(b, dict) and b.get("type") == "image_url"
            for b in self.content
        )

    @property
    def text_only(self) -> str:
        """Flatten multimodal content to text. Returns content as-is if already str."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            b["text"]
            for b in self.content
            if isinstance(b, dict) and b.get("type") == "text"
        )


# ── TextBackend Protocol ──────────────────────────────────────────────────────


@runtime_checkable
class TextBackend(Protocol):
    """
    Abstract interface for direct text generation backends.

    All generate/chat methods have both sync and async variants.
    Streaming variants yield str tokens rather than a complete result.

    Implementations must be safe to construct once and call concurrently
    (the model is loaded at construction time and reused across calls).
    """

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerateResult:
        """Synchronous generate — blocks until the full response is produced."""
        ...

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """Synchronous streaming — yields tokens as they are produced."""
        ...

    async def generate_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerateResult:
        """Async generate — runs in thread pool, never blocks the event loop."""
        ...

    async def generate_stream_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Async streaming — yields tokens without blocking the event loop."""
        ...

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> GenerateResult:
        """Chat completion — formats messages into a prompt and generates."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier for the loaded model (path stem or HF repo ID)."""
        ...

    @property
    def vram_mb(self) -> int:
        """Approximate VRAM footprint in MB. Used by cf-orch service registry."""
        ...


# ── FilterBackend Protocol ────────────────────────────────────────────────────


@runtime_checkable
class FilterBackend(Protocol):
    """
    Abstract interface for token-classification / PII-filter backends.

    Separate from TextBackend — returns entity spans and redacted text,
    not generated text.
    """

    def classify(self, text: str) -> list[dict]:
        """Synchronous classify — returns list of entity span dicts."""
        ...

    async def classify_async(self, text: str) -> list[dict]:
        """Async classify — runs in thread pool."""
        ...

    @property
    def model_name(self) -> str: ...

    @property
    def vram_mb(self) -> int: ...


# ── Backend selection ─────────────────────────────────────────────────────────


def _select_backend(model_path: str, backend: str | None) -> str:
    """
    Return "llamacpp", "transformers", "ollama", or "vllm" for the given model path.

    Parameters
    ----------
    model_path  Path to the model file, HuggingFace repo ID, "ollama://<name>",
                or "vllm://<model-id>".
    backend     Explicit override from the caller
                ("llamacpp" | "transformers" | "ollama" | "vllm" | None).
                When provided, trust it without inspection.

    Raise ValueError for unrecognised override values.
    """
    _VALID = ("llamacpp", "transformers", "ollama", "vllm", "classifier")

    # 1. Caller-supplied override — highest trust, no inspection needed.
    resolved = backend or os.environ.get("CF_TEXT_BACKEND")
    if resolved:
        if resolved not in _VALID:
            raise ValueError(
                f"CF_TEXT_BACKEND={resolved!r} is not valid. Choose: {', '.join(_VALID)}"
            )
        return resolved

    # 2. Proxy prefixes — unambiguous routing regardless of model name format.
    if model_path.startswith("ollama://"):
        return "ollama"
    if model_path.startswith("vllm://"):
        return "vllm"

    # 3. Format detection — GGUF files are unambiguously llama-cpp territory.
    if model_path.lower().endswith(".gguf"):
        return "llamacpp"
    # 3b. GGUF directory — avocet downloads whole repos; scan for .gguf contents.
    if os.path.isdir(model_path):
        import glob as _glob
        if _glob.glob(os.path.join(model_path, "*.gguf")) or _glob.glob(os.path.join(model_path, "*.GGUF")):
            return "llamacpp"

    # 4. Safe default — transformers covers HF repo IDs and safetensors dirs.
    return "transformers"


# ── Factory ───────────────────────────────────────────────────────────────────


def make_text_backend(
    model_path: str,
    backend: str | None = None,
    mock: bool | None = None,
    mmproj_path: str = "",
) -> "TextBackend":
    """
    Return a TextBackend for the given model.

    mock=True or CF_TEXT_MOCK=1  → MockTextBackend (no GPU, no model file needed)
    Otherwise                    → backend resolved via _select_backend()
    """
    use_mock = mock if mock is not None else os.environ.get("CF_TEXT_MOCK", "") == "1"
    if use_mock:
        from circuitforge_core.text.backends.mock import MockTextBackend
        return MockTextBackend(model_name=model_path)

    resolved = _select_backend(model_path, backend)

    if resolved == "llamacpp":
        from circuitforge_core.text.backends.llamacpp import LlamaCppBackend
        return LlamaCppBackend(model_path=model_path, mmproj_path=mmproj_path)

    if resolved == "transformers":
        from circuitforge_core.text.backends.transformers import TransformersBackend
        return TransformersBackend(model_path=model_path)

    if resolved == "ollama":
        from circuitforge_core.text.backends.ollama import OllamaBackend
        return OllamaBackend(model_path=model_path)

    if resolved == "vllm":
        from circuitforge_core.text.backends.vllm import VllmBackend
        return VllmBackend(model_path=model_path)

    raise ValueError(
        f"Unknown backend {resolved!r}. "
        "Expected 'llamacpp', 'transformers', 'ollama', 'vllm', or 'classifier'."
    )


def make_classifier_backend(model_path: str) -> "FilterBackend":
    """
    Return a FilterBackend for the given token-classification model.

    CF_TEXT_MOCK=1  → MockClassifierBackend (no GPU, no model file needed)
    Otherwise       → ClassifierBackend via transformers pipeline
    """
    if os.environ.get("CF_TEXT_MOCK", "") == "1":
        from circuitforge_core.text.backends.mock import MockClassifierBackend
        return MockClassifierBackend(model_name=model_path)

    from circuitforge_core.text.backends.classifier import ClassifierBackend
    return ClassifierBackend(model_path=model_path)
