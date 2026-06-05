# circuitforge_core/text/backends/llamacpp.py — llama-cpp-python backend
#
# BSL 1.1: real inference. Requires llama-cpp-python + a GGUF model file.
# Install: pip install circuitforge-core[text-llamacpp]
#
# VRAM estimates (Q4_K_M quant):
#   1B  → ~700MB    3B  → ~2048MB   7B  → ~4096MB
#   13B → ~7500MB   70B → ~40000MB
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import AsyncIterator, Iterator

from circuitforge_core.text.backends.base import ChatMessage, GenerateResult

logger = logging.getLogger(__name__)

# Q4_K_M is the recommended default — best accuracy/size tradeoff for local use.
_DEFAULT_N_CTX = int(os.environ.get("CF_TEXT_CTX", "4096"))
_DEFAULT_N_GPU_LAYERS = int(os.environ.get("CF_TEXT_GPU_LAYERS", "-1"))  # -1 = all layers


def _estimate_vram_mb(model_path: str) -> int:
    """Rough VRAM estimate from file size. Accurate enough for cf-orch budgeting."""
    try:
        size_mb = Path(model_path).stat().st_size // (1024 * 1024)
        # GGUF models typically need ~1.1× file size in VRAM (KV cache overhead)
        return int(size_mb * 1.1)
    except OSError:
        return 4096  # conservative default


class LlamaCppBackend:
    """
    Direct llama-cpp-python inference backend for GGUF models.

    The model is loaded once at construction. All inference runs in a thread
    pool executor so async callers never block the event loop.

    Context window, GPU layers, and thread count are configurable via env:
        CF_TEXT_CTX          token context window (default 4096)
        CF_TEXT_GPU_LAYERS   GPU layers to offload, -1 = all (default -1)
        CF_TEXT_THREADS      CPU thread count (default: auto)

    Requires: pip install circuitforge-core[text-llamacpp]
    """

    def __init__(self, model_path: str, mmproj_path: str = "", chat_format: str = "") -> None:
        """Load a GGUF model.

        ``mmproj_path``: path to a separate multimodal projector file (needed
        for LLaVA-style VLMs where the visual encoder is a separate .gguf).
        Qwen2-VL and similar models with an embedded projector don't need this.

        ``chat_format``: llama-cpp chat template override (e.g. "llava-1-5",
        "moondream").  Required when mmproj_path is set.
        """
        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppBackend. "
                "Install with: pip install circuitforge-core[text-llamacpp]"
            ) from exc

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"GGUF model not found: {model_path}\n"
                "Download a GGUF model and set CF_TEXT_MODEL to its path."
            )

        # If given a directory, find the .gguf file inside it.
        if Path(model_path).is_dir():
            candidates = sorted(Path(model_path).glob("*.gguf")) or sorted(Path(model_path).glob("*.GGUF"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .gguf file found in directory: {model_path}"
                )
            model_path = str(candidates[0])

        n_threads = int(os.environ.get("CF_TEXT_THREADS", "0")) or None

        kwargs: dict = dict(
            model_path=model_path,
            n_ctx=_DEFAULT_N_CTX,
            n_gpu_layers=_DEFAULT_N_GPU_LAYERS,
            n_threads=n_threads,
            verbose=False,
        )
        if mmproj_path:
            kwargs["clip_model_path"] = mmproj_path
            kwargs["chat_format"] = chat_format or "llava-1-5"
            logger.info(
                "Loading VLM %s with mmproj %s (ctx=%d, gpu_layers=%d)",
                model_path, mmproj_path, _DEFAULT_N_CTX, _DEFAULT_N_GPU_LAYERS,
            )
        else:
            logger.info(
                "Loading GGUF model %s (ctx=%d, gpu_layers=%d)",
                model_path, _DEFAULT_N_CTX, _DEFAULT_N_GPU_LAYERS,
            )

        self._llm = Llama(**kwargs)
        self._model_path = model_path
        self._vram_mb = _estimate_vram_mb(model_path)
        # True when the model was initialised with a visual encoder (explicit
        # mmproj) or when it is a known self-contained VLM (Qwen2-VL, etc.).
        self._is_vlm = bool(mmproj_path) or self._detect_embedded_vlm()

    def _detect_embedded_vlm(self) -> bool:
        """Heuristic: check model metadata for a known multimodal architecture."""
        try:
            meta = self._llm.metadata or {}
            arch = str(meta.get("general.architecture", "")).lower()
            # Qwen2-VL and similar embed the vision encoder inside the GGUF.
            return any(tag in arch for tag in ("qwen2_vl", "llava", "moondream", "minicpm-v"))
        except Exception:
            return False

    @property
    def model_name(self) -> str:
        return Path(self._model_path).stem

    @property
    def vram_mb(self) -> int:
        return self._vram_mb

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerateResult:
        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            stream=False,
        )
        text = output["choices"][0]["text"]
        tokens_used = output["usage"]["completion_tokens"]
        return GenerateResult(text=text, tokens_used=tokens_used, model=self.model_name)

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        for chunk in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            stream=True,
        ):
            yield chunk["choices"][0]["text"]

    async def generate_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerateResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop),
        )

    async def generate_stream_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        # llama_cpp streaming is synchronous — run in executor and re-emit tokens
        import queue
        import threading

        token_queue: queue.Queue = queue.Queue()
        _DONE = object()

        def _produce() -> None:
            try:
                for chunk in self._llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop or [],
                    stream=True,
                ):
                    token_queue.put(chunk["choices"][0]["text"])
            finally:
                token_queue.put(_DONE)

        thread = threading.Thread(target=_produce, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        while True:
            token = await loop.run_in_executor(None, token_queue.get)
            if token is _DONE:
                break
            yield token

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> GenerateResult:
        # Detect image content before calling the model.
        if any(m.has_images for m in messages) and not self._is_vlm:
            raise ValueError(
                "model does not support image input — "
                "load a VLM (with mmproj_path) or route to cf-vision/cf-docuvision"
            )
        # llama-cpp-python create_chat_completion accepts content as str or
        # list-of-blocks (OpenAI multimodal format) natively.
        output = self._llm.create_chat_completion(
            messages=[m.to_dict() for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = output["choices"][0]["message"]["content"]
        tokens_used = output["usage"]["completion_tokens"]
        return GenerateResult(text=text, tokens_used=tokens_used, model=self.model_name)
