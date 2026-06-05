# circuitforge_core/text/backends/transformers.py — HuggingFace transformers backend
#
# BSL 1.1: real inference. Requires torch + transformers + a model checkpoint.
# Install: pip install circuitforge-core[text-transformers]
#
# Best for: HF repo IDs, safetensors checkpoints, models without GGUF versions.
# For GGUF models prefer LlamaCppBackend — lower overhead, smaller install.
from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator, Iterator

from circuitforge_core.text.backends.base import ChatMessage, GenerateResult

logger = logging.getLogger(__name__)

_DEFAULT_MAX_NEW_TOKENS = 512
_LOAD_IN_4BIT = os.environ.get("CF_TEXT_4BIT", "0") == "1"
_LOAD_IN_8BIT = os.environ.get("CF_TEXT_8BIT", "0") == "1"


class TransformersBackend:
    """
    HuggingFace transformers inference backend.

    Loads any causal LM available on HuggingFace Hub or a local checkpoint dir.
    Supports 4-bit and 8-bit quantization via bitsandbytes when VRAM is limited:
        CF_TEXT_4BIT=1  — load_in_4bit (requires bitsandbytes)
        CF_TEXT_8BIT=1  — load_in_8bit (requires bitsandbytes)

    Chat completion uses the tokenizer's apply_chat_template() when available,
    falling back to a simple "User: / Assistant:" prompt format.

    Requires: pip install circuitforge-core[text-transformers]
    """

    def __init__(self, model_path: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for TransformersBackend. "
                "Install with: pip install circuitforge-core[text-transformers]"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading transformers model %s on %s", model_path, self._device)

        load_kwargs: dict = {"device_map": "auto" if self._device == "cuda" else None}
        if _LOAD_IN_4BIT or _LOAD_IN_8BIT:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=_LOAD_IN_4BIT,
                load_in_8bit=_LOAD_IN_8BIT,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if self._device == "cpu":
            self._model = self._model.to("cpu")

        self._model_path = model_path
        self._TextIteratorStreamer = TextIteratorStreamer

    @property
    def model_name(self) -> str:
        # HF repo IDs contain "/" — use the part after the slash as a short name
        return self._model_path.split("/")[-1]

    @property
    def vram_mb(self) -> int:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() // (1024 * 1024)
        except Exception:
            pass
        return 0

    def _build_inputs(self, prompt: str):
        return self._tokenizer(prompt, return_tensors="pt").to(self._device)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerateResult:
        inputs = self._build_inputs(prompt)
        input_len = inputs["input_ids"].shape[1]
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenerateResult(text=text, tokens_used=len(new_tokens), model=self.model_name)

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        import threading

        inputs = self._build_inputs(prompt)
        streamer = self._TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            streamer=streamer,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        yield from streamer

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
        import queue
        import threading

        token_queue: queue.Queue = queue.Queue()
        _DONE = object()

        def _produce() -> None:
            try:
                for token in self.generate_stream(
                    prompt, max_tokens=max_tokens, temperature=temperature
                ):
                    token_queue.put(token)
            finally:
                token_queue.put(_DONE)

        threading.Thread(target=_produce, daemon=True).start()
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
        # Use the tokenizer's chat template when available (instruct models)
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            prompt = self._tokenizer.apply_chat_template(
                [m.to_dict() for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(
                f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
                for m in messages
                if m.role != "system"
            ) + "\nAssistant:"

        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
