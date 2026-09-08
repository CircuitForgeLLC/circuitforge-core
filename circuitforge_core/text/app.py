"""
cf-text FastAPI service — managed by cf-orch.

Lightweight local text generation and PII filtering. Supports GGUF models via
llama.cpp, HuggingFace transformers, and token-classification models (classifier
backend) for PII detection and redaction.

Endpoints:
  GET  /health      → {"status": "ok", "model": str, "vram_mb": int, "backend": str}
  POST /generate    → GenerateResponse          (text-gen backends only)
  POST /chat        → GenerateResponse          (text-gen backends only)
  POST /filter      → FilterResponse            (classifier backend only)

Usage:
    python -m circuitforge_core.text.app \
        --model /Library/Assets/LLM/qwen2.5-3b-instruct-q4_k_m.gguf \
        --port 8006 \
        --gpu-id 0

Multi-GPU (spans two GPUs via CUDA_VISIBLE_DEVICES, device_map=auto):
    python -m circuitforge_core.text.app \
        --model /Library/Assets/LLM/deepseek-14b \
        --port 8006 \
        --gpu-ids 0,1

Mock mode (no model or GPU required):
    CF_TEXT_MOCK=1 python -m circuitforge_core.text.app --port 8006
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from functools import partial
from typing import Annotated, Literal, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from circuitforge_core.text.backends.base import ChatMessage as BackendChatMessage
from circuitforge_core.text.backends.base import make_classifier_backend, make_text_backend
from circuitforge_core.text.filter import FilterResult, PIIFilter

logger = logging.getLogger(__name__)

_backend = None
_pii_filter: PIIFilter | None = None


# ── Content block types (OpenAI multimodal format) ────────────────────────────


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImageURL(BaseModel):
    type: Literal["image_url"]
    image_url: dict[str, str]


ContentBlock = Annotated[
    Union[ContentBlockText, ContentBlockImageURL],
    Field(discriminator="type"),
]


def _to_backend_message(role: str, content: "str | list[ContentBlock]") -> "BackendChatMessage":
    """Convert an API message to a BackendChatMessage with raw content dicts."""
    if isinstance(content, str):
        return BackendChatMessage(role, content)
    return BackendChatMessage(role, [b.model_dump() for b in content])


# ── Request / response models ─────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    stop: list[str] | None = None


class ChatMessageModel(BaseModel):
    role: str
    content: Union[str, list[ContentBlock]] = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessageModel]
    max_tokens: int = 512
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    text: str
    tokens_used: int = 0
    model: str = ""


class FilterRequest(BaseModel):
    text: str


class PIISpanResponse(BaseModel):
    label: str
    start: int
    end: int
    text: str
    score: float


class FilterResponse(BaseModel):
    redacted_text: str
    spans: list[PIISpanResponse]
    original_text: str
    model: str = ""


# ── OpenAI-compat request / response (for LLMRouter openai_compat path) ──────


class OAIMessageModel(BaseModel):
    role: str
    content: Union[str, list[ContentBlock]] = ""


class OAIChatRequest(BaseModel):
    model: str = "cf-text"
    messages: list[OAIMessageModel]
    max_tokens: int | None = None
    temperature: float = 0.7
    stream: bool = False


class OAIChoice(BaseModel):
    index: int = 0
    message: OAIMessageModel
    finish_reason: str = "stop"


class OAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OAIChoice]
    usage: OAIUsage


# ── App factory ───────────────────────────────────────────────────────────────


def create_app(
    model_path: str,
    gpu_id: int = 0,
    gpu_ids: str | None = None,
    backend: str | None = None,
    mock: bool = False,
    mmproj_path: str = "",
) -> FastAPI:
    """Start the cf-text FastAPI app.

    ``gpu_ids``: comma-separated CUDA device indices for multi-GPU spanning
    (e.g. "0,1"). When set, overrides ``gpu_id`` and sets
    ``CUDA_VISIBLE_DEVICES`` to the full list so HuggingFace Accelerate's
    ``device_map="auto"`` can shard the model across all listed devices.

    When ``backend="classifier"``, the service skips the text-gen backends
    and loads a token-classification pipeline instead. Only ``POST /filter``
    is available in that mode; ``/generate`` and ``/chat`` return 501.
    """
    global _backend, _pii_filter

    if not mock and not model_path:
        raise ValueError(
            "cf-text: --model is required (got empty string). "
            "Pass a GGUF path, a HuggingFace model ID, or set CF_TEXT_MOCK=1 for mock mode."
        )

    visible = gpu_ids if gpu_ids else str(gpu_id)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", visible)

    resolved_backend = backend or os.environ.get("CF_TEXT_BACKEND", "")
    if resolved_backend == "classifier" or (not resolved_backend and False):
        classifier_backend = make_classifier_backend(model_path)
        _pii_filter = PIIFilter.from_backend(classifier_backend)
        logger.info(
            "cf-text (classifier) ready: model=%r vram=%dMB",
            classifier_backend.model_name,
            classifier_backend.vram_mb,
        )
    else:
        _backend = make_text_backend(model_path, backend=backend, mock=mock, mmproj_path=mmproj_path)
        logger.info("cf-text ready: model=%r vram=%dMB", _backend.model_name, _backend.vram_mb)

    app = FastAPI(title="cf-text", version="0.1.0")

    # Readiness probe state. A plain dict rather than a class: it is only ever
    # read and rewritten wholesale by the health handler, and a stale read under
    # concurrency costs at most one extra probe.
    _probe: dict = {"ts": 0.0, "ok": False, "err": "not yet probed"}
    _probe_ttl = float(os.environ.get("CF_TEXT_HEALTH_PROBE_TTL", "30"))

    def _probe_backend() -> tuple[bool, str | None]:
        """Generate a single token, to prove the model actually serves.

        WHY (2026-09-08). The previous check asserted only that the backend
        object was non-None, then reported model_name (a path stem) and vram_mb
        (estimated from the file size on disk). Neither reads the model, so it
        could not fail however broken the model was.

        On 2026-09-07 a cf-text instance on node `sif` served
        GET /health -> 200 {"status":"ok","model":"capybarahermes-2.5-mistral-7b.Q6_K",
        "vram_mb":6232} while every POST /v1/chat/completions threw in 60ms. The
        coordinator had no way to know, kept routing to it, and returned
        502 "Service returned 500" to every caller for hours with the whole
        fleet showing green.

        Cached for CF_TEXT_HEALTH_PROBE_TTL seconds (default 30): the
        coordinator polls health continuously and a generation per poll would be
        a real cost on a shared GPU. The TTL is the trade -- a fault surfaces
        within one TTL rather than instantly.
        """
        now = time.time()
        if now - _probe["ts"] < _probe_ttl:
            return _probe["ok"], _probe["err"]
        try:
            _backend.generate("ok", max_tokens=1)
            _probe.update({"ts": now, "ok": True, "err": None})
        except Exception as exc:  # noqa: BLE001 -- any failure means "not serving"
            _probe.update({"ts": now, "ok": False,
                           "err": f"{exc.__class__.__name__}: {exc}"[:200]})
            logger.warning("cf-text health probe failed: %s", _probe["err"])
        return _probe["ok"], _probe["err"]

    @app.get("/health")
    def health() -> dict:
        if _pii_filter is not None:
            b = _pii_filter._backend
            return {"status": "ok", "model": b.model_name, "vram_mb": b.vram_mb, "backend": "classifier"}
        if _backend is None:
            raise HTTPException(503, detail="backend not initialised")
        ok, err = _probe_backend()
        if not ok:
            # 503 rather than 200-with-a-flag: the coordinator already treats a
            # non-2xx health response as unhealthy, so this surfaces the fault
            # without needing a matching change on the coordinator side.
            raise HTTPException(503, detail=f"model not serving: {err}")
        return {
            "status": "ok",
            "model": _backend.model_name,
            "vram_mb": _backend.vram_mb,
            # Surfaced so an operator can distinguish a freshly-verified "ok"
            # from one coasting on a cached probe -- exactly the distinction
            # that would have made the 2026-09-07 outage obvious at a glance.
            "probed_age_s": round(time.time() - _probe["ts"], 1),
        }

    @app.post("/filter")
    async def filter_text(req: FilterRequest) -> FilterResponse:
        if _pii_filter is None:
            raise HTTPException(
                501,
                detail="This cf-text instance is not running a classifier backend. "
                       "Start with --backend classifier and a token-classification model.",
            )
        result = await _pii_filter.filter_async(req.text)
        return FilterResponse(
            redacted_text=result.redacted_text,
            spans=[
                PIISpanResponse(
                    label=s.label,
                    start=s.start,
                    end=s.end,
                    text=s.text,
                    score=s.score,
                )
                for s in result.spans
            ],
            original_text=result.original_text,
            model=_pii_filter._backend.model_name,
        )

    @app.post("/generate")
    async def generate(req: GenerateRequest) -> GenerateResponse:
        if _pii_filter is not None:
            raise HTTPException(501, detail="classifier backend loaded — use POST /filter")
        if _backend is None:
            raise HTTPException(503, detail="backend not initialised")
        result = await _backend.generate_async(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stop=req.stop,
        )
        return GenerateResponse(
            text=result.text,
            tokens_used=result.tokens_used,
            model=result.model,
        )

    @app.post("/chat")
    async def chat(req: ChatRequest) -> GenerateResponse:
        if _pii_filter is not None:
            raise HTTPException(501, detail="classifier backend loaded — use POST /filter")
        if _backend is None:
            raise HTTPException(503, detail="backend not initialised")
        messages = [_to_backend_message(m.role, m.content) for m in req.messages]
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                partial(_backend.chat, messages,
                        max_tokens=req.max_tokens, temperature=req.temperature),
            )
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        return GenerateResponse(
            text=result.text,
            tokens_used=result.tokens_used,
            model=result.model,
        )

    @app.post("/v1/chat/completions")
    async def oai_chat_completions(req: OAIChatRequest) -> OAIChatResponse:
        """OpenAI-compatible chat completions endpoint.

        Allows LLMRouter (and any openai_compat client) to use cf-text
        without a custom backend type — just set base_url to this service's
        /v1 prefix.
        """
        if _backend is None:
            raise HTTPException(503, detail="backend not initialised")
        messages = [_to_backend_message(m.role, m.content) for m in req.messages]
        max_tok = req.max_tokens or 512
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                partial(_backend.chat, messages, max_tokens=max_tok, temperature=req.temperature),
            )
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        return OAIChatResponse(
            id=f"cftext-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=result.model or req.model,
            choices=[OAIChoice(message=OAIMessageModel(role="assistant", content=result.text))],
            usage=OAIUsage(completion_tokens=result.tokens_used, total_tokens=result.tokens_used),
        )

    return app


# ── CLI entrypoint ────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cf-text inference server")
    parser.add_argument("--model", default=os.environ.get("CF_TEXT_MODEL", "mock"),
                        help="Path to GGUF file or HF model ID")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="CUDA device index to use (single GPU)")
    parser.add_argument("--gpu-ids", default=None,
                        help="Comma-separated CUDA device indices for multi-GPU spanning "
                             "(e.g. '0,1'). Overrides --gpu-id when set.")
    parser.add_argument(
        "--backend",
        choices=["llamacpp", "transformers", "ollama", "vllm", "classifier"],
        default=None,
    )
    parser.add_argument(
        "--mmproj", default="",
        help="Path to multimodal projector file for VLM GGUF models (LLaVA-style). "
             "Qwen2-VL and other self-contained VLMs don't need this.",
    )
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode (no model or GPU needed)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    args = _parse_args()
    mock = args.mock or os.environ.get("CF_TEXT_MOCK", "") == "1" or args.model == "mock"
    app = create_app(
        model_path=args.model,
        gpu_id=args.gpu_id,
        gpu_ids=args.gpu_ids,
        backend=args.backend,
        mock=mock,
        mmproj_path=args.mmproj,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
