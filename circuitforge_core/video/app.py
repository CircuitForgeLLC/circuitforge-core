"""
cf-video FastAPI service — managed by cf-orch.

Endpoints:
  GET  /health     → {"status": "ok", "model": str, "vram_mb": int}
  POST /caption    → CaptionResponse (scene + timestamped events)
  POST /find       → FindResponse (temporal grounding span)

Usage:
    python -m circuitforge_core.video.app \
        --model /Library/Assets/LLM/cf-video/models/NemoStation--Marlin-2B \
        --port 8016 \
        --gpu-id 0

The service loads the model once at startup and blocks until it is ready.
cf-orch health-polls /health before routing any inference requests.

Model requirements:
    transformers >= 5.7.0
    torch        >= 2.11.0
    torchcodec   (installed)
    qwen-vl-utils >= 0.0.14 (installed)

Security:
    Marlin requires trust_remote_code=True. Review the model's
    modeling_marlin.py before deploying on a production node.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from circuitforge_core.video.backends.base import VideoBackend, make_video_backend

app = FastAPI(title="cf-video", version="0.1.0")
_backend: VideoBackend | None = None


# ── Request / response models ─────────────────────────────────────────────────

class CaptionRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to the video file on this node")
    max_new_tokens: int = Field(2048, ge=64, le=8192)


class VideoEventOut(BaseModel):
    start: float
    end: float
    description: str


class CaptionResponse(BaseModel):
    scene: str
    events: list[VideoEventOut]
    caption: str
    model: str


class FindRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to the video file on this node")
    event: str = Field(..., min_length=1, description="Natural-language event description to locate")
    max_new_tokens: int = Field(256, ge=32, le=2048)


class FindResponse(BaseModel):
    span: list[float] | None = Field(
        None,
        description="[start_sec, end_sec] or null when the model could not ground the event",
    )
    format_ok: bool
    raw: str
    model: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    if _backend is None:
        raise HTTPException(503, detail="backend not initialised")
    return {
        "status": "ok",
        "model": _backend.model_name,
        "vram_mb": _backend.vram_mb,
    }


@app.post("/caption", response_model=CaptionResponse)
def caption(req: CaptionRequest) -> CaptionResponse:
    if _backend is None:
        raise HTTPException(503, detail="backend not initialised")
    try:
        result = _backend.caption(req.video_path, max_new_tokens=req.max_new_tokens)
    except FileNotFoundError as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("caption failed for %r", req.video_path)
        raise HTTPException(500, detail=str(exc)) from exc

    return CaptionResponse(
        scene=result.scene,
        events=[
            VideoEventOut(start=ev.start, end=ev.end, description=ev.description)
            for ev in result.events
        ],
        caption=result.caption,
        model=result.model,
    )


@app.post("/find", response_model=FindResponse)
def find(req: FindRequest) -> FindResponse:
    if _backend is None:
        raise HTTPException(503, detail="backend not initialised")
    try:
        result = _backend.find(
            req.video_path,
            req.event,
            max_new_tokens=req.max_new_tokens,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("find failed for %r event=%r", req.video_path, req.event)
        raise HTTPException(500, detail=str(exc)) from exc

    return FindResponse(
        span=list(result.span) if result.span is not None else None,
        format_ok=result.format_ok,
        raw=result.raw,
        model=result.model,
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cf-video service (Marlin-2B)")
    p.add_argument(
        "--model",
        required=True,
        help="Local filesystem path to the Marlin model directory (safetensors)",
    )
    p.add_argument("--port", type=int, default=8016)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument(
        "--gpu-id", type=int, default=0,
        help="CUDA device index; overridden by CUDA_VISIBLE_DEVICES when set by cf-orch",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--mock", action="store_true",
        help="Run with MockVideoBackend (no GPU, for testing)",
    )
    return p.parse_args()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    args = _parse_args()

    # Pin GPU selection unconditionally — --gpu-id is authoritative.
    # Force PCI_BUS_ID ordering so --gpu-id matches nvidia-smi (not CUDA's
    # default FASTEST_FIRST, which can swap indices on multi-GPU nodes).
    if args.device == "cuda" and not args.mock:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    mock = args.mock or args.model == "mock"
    device = "cpu" if mock else args.device

    _backend = make_video_backend(
        model_path=args.model,
        mock=mock,
        device=device,
        gpu_id=args.gpu_id,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
