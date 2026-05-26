"""
VideoBackend Protocol — backend-agnostic interface for video VLM inference.

Implementations:
  MarlinBackend  — NemoStation/Marlin-2B (dense captioning + temporal grounding)
  MockVideoBackend — deterministic stub for unit tests

Both endpoints accept a video_path (local filesystem path) so the service
receives pre-staged video files rather than raw byte streams. Large uploads
should be staged by the caller before hitting /caption or /find.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VideoEvent:
    """A single timestamped event from a caption pass."""
    start: float   # seconds from video start
    end: float     # seconds from video start
    description: str


@dataclass(frozen=True)
class CaptionResult:
    """Result from a /caption call."""
    scene: str               # scene-level description paragraph
    events: list[VideoEvent] # timestamped event list (may be empty)
    caption: str             # full raw caption string from the model
    model: str               # model name / path


@dataclass(frozen=True)
class FindResult:
    """Result from a /find call."""
    span: tuple[float, float] | None  # (start_sec, end_sec) or None on parse failure
    format_ok: bool                   # True when model output matched expected format
    raw: str                          # raw model output for debugging
    model: str


# ── Backend Protocol ─────────────────────────────────────────────────────────

@runtime_checkable
class VideoBackend(Protocol):
    """Minimal interface all video backends must satisfy."""

    def caption(
        self,
        video_path: str,
        *,
        max_new_tokens: int = 2048,
    ) -> CaptionResult: ...

    def find(
        self,
        video_path: str,
        event: str,
        *,
        max_new_tokens: int = 256,
    ) -> FindResult: ...

    @property
    def model_name(self) -> str: ...

    @property
    def vram_mb(self) -> int: ...


# ── Factory ──────────────────────────────────────────────────────────────────

def make_video_backend(
    model_path: str,
    *,
    mock: bool = False,
    device: str = "cuda",
    gpu_id: int = 0,
) -> VideoBackend:
    """Instantiate the appropriate VideoBackend.

    Args:
        model_path: Local filesystem path to the model directory (safetensors).
        mock:       When True, return MockVideoBackend (no GPU required).
        device:     Torch device string ("cuda" or "cpu").
        gpu_id:     CUDA device index — used only when CUDA_VISIBLE_DEVICES is
                    not already set externally (cf-orch sets it before spawning).
    """
    if mock:
        from circuitforge_core.video.backends.mock import MockVideoBackend
        return MockVideoBackend(model_path)
    from circuitforge_core.video.backends.marlin import MarlinBackend
    return MarlinBackend(model_path=model_path, device=device)
