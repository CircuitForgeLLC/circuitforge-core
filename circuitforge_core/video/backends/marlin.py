"""
MarlinBackend — NemoStation/Marlin-2B video VLM via HuggingFace Transformers.

Marlin-2B is a decoder-only video understanding model that produces:
  - Dense scene captions with second-precise event timestamps (/caption)
  - Temporal grounding of natural-language events (/find)

Requirements (install separately):
    pip install "transformers>=5.7.0" "torch>=2.11.0" torchcodec "qwen-vl-utils>=0.0.14" av pillow

Security note:
    trust_remote_code=True is required. The model ships a custom
    AutoModelForCausalLM subclass (modeling_marlin.py). Review that file
    before enabling on any node. The modeling code runs in-process with
    full filesystem access.

Environment variables forwarded to the model's preprocessing layer:
    FORCE_QWENVL_VIDEO_READER  default: torchcodec   (video decode backend)
    VIDEO_MAX_PIXELS           default: 200704        (max pixels per frame)
    FPS                        default: 2.0           (frame sample rate)
    FPS_MAX_FRAMES             default: 240           (frame cap ~2 min video)
    FPS_MIN_FRAMES             default: 4             (minimum frames)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from circuitforge_core.video.backends.base import CaptionResult, FindResult, VideoEvent

logger = logging.getLogger(__name__)

# Default env overrides so torchcodec is preferred over the slower av/ffmpeg path.
_DEFAULT_ENV: dict[str, str] = {
    "FORCE_QWENVL_VIDEO_READER": "torchcodec",
}


class MarlinBackend:
    """
    Load Marlin-2B once, expose caption() and find() as synchronous calls.

    The model is loaded eagerly in __init__ — if loading fails (OOM, missing
    weights, transformers version mismatch) the error propagates immediately
    rather than on first inference, so cf-orch's 2-second liveness check can
    catch it.
    """

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self._model_path = model_path
        self._device = device

        # Apply env defaults before importing transformers — the model's
        # custom __init__.py reads these at import time.
        for key, val in _DEFAULT_ENV.items():
            os.environ.setdefault(key, val)

        self._model = self._load_model(model_path, device)
        self._vram_mb = self._estimate_vram_mb()
        logger.info(
            "MarlinBackend: loaded %r on %s (~%d MB VRAM)",
            model_path, device, self._vram_mb,
        )

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_model(self, model_path: str, device: str):
        import torch
        from transformers import AutoModelForCausalLM

        # Verify weights exist before handing to transformers — gives a clear
        # error instead of a cryptic trust_remote_code failure.
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Marlin model directory not found: {model_path!r}. "
                "Download via Avocet or: "
                f"huggingface-cli download NemoStation/Marlin-2B --local-dir {model_path}"
            )

        logger.info("MarlinBackend: loading model from %r ...", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,      # Required — custom modeling code in repo
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        model.eval()
        logger.info("MarlinBackend: model loaded")
        return model

    def _estimate_vram_mb(self) -> int:
        """Read allocated VRAM from torch after load; fall back to catalog estimate."""
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated() / 1024 / 1024)
        except Exception:
            pass
        return 4500  # Catalog estimate for Marlin-2B BF16

    # ── Inference ────────────────────────────────────────────────────────────

    def caption(
        self,
        video_path: str,
        *,
        max_new_tokens: int = 2048,
    ) -> CaptionResult:
        """Produce a dense caption with scene description and timestamped events."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path!r}")

        raw_result: dict = self._model.caption(
            video_path,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        events = [
            VideoEvent(
                start=float(ev["start"]),
                end=float(ev["end"]),
                description=str(ev["description"]),
            )
            for ev in raw_result.get("events", [])
        ]

        return CaptionResult(
            scene=str(raw_result.get("scene", "")),
            events=events,
            caption=str(raw_result.get("caption", "")),
            model=self.model_name,
        )

    def find(
        self,
        video_path: str,
        event: str,
        *,
        max_new_tokens: int = 256,
    ) -> FindResult:
        """Ground a natural-language event query to a video time span."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path!r}")
        if not event.strip():
            raise ValueError("event query must not be empty")

        raw_result: dict = self._model.find(
            video_path,
            event=event,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Marlin returns span as a (start, end) tuple or None.
        raw_span = raw_result.get("span")
        span: tuple[float, float] | None = None
        if raw_span is not None:
            try:
                span = (float(raw_span[0]), float(raw_span[1]))
            except (TypeError, IndexError, ValueError):
                logger.warning(
                    "MarlinBackend.find: could not parse span %r for event %r",
                    raw_span, event,
                )

        return FindResult(
            span=span,
            format_ok=bool(raw_result.get("format_ok", False)),
            raw=str(raw_result.get("raw", "")),
            model=self.model_name,
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_path

    @property
    def vram_mb(self) -> int:
        return self._vram_mb
