"""
MockVideoBackend — deterministic stub for unit tests and CI.

Returns fixed CaptionResult / FindResult without any model or video I/O.
"""
from __future__ import annotations

import os

from circuitforge_core.video.backends.base import (
    CaptionResult,
    FindResult,
    VideoEvent,
)

_MOCK_SCENE = "A mock scene with placeholder content."
_MOCK_EVENTS = [
    VideoEvent(start=0.0, end=3.0, description="Mock event one"),
    VideoEvent(start=3.5, end=7.2, description="Mock event two"),
]
_MOCK_CAPTION = "Scene: A mock scene with placeholder content. Events: [0.0-3.0] Mock event one. [3.5-7.2] Mock event two."
_MOCK_FIND_SPAN = (3.5, 7.2)


class MockVideoBackend:
    """No-GPU stub. Safe for import on any machine."""

    def __init__(self, model_path: str = "mock") -> None:
        self._model_path = model_path

    def caption(
        self,
        video_path: str,
        *,
        max_new_tokens: int = 2048,
    ) -> CaptionResult:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path!r}")
        return CaptionResult(
            scene=_MOCK_SCENE,
            events=list(_MOCK_EVENTS),
            caption=_MOCK_CAPTION,
            model=self.model_name,
        )

    def find(
        self,
        video_path: str,
        event: str,
        *,
        max_new_tokens: int = 256,
    ) -> FindResult:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path!r}")
        return FindResult(
            span=_MOCK_FIND_SPAN,
            format_ok=True,
            raw="From 3.5 to 7.2.",
            model=self.model_name,
        )

    @property
    def model_name(self) -> str:
        return self._model_path

    @property
    def vram_mb(self) -> int:
        return 0
