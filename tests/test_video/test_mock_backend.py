"""
Tests for MockVideoBackend and the VideoBackend protocol.

All tests run without a GPU, torch install, or any real video file
(MockVideoBackend only checks os.path.exists, not video validity).
"""
from __future__ import annotations

import os
import tempfile

import pytest

from circuitforge_core.video.backends.base import (
    CaptionResult,
    FindResult,
    VideoBackend,
    VideoEvent,
    make_video_backend,
)
from circuitforge_core.video.backends.mock import MockVideoBackend


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def video_file(tmp_path):
    """Create a temporary file that satisfies os.path.exists() checks."""
    p = tmp_path / "test.mp4"
    p.write_bytes(b"\x00" * 16)  # placeholder bytes; mock never reads content
    return str(p)


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_mock_satisfies_protocol():
    backend = MockVideoBackend()
    assert isinstance(backend, VideoBackend)


def test_mock_model_name_default():
    assert MockVideoBackend().model_name == "mock"


def test_mock_model_name_custom():
    assert MockVideoBackend(model_path="custom-path").model_name == "custom-path"


def test_mock_vram_mb():
    assert MockVideoBackend().vram_mb == 0


# ── caption() ─────────────────────────────────────────────────────────────────


def test_caption_returns_caption_result(video_file):
    result = MockVideoBackend().caption(video_file)
    assert isinstance(result, CaptionResult)


def test_caption_scene_is_str(video_file):
    result = MockVideoBackend().caption(video_file)
    assert isinstance(result.scene, str)
    assert result.scene  # non-empty


def test_caption_events_are_video_events(video_file):
    result = MockVideoBackend().caption(video_file)
    assert isinstance(result.events, list)
    for ev in result.events:
        assert isinstance(ev, VideoEvent)


def test_caption_events_have_numeric_timestamps(video_file):
    result = MockVideoBackend().caption(video_file)
    for ev in result.events:
        assert isinstance(ev.start, float)
        assert isinstance(ev.end, float)
        assert ev.start <= ev.end


def test_caption_caption_str(video_file):
    result = MockVideoBackend().caption(video_file)
    assert isinstance(result.caption, str)
    assert result.caption


def test_caption_model_matches_path(video_file):
    result = MockVideoBackend(model_path="test-model").caption(video_file)
    assert result.model == "test-model"


def test_caption_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        MockVideoBackend().caption("/nonexistent/video.mp4")


def test_caption_max_new_tokens_accepted(video_file):
    """max_new_tokens kwarg must be accepted without error."""
    result = MockVideoBackend().caption(video_file, max_new_tokens=512)
    assert isinstance(result, CaptionResult)


# ── find() ────────────────────────────────────────────────────────────────────


def test_find_returns_find_result(video_file):
    result = MockVideoBackend().find(video_file, "someone waves")
    assert isinstance(result, FindResult)


def test_find_span_is_tuple_or_none(video_file):
    result = MockVideoBackend().find(video_file, "mock event")
    # MockVideoBackend always returns a span
    assert result.span is not None
    assert len(result.span) == 2
    assert result.span[0] <= result.span[1]


def test_find_format_ok_true(video_file):
    result = MockVideoBackend().find(video_file, "mock event")
    assert result.format_ok is True


def test_find_raw_is_str(video_file):
    result = MockVideoBackend().find(video_file, "mock event")
    assert isinstance(result.raw, str)


def test_find_model_matches_path(video_file):
    result = MockVideoBackend(model_path="my-model").find(video_file, "event")
    assert result.model == "my-model"


def test_find_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        MockVideoBackend().find("/nonexistent/video.mp4", "event")


def test_find_max_new_tokens_accepted(video_file):
    result = MockVideoBackend().find(video_file, "event", max_new_tokens=128)
    assert isinstance(result, FindResult)


# ── make_video_backend factory ────────────────────────────────────────────────


def test_factory_returns_mock_when_flag_set():
    backend = make_video_backend(model_path="mock", mock=True)
    assert isinstance(backend, MockVideoBackend)


def test_factory_mock_uses_model_path():
    backend = make_video_backend(model_path="some-path", mock=True)
    assert backend.model_name == "some-path"
