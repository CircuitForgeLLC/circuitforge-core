"""
Tests for the cf-video FastAPI app using mock backend.

Tests run without GPU, torch, or a real video file.
MockVideoBackend checks os.path.exists() but never reads video content,
so a zero-byte placeholder is sufficient.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import circuitforge_core.video.app as video_app
from circuitforge_core.video.backends.mock import MockVideoBackend


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def inject_mock_backend():
    """Replace global backend with mock before each test; restore after."""
    original = video_app._backend
    video_app._backend = MockVideoBackend()
    yield
    video_app._backend = original


@pytest.fixture()
def client():
    return TestClient(video_app.app)


@pytest.fixture()
def video_file(tmp_path):
    """Placeholder file that satisfies os.path.exists() inside the mock."""
    p = tmp_path / "sample.mp4"
    p.write_bytes(b"\x00" * 16)
    return str(p)


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "mock"
    assert data["vram_mb"] == 0


def test_health_503_when_no_backend(client):
    video_app._backend = None
    resp = client.get("/health")
    assert resp.status_code == 503


# ── /caption ──────────────────────────────────────────────────────────────────


def test_caption_returns_200(client, video_file):
    resp = client.post("/caption", json={"video_path": video_file})
    assert resp.status_code == 200


def test_caption_response_has_scene(client, video_file):
    data = client.post("/caption", json={"video_path": video_file}).json()
    assert isinstance(data["scene"], str)
    assert data["scene"]


def test_caption_response_has_events(client, video_file):
    data = client.post("/caption", json={"video_path": video_file}).json()
    assert isinstance(data["events"], list)
    assert len(data["events"]) >= 1


def test_caption_events_have_timestamps(client, video_file):
    data = client.post("/caption", json={"video_path": video_file}).json()
    for ev in data["events"]:
        assert "start" in ev
        assert "end" in ev
        assert "description" in ev
        assert ev["start"] <= ev["end"]


def test_caption_response_has_caption(client, video_file):
    data = client.post("/caption", json={"video_path": video_file}).json()
    assert isinstance(data["caption"], str)
    assert data["caption"]


def test_caption_response_model_field(client, video_file):
    data = client.post("/caption", json={"video_path": video_file}).json()
    assert isinstance(data["model"], str)


def test_caption_404_on_missing_file(client):
    resp = client.post("/caption", json={"video_path": "/no/such/file.mp4"})
    assert resp.status_code == 404


def test_caption_503_when_no_backend(client, video_file):
    video_app._backend = None
    resp = client.post("/caption", json={"video_path": video_file})
    assert resp.status_code == 503


def test_caption_custom_max_new_tokens(client, video_file):
    resp = client.post(
        "/caption",
        json={"video_path": video_file, "max_new_tokens": 512},
    )
    assert resp.status_code == 200


def test_caption_rejects_max_new_tokens_below_min(client, video_file):
    resp = client.post(
        "/caption",
        json={"video_path": video_file, "max_new_tokens": 10},
    )
    assert resp.status_code == 422


def test_caption_rejects_max_new_tokens_above_max(client, video_file):
    resp = client.post(
        "/caption",
        json={"video_path": video_file, "max_new_tokens": 99999},
    )
    assert resp.status_code == 422


# ── /find ─────────────────────────────────────────────────────────────────────


def test_find_returns_200(client, video_file):
    resp = client.post(
        "/find",
        json={"video_path": video_file, "event": "someone waves"},
    )
    assert resp.status_code == 200


def test_find_response_has_span(client, video_file):
    data = client.post(
        "/find",
        json={"video_path": video_file, "event": "mock event"},
    ).json()
    # MockVideoBackend always returns a non-null span
    assert data["span"] is not None
    assert len(data["span"]) == 2
    assert data["span"][0] <= data["span"][1]


def test_find_span_is_list_of_floats(client, video_file):
    data = client.post(
        "/find",
        json={"video_path": video_file, "event": "mock event"},
    ).json()
    span = data["span"]
    assert all(isinstance(v, float) for v in span)


def test_find_format_ok_field(client, video_file):
    data = client.post(
        "/find",
        json={"video_path": video_file, "event": "mock event"},
    ).json()
    assert data["format_ok"] is True


def test_find_raw_field(client, video_file):
    data = client.post(
        "/find",
        json={"video_path": video_file, "event": "mock event"},
    ).json()
    assert isinstance(data["raw"], str)


def test_find_model_field(client, video_file):
    data = client.post(
        "/find",
        json={"video_path": video_file, "event": "mock event"},
    ).json()
    assert isinstance(data["model"], str)


def test_find_404_on_missing_file(client):
    resp = client.post(
        "/find",
        json={"video_path": "/no/such/file.mp4", "event": "wave"},
    )
    assert resp.status_code == 404


def test_find_503_when_no_backend(client, video_file):
    video_app._backend = None
    resp = client.post(
        "/find",
        json={"video_path": video_file, "event": "wave"},
    )
    assert resp.status_code == 503


def test_find_rejects_empty_event(client, video_file):
    resp = client.post(
        "/find",
        json={"video_path": video_file, "event": ""},
    )
    assert resp.status_code == 422


def test_find_custom_max_new_tokens(client, video_file):
    resp = client.post(
        "/find",
        json={"video_path": video_file, "event": "wave", "max_new_tokens": 128},
    )
    assert resp.status_code == 200


def test_find_rejects_max_new_tokens_below_min(client, video_file):
    resp = client.post(
        "/find",
        json={"video_path": video_file, "event": "wave", "max_new_tokens": 10},
    )
    assert resp.status_code == 422


def test_find_rejects_max_new_tokens_above_max(client, video_file):
    resp = client.post(
        "/find",
        json={"video_path": video_file, "event": "wave", "max_new_tokens": 99999},
    )
    assert resp.status_code == 422
