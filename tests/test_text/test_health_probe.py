"""cf-text /health must exercise the completion path (2026-09-08).

Motivating incident. A cf-text instance on node `sif` served

    GET  /health               -> 200 {"status":"ok",
                                       "model":"capybarahermes-2.5-mistral-7b.Q6_K",
                                       "vram_mb":6232}
    POST /v1/chat/completions  -> 500, in 60ms

for hours. The coordinator had no way to tell, kept routing to it, and returned
502 "Service returned 500" to every caller while the whole fleet showed green.

The old check asserted only that the backend object was non-None, then reported
attributes fixed at construction: model_name is a path stem, vram_mb is
estimated from the file size on disk. Neither reads the model, so the check
could not fail however broken the model was. A liveness check that cannot fail
is not a check.

These tests hold the probe to the one property that matters -- a backend that
cannot generate must not report healthy -- plus the cost trade that keeps it
affordable to poll.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from circuitforge_core.text.app import create_app


@pytest.fixture
def client(monkeypatch):
    # _pii_filter is a module-level global that create_app() sets but never
    # clears, so a classifier test running earlier in the session leaves it set
    # and /health takes the classifier branch before ever reaching the probe.
    # Reset it here rather than depending on test order.
    import circuitforge_core.text.app as app_mod
    monkeypatch.setattr(app_mod, "_pii_filter", None, raising=False)
    monkeypatch.setenv("CF_TEXT_HEALTH_PROBE_TTL", "0")   # probe on every call
    return TestClient(create_app(model_path="", mock=True))


def _break_generation(monkeypatch, exc=RuntimeError("CUDA context lost")):
    """Fail the way a dead model does: object intact, attributes fine, generation throws."""
    import circuitforge_core.text.app as app_mod

    def boom(*a, **kw):
        raise exc

    monkeypatch.setattr(app_mod._backend, "generate", boom)


def test_a_working_backend_reports_healthy(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_reports_how_stale_its_evidence_is(client):
    """So an operator can tell a freshly-verified ok from one coasting on a
    cached probe -- the distinction that would have made the outage obvious."""
    assert "probed_age_s" in client.get("/health").json()


def test_a_backend_that_cannot_generate_is_reported_unhealthy(client, monkeypatch):
    """THE regression test: the exact shape of the sif incident."""
    _break_generation(monkeypatch)
    r = client.get("/health")
    assert r.status_code == 503
    assert "not serving" in r.json()["detail"]


def test_the_failure_reason_is_surfaced_not_swallowed(client, monkeypatch):
    """Otherwise the operator learns "unhealthy" and nothing else, and has to go
    read logs on a node they may not be able to reach -- which is what made the
    original diagnosis take hours."""
    _break_generation(monkeypatch, ValueError("model handle is closed"))
    detail = client.get("/health").json()["detail"]
    assert "ValueError" in detail and "model handle is closed" in detail


def test_recovery_is_detected_without_a_restart(client, monkeypatch):
    """A transient fault must not latch the service unhealthy forever."""
    import circuitforge_core.text.app as app_mod
    original = app_mod._backend.generate

    _break_generation(monkeypatch)
    assert client.get("/health").status_code == 503

    monkeypatch.setattr(app_mod._backend, "generate", original)
    assert client.get("/health").status_code == 200


def test_the_probe_is_cached_so_polling_does_not_cost_a_generation_each_time(monkeypatch):
    """The coordinator polls health continuously; a generation per poll would be
    a real cost on a shared GPU. The TTL is the detection-latency trade."""
    import circuitforge_core.text.app as app_mod
    monkeypatch.setattr(app_mod, "_pii_filter", None, raising=False)
    monkeypatch.setenv("CF_TEXT_HEALTH_PROBE_TTL", "3600")
    c = TestClient(create_app(model_path="", mock=True))

    calls = []
    real = app_mod._backend.generate

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(app_mod._backend, "generate", counting)
    for _ in range(5):
        c.get("/health")
    assert len(calls) <= 1, "health probed the model on every poll despite the TTL"


def test_the_probe_asks_for_a_single_token_so_it_stays_cheap(client, monkeypatch):
    import circuitforge_core.text.app as app_mod
    seen = {}

    def capture(prompt, **kw):
        seen.update(kw)
        return "ok"

    monkeypatch.setattr(app_mod._backend, "generate", capture)
    client.get("/health")
    assert seen.get("max_tokens") == 1
