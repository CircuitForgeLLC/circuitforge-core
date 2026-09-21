"""Tests for circuitforge_core.api.feedback — shared feedback router factory."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from circuitforge_core.api.feedback import make_feedback_router

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(
    demo_mode_fn=None,
    repo: str = "Circuit-Forge/test",
    product: str = "test",
) -> TestClient:
    app = FastAPI()
    router = make_feedback_router(repo=repo, product=product, demo_mode_fn=demo_mode_fn)
    app.include_router(router, prefix="/feedback")
    return TestClient(app)


def _mock_forgejo_get(label_list: list[dict] | None = None):
    """Return a mock for requests.get that returns an empty label list."""
    mock = MagicMock()
    mock.ok = True
    mock.json.return_value = label_list or []
    return mock


def _mock_forgejo_post_issue(number: int = 42, url: str = "https://example.com/issues/42"):
    """Return a mock for requests.post that simulates a successful issue creation."""
    mock = MagicMock()
    mock.ok = True
    mock.json.return_value = {"number": number, "html_url": url}
    return mock


_VALID_PAYLOAD = {
    "title": "Something broke",
    "description": "It stopped working after the update.",
    "type": "bug",
    "repro": "1. Open app\n2. Click submit\n3. See error",
    "tab": "dashboard",
    "submitter": "alan@example.com",
}

# ---------------------------------------------------------------------------
# GET /feedback/status
# ---------------------------------------------------------------------------

def test_status_no_token_returns_disabled(monkeypatch):
    """GET /status returns enabled=False when FORGEJO_API_TOKEN is not set."""
    monkeypatch.delenv("FORGEJO_API_TOKEN", raising=False)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    client = _make_client()
    resp = client.get("/feedback/status")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": False}


def test_status_with_token_returns_enabled(monkeypatch):
    """GET /status returns enabled=True when token is set and not in demo mode."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token-abc")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    client = _make_client()
    resp = client.get("/feedback/status")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": True}


def test_status_demo_mode_env_returns_disabled(monkeypatch):
    """GET /status returns enabled=False when DEMO_MODE=1 even with a token."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token-abc")
    monkeypatch.setenv("DEMO_MODE", "1")
    client = _make_client()
    resp = client.get("/feedback/status")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": False}


def test_status_demo_mode_fn_returns_disabled(monkeypatch):
    """GET /status returns enabled=False when demo_mode_fn() returns True."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token-abc")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    client = _make_client(demo_mode_fn=lambda: True)
    resp = client.get("/feedback/status")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": False}


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------

def test_post_no_token_returns_503(monkeypatch):
    """POST / returns 503 when FORGEJO_API_TOKEN is not configured."""
    monkeypatch.delenv("FORGEJO_API_TOKEN", raising=False)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    client = _make_client()
    resp = client.post("/feedback", json=_VALID_PAYLOAD)
    assert resp.status_code == 503
    assert "FORGEJO_API_TOKEN" in resp.json()["detail"]


def test_post_demo_mode_fn_returns_403(monkeypatch):
    """POST / returns 403 when demo_mode_fn returns True."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token-abc")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    client = _make_client(demo_mode_fn=lambda: True)
    resp = client.post("/feedback", json=_VALID_PAYLOAD)
    assert resp.status_code == 403
    assert "demo" in resp.json()["detail"].lower()


def test_post_success_returns_issue_number_and_url(monkeypatch):
    """POST / returns issue_number and issue_url on success."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token-abc")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("FORGEJO_API_URL", "https://forgejo.test/api/v1")

    mock_get = _mock_forgejo_get()
    mock_post_label = MagicMock(ok=True)
    mock_post_label.json.return_value = {"id": 99, "name": "beta-feedback"}
    mock_post_issue = _mock_forgejo_post_issue(number=7, url="https://forgejo.test/Circuit-Forge/test/issues/7")

    # requests.post is called multiple times: once per new label, then once for the issue.
    # We use side_effect to distinguish label creation calls from the issue creation call.
    issue_call_kwargs: list[dict] = []

    def post_side_effect(url, **kwargs):
        if "/labels" in url:
            return mock_post_label
        issue_call_kwargs.append(kwargs)
        return mock_post_issue

    client = _make_client()
    with patch("circuitforge_core.api.feedback.requests.get", return_value=mock_get), \
         patch("circuitforge_core.api.feedback.requests.post", side_effect=post_side_effect):
        resp = client.post("/feedback", json=_VALID_PAYLOAD)

    assert resp.status_code == 200
    data = resp.json()
    assert data["issue_number"] == 7
    assert data["issue_url"] == "https://forgejo.test/Circuit-Forge/test/issues/7"

    # Verify label IDs were forwarded to the issue creation call.
    # mock_post_label returns id=99 for each of the 3 new labels → [99, 99, 99]
    assert issue_call_kwargs, "Issue creation call was not made"
    assert issue_call_kwargs[0]["json"]["labels"] == [99, 99, 99]


def test_post_default_forgejo_url_is_current_domain_when_env_unset(monkeypatch):
    """When FORGEJO_API_URL is not set, the default must point at the live
    Forgejo domain (git.circuitforge.tech), not the retired
    git.opensourcesolarpunk.com, which 410s. Every other test in this file
    explicitly sets FORGEJO_API_URL, so none of them would have caught a
    regression to a stale default -- this test exercises that default
    directly. Regression test for a live bug found during Peregrine's
    v1.0.0 RC1 cloud verification (2026-09-20): real feedback submissions
    were failing in production on both cloud and self-hosted installs that
    never set this env var, since it always fell through to the dead
    domain."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token-abc")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("FORGEJO_API_URL", raising=False)

    mock_get = _mock_forgejo_get()
    mock_post_label = MagicMock(ok=True)
    mock_post_label.json.return_value = {"id": 99, "name": "beta-feedback"}
    mock_post_issue = _mock_forgejo_post_issue(number=7, url="https://git.circuitforge.tech/Circuit-Forge/test/issues/7")

    requested_urls: list[str] = []

    def get_side_effect(url, **kwargs):
        requested_urls.append(url)
        return mock_get

    def post_side_effect(url, **kwargs):
        requested_urls.append(url)
        if "/labels" in url:
            return mock_post_label
        return mock_post_issue

    client = _make_client()
    with patch("circuitforge_core.api.feedback.requests.get", side_effect=get_side_effect), \
         patch("circuitforge_core.api.feedback.requests.post", side_effect=post_side_effect):
        resp = client.post("/feedback", json=_VALID_PAYLOAD)

    assert resp.status_code == 200
    assert requested_urls, "No HTTP calls were made"
    for url in requested_urls:
        assert url.startswith("https://git.circuitforge.tech/api/v1"), (
            f"Expected the current Forgejo domain, got: {url}"
        )
        assert "opensourcesolarpunk" not in url


def test_post_returns_502_on_label_creation_failure(monkeypatch):
    """POST / returns 502 when Forgejo label creation fails."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("FORGEJO_API_URL", "https://forgejo.test/api/v1")

    label_list_response = MagicMock()
    label_list_response.ok = True
    label_list_response.json.return_value = []  # no existing labels

    label_create_response = MagicMock()
    label_create_response.ok = False
    label_create_response.text = "forbidden"

    client = _make_client()
    with patch("circuitforge_core.api.feedback.requests.get", return_value=label_list_response), \
         patch("circuitforge_core.api.feedback.requests.post", return_value=label_create_response):
        res = client.post("/feedback", json={
            "title": "Test", "description": "desc", "type": "other",
        })
    assert res.status_code == 502
    assert "beta-feedback" in res.json()["detail"]


def test_post_forgejo_error_returns_502(monkeypatch):
    """POST / returns 502 when Forgejo returns a non-ok response for issue creation."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token-abc")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("FORGEJO_API_URL", "https://forgejo.test/api/v1")

    mock_get = _mock_forgejo_get()
    mock_issue_error = MagicMock(ok=False)
    mock_issue_error.text = "Internal Server Error"

    def post_side_effect(url, **kwargs):
        if "/labels" in url:
            m = MagicMock(ok=True)
            m.json.return_value = {"id": 1, "name": "beta-feedback"}
            return m
        return mock_issue_error

    client = _make_client()
    with patch("circuitforge_core.api.feedback.requests.get", return_value=mock_get), \
         patch("circuitforge_core.api.feedback.requests.post", side_effect=post_side_effect):
        resp = client.post("/feedback", json=_VALID_PAYLOAD)

    assert resp.status_code == 502
    assert "Forgejo error" in resp.json()["detail"]


def test_post_product_name_appears_in_issue_body(monkeypatch):
    """The product name passed to make_feedback_router appears in the issue body context."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token-abc")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("FORGEJO_API_URL", "https://forgejo.test/api/v1")

    captured_body: list[str] = []
    mock_get = _mock_forgejo_get()

    def post_side_effect(url, **kwargs):
        if "/labels" in url:
            m = MagicMock(ok=True)
            m.json.return_value = {"id": 1, "name": "beta-feedback"}
            return m
        # Capture the body sent for the issue creation call
        captured_body.append(kwargs.get("json", {}).get("body", ""))
        m = MagicMock(ok=True)
        m.json.return_value = {"number": 1, "html_url": "https://forgejo.test/issues/1"}
        return m

    client = _make_client(product="kiwi")
    with patch("circuitforge_core.api.feedback.requests.get", return_value=mock_get), \
         patch("circuitforge_core.api.feedback.requests.post", side_effect=post_side_effect):
        resp = client.post(
            "/feedback",
            json={
                "title": "Pantry bug",
                "description": "Items disappear.",
                "type": "bug",
                "tab": "pantry",
            },
        )

    assert resp.status_code == 200
    assert captured_body, "No issue body was captured"
    assert "kiwi" in captured_body[0], f"Product name not found in body: {captured_body[0]}"


def test_post_bug_with_repro_includes_repro_section(monkeypatch):
    """A bug report with a repro string includes the Reproduction Steps section in the body."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token-abc")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("FORGEJO_API_URL", "https://forgejo.test/api/v1")

    captured_body: list[str] = []
    mock_get = _mock_forgejo_get()

    def post_side_effect(url, **kwargs):
        if "/labels" in url:
            m = MagicMock(ok=True)
            m.json.return_value = {"id": 1, "name": "bug"}
            return m
        captured_body.append(kwargs.get("json", {}).get("body", ""))
        m = MagicMock(ok=True)
        m.json.return_value = {"number": 2, "html_url": "https://forgejo.test/issues/2"}
        return m

    repro_text = "1. Open the app\n2. Click the button\n3. Observe crash"
    client = _make_client()
    with patch("circuitforge_core.api.feedback.requests.get", return_value=mock_get), \
         patch("circuitforge_core.api.feedback.requests.post", side_effect=post_side_effect):
        resp = client.post(
            "/feedback",
            json={
                "title": "App crashes",
                "description": "The app crashes on button click.",
                "type": "bug",
                "repro": repro_text,
                "tab": "home",
            },
        )

    assert resp.status_code == 200
    assert captured_body, "No issue body was captured"
    body = captured_body[0]
    assert "Reproduction Steps" in body, f"'Reproduction Steps' not found in body: {body}"
    assert repro_text in body, f"Repro text not found in body: {body}"


def test_status_demo_mode_env_true_string(monkeypatch):
    """GET /status treats DEMO_MODE=true as demo mode."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token-abc")
    monkeypatch.setenv("DEMO_MODE", "true")
    client = _make_client()
    resp = client.get("/feedback/status")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": False}


def test_post_existing_labels_reused(monkeypatch):
    """When labels already exist on Forgejo, their IDs are reused (no POST to /labels)."""
    monkeypatch.setenv("FORGEJO_API_TOKEN", "test-token-abc")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("FORGEJO_API_URL", "https://forgejo.test/api/v1")

    existing_labels = [
        {"name": "beta-feedback", "id": 10},
        {"name": "needs-triage", "id": 11},
        {"name": "bug", "id": 12},
    ]
    mock_get = _mock_forgejo_get(existing_labels)
    label_post_calls: list[str] = []

    def post_side_effect(url, **kwargs):
        if "/labels" in url:
            label_post_calls.append(url)
            m = MagicMock(ok=True)
            m.json.return_value = {"id": 99, "name": "new-label"}
            return m
        m = MagicMock(ok=True)
        m.json.return_value = {"number": 5, "html_url": "https://forgejo.test/issues/5"}
        return m

    client = _make_client()
    with patch("circuitforge_core.api.feedback.requests.get", return_value=mock_get), \
         patch("circuitforge_core.api.feedback.requests.post", side_effect=post_side_effect):
        resp = client.post("/feedback", json=_VALID_PAYLOAD)

    assert resp.status_code == 200
    assert label_post_calls == [], "Should not POST to /labels when all labels already exist"
