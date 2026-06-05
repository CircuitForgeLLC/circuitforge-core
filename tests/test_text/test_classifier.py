# tests/test_text/test_classifier.py — PII filter backend and endpoint tests
import pytest
from httpx import AsyncClient, ASGITransport

from circuitforge_core.text.backends.mock import MockClassifierBackend
from circuitforge_core.text.filter import PIIFilter, PIISpan, FilterResult, _redact, _spans_from_pipeline


# ── Unit: _spans_from_pipeline ────────────────────────────────────────────────


def test_spans_from_pipeline_normalises_bio_prefix():
    raw = [{"entity_group": "B-NAME", "score": 0.9, "word": "Alice", "start": 0, "end": 5}]
    spans = _spans_from_pipeline(raw)
    assert spans[0].label == "NAME"


def test_spans_from_pipeline_uppercase():
    raw = [{"entity_group": "email", "score": 0.8, "word": "a@b.com", "start": 10, "end": 17}]
    spans = _spans_from_pipeline(raw)
    assert spans[0].label == "EMAIL"


def test_spans_from_pipeline_returns_typed_objects():
    raw = [{"entity_group": "PHONE_NUM", "score": 0.95, "word": "555-1234", "start": 5, "end": 13}]
    spans = _spans_from_pipeline(raw)
    assert isinstance(spans[0], PIISpan)
    assert spans[0].score == pytest.approx(0.95)
    assert spans[0].start == 5
    assert spans[0].end == 13


# ── Unit: _redact ─────────────────────────────────────────────────────────────


def test_redact_replaces_spans():
    text = "Call Alice at 555-1234 now"
    spans = [
        PIISpan(label="NAME", start=5, end=10, text="Alice", score=0.99),
        PIISpan(label="PHONE_NUM", start=14, end=22, text="555-1234", score=0.97),
    ]
    assert _redact(text, spans) == "Call [NAME] at [PHONE_NUM] now"


def test_redact_handles_overlapping_order():
    # Spans processed right-to-left — earlier offsets must still be valid
    text = "Jane Doe jane@example.com"
    spans = [
        PIISpan(label="NAME", start=0, end=8, text="Jane Doe", score=0.99),
        PIISpan(label="EMAIL", start=9, end=25, text="jane@example.com", score=0.97),
    ]
    result = _redact(text, spans)
    assert "[NAME]" in result
    assert "[EMAIL]" in result
    assert "Jane Doe" not in result
    assert "jane@example.com" not in result


def test_redact_no_spans_returns_original():
    text = "No PII here"
    assert _redact(text, []) == text


# ── Unit: PIIFilter with MockClassifierBackend ────────────────────────────────


def test_pii_filter_sync():
    backend = MockClassifierBackend()
    pii_filter = PIIFilter.from_backend(backend)
    # Mock backend returns spans for "Jane Doe" at 0-8 and "jane@example.com" at 18-34
    result = pii_filter.filter("Jane Doe emailed jane@example.com today")
    assert isinstance(result, FilterResult)
    assert "[NAME]" in result.redacted_text
    assert "[EMAIL]" in result.redacted_text
    assert len(result.spans) == 2


def test_pii_filter_preserves_original_text():
    backend = MockClassifierBackend()
    pii_filter = PIIFilter.from_backend(backend)
    text = "Jane Doe emailed jane@example.com today"
    result = pii_filter.filter(text)
    assert result.original_text == text


@pytest.mark.asyncio
async def test_pii_filter_async():
    backend = MockClassifierBackend()
    pii_filter = PIIFilter.from_backend(backend)
    result = await pii_filter.filter_async("Jane Doe emailed jane@example.com today")
    assert "[NAME]" in result.redacted_text
    assert len(result.spans) == 2


def test_pii_filter_result_is_frozen():
    backend = MockClassifierBackend()
    pii_filter = PIIFilter.from_backend(backend)
    result = pii_filter.filter("test")
    with pytest.raises((AttributeError, TypeError)):
        result.redacted_text = "mutated"  # type: ignore[misc]


# ── Integration: /filter HTTP endpoint ───────────────────────────────────────


@pytest.fixture
def classifier_app(monkeypatch):
    """cf-text app in classifier mode using mock backend."""
    import os
    monkeypatch.setenv("CF_TEXT_MOCK", "1")
    monkeypatch.setenv("CF_TEXT_BACKEND", "classifier")
    import importlib
    import circuitforge_core.text.app as app_mod
    importlib.reload(app_mod)
    yield app_mod.create_app(model_path="openai/privacy-filter", backend="classifier", mock=False)
    monkeypatch.delenv("CF_TEXT_MOCK", raising=False)
    monkeypatch.delenv("CF_TEXT_BACKEND", raising=False)


@pytest.mark.asyncio
async def test_filter_endpoint_returns_redacted(classifier_app):
    async with AsyncClient(transport=ASGITransport(app=classifier_app), base_url="http://test") as client:
        resp = await client.post("/filter", json={"text": "Jane Doe emailed jane@example.com today"})
    assert resp.status_code == 200
    body = resp.json()
    assert "[NAME]" in body["redacted_text"]
    assert "[EMAIL]" in body["redacted_text"]
    assert len(body["spans"]) == 2


@pytest.mark.asyncio
async def test_filter_endpoint_includes_original(classifier_app):
    text = "Jane Doe emailed jane@example.com today"
    async with AsyncClient(transport=ASGITransport(app=classifier_app), base_url="http://test") as client:
        resp = await client.post("/filter", json={"text": text})
    assert resp.json()["original_text"] == text


@pytest.mark.asyncio
async def test_generate_returns_501_in_classifier_mode(classifier_app):
    async with AsyncClient(transport=ASGITransport(app=classifier_app), base_url="http://test") as client:
        resp = await client.post("/generate", json={"prompt": "hello"})
    assert resp.status_code == 501


@pytest.mark.asyncio
async def test_health_reports_classifier_backend(classifier_app):
    async with AsyncClient(transport=ASGITransport(app=classifier_app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["backend"] == "classifier"
