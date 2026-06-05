# circuitforge_core/text/filter.py — PII detection and redaction
#
# BSL 1.1. Products import PIIFilter for pre-send redaction and audit trails.
# Requires a running cf-filter service (or ClassifierBackend for in-process use).
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from circuitforge_core.text.backends.base import FilterBackend, make_classifier_backend


@dataclass(frozen=True)
class PIISpan:
    """A single detected PII entity in the source text."""

    label: str    # e.g. NAME | EMAIL | PHONE_NUM | ADDRESS | SSN | DOB | IP_ADDRESS
    start: int    # char offset (inclusive) in original_text
    end: int      # char offset (exclusive) in original_text
    text: str     # original span text
    score: float  # confidence score from the classifier


@dataclass(frozen=True)
class FilterResult:
    """Output of PIIFilter.filter().

    ``redacted_text``: safe-to-send copy with each span replaced by ``[LABEL]``.
    ``spans``:         all detected entities — for audit logs or caller-side decisions.
    ``original_text``: the input text (stored for round-trip comparisons).
    """

    redacted_text: str
    spans: list[PIISpan] = field(default_factory=list)
    original_text: str = ""


def _redact(text: str, spans: list[PIISpan]) -> str:
    """Replace each span in text with ``[LABEL]``, processing right-to-left so
    earlier offsets remain valid after each substitution."""
    result = text
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        result = result[: span.start] + f"[{span.label}]" + result[span.end :]
    return result


def _spans_from_pipeline(raw: list[dict[str, Any]]) -> list[PIISpan]:
    """Convert raw pipeline output dicts into typed PIISpan objects.

    Pipeline returns dicts with keys: entity_group, score, word, start, end.
    Normalise label to uppercase and strip any residual BIO prefixes.
    """
    spans: list[PIISpan] = []
    for item in raw:
        label = re.sub(r"^[BI]-", "", item.get("entity_group", "")).upper()
        spans.append(
            PIISpan(
                label=label,
                start=int(item["start"]),
                end=int(item["end"]),
                text=item.get("word", ""),
                score=float(item.get("score", 0.0)),
            )
        )
    return spans


class PIIFilter:
    """
    High-level PII filter backed by a token-classification model.

    Usage:
        pii_filter = PIIFilter.from_model("openai/privacy-filter")
        result = await pii_filter.filter_async(resume_text)
        safe_text = result.redacted_text   # send to cloud LLM
        spans     = result.spans           # store for audit trail

    For in-process use (no cf-orch), pass a model path and it loads directly.
    For service-backed use, see PIIFilter.from_backend().
    """

    def __init__(self, backend: FilterBackend) -> None:
        self._backend = backend

    @classmethod
    def from_model(cls, model_path: str) -> "PIIFilter":
        """Load a classifier model in-process (no cf-orch required)."""
        return cls(make_classifier_backend(model_path))

    @classmethod
    def from_backend(cls, backend: FilterBackend) -> "PIIFilter":
        """Wrap an already-constructed FilterBackend."""
        return cls(backend)

    def filter(self, text: str) -> FilterResult:
        """Synchronous filter — blocks until classification is complete."""
        raw = self._backend.classify(text)
        spans = _spans_from_pipeline(raw)
        return FilterResult(
            redacted_text=_redact(text, spans),
            spans=spans,
            original_text=text,
        )

    async def filter_async(self, text: str) -> FilterResult:
        """Async filter — runs classifier in thread pool."""
        raw = await self._backend.classify_async(text)
        spans = _spans_from_pipeline(raw)
        return FilterResult(
            redacted_text=_redact(text, spans),
            spans=spans,
            original_text=text,
        )
