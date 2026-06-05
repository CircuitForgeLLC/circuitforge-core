"""
circuitforge_core.reranker — shared reranker module for RAG pipelines.

Provides a modality-aware scoring interface for ranking candidates against a
query. Built to handle text today and audio/image/video in future branches.

Architecture:

    Reranker (Protocol / trunk)
    └── TextReranker (branch)
        ├── MockTextReranker   — no deps, deterministic, for tests
        ├── BGETextReranker    — FlagEmbedding cross-encoder, MIT, Free tier
        └── Qwen3TextReranker  — generative reranker, MIT/BSL, Paid tier (Phase 2)

Quick start (mock mode — no model required):

    import os; os.environ["CF_RERANKER_MOCK"] = "1"
    from circuitforge_core.reranker import rerank

    results = rerank("chicken soup", ["hearty chicken noodle", "chocolate cake", "tomato basil soup"])
    for r in results:
        print(r.rank, r.score, r.candidate[:40])

Real inference (BGE cross-encoder):

    export CF_RERANKER_MODEL=BAAI/bge-reranker-base
    from circuitforge_core.reranker import rerank
    results = rerank(query, candidates, top_n=20)

Explicit backend (per-request or per-user):

    from circuitforge_core.reranker import make_reranker
    reranker = make_reranker("BAAI/bge-reranker-v2-m3", backend="bge")
    results = reranker.rerank(query, candidates, top_n=10)

Batch scoring (efficient for large corpora):

    from circuitforge_core.reranker import make_reranker
    reranker = make_reranker("BAAI/bge-reranker-base")
    batch = reranker.rerank_batch(queries, candidate_lists, top_n=10)

Environment variables:
    CF_RERANKER_MODEL   model ID or path (default: "BAAI/bge-reranker-base")
    CF_RERANKER_BACKEND backend override: "bge" | "mock" (default: auto-detect)
    CF_RERANKER_MOCK    set to "1" to force mock backend (no model required)

cf-orch service profile (Phase 3 — remote backend):
    service_type:   cf-reranker
    max_mb:         per-model (base ≈ 600, large ≈ 1400, 8B ≈ 8192)
    shared:         true
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)

from circuitforge_core.reranker.base import RerankResult, Reranker, TextReranker
from circuitforge_core.reranker.adapters.mock import MockTextReranker

# ── Process-level singleton ───────────────────────────────────────────────────

_reranker: TextReranker | None = None

_DEFAULT_MODEL = "BAAI/bge-reranker-base"


def _get_reranker() -> TextReranker:
    global _reranker
    if _reranker is None:
        _reranker = make_reranker()
    return _reranker


def make_reranker(
    model_id: str | None = None,
    backend: str | None = None,
    mock: bool | None = None,
) -> TextReranker:
    """
    Create a TextReranker for the given model.

    Use this when you need an explicit reranker instance (e.g. per-service
    with a specific model) rather than the process-level singleton.

    model_id  — HuggingFace model ID or local path. Defaults to
                CF_RERANKER_MODEL env var, then BAAI/bge-reranker-base.
    backend   — "bge" | "mock". Auto-detected from model_id if omitted.
    mock      — Force mock backend. Defaults to CF_RERANKER_MOCK env var.
    """
    _mock = mock if mock is not None else os.environ.get("CF_RERANKER_MOCK", "") == "1"
    if _mock:
        return MockTextReranker()

    _model_id = model_id or os.environ.get("CF_RERANKER_MODEL", _DEFAULT_MODEL)
    _backend = backend or os.environ.get("CF_RERANKER_BACKEND", "")

    # Auto-route to cf-orch when CF_ORCH_URL is set and no explicit backend override.
    # Cloud deployments set CF_ORCH_URL; local dev leaves it unset → local inference.
    if not _backend:
        orch_url = os.environ.get("CF_ORCH_URL", "")
        if orch_url:
            from circuitforge_core.reranker.adapters.remote import RemoteTextReranker
            logger.info("[reranker] CF_ORCH_URL set — using remote cf-reranker via cf-orch")
            return RemoteTextReranker.from_cf_orch(
                orch_url=orch_url,
                service="cf-reranker",
                ttl_s=float(os.environ.get("CF_RERANKER_TTL", "3600")),
            )
        _backend = "bge"  # local default

    if _backend == "mock":
        return MockTextReranker()

    if _backend == "bge":
        from circuitforge_core.reranker.adapters.bge import BGETextReranker
        return BGETextReranker(_model_id)

    if _backend == "qwen3":
        from circuitforge_core.reranker.adapters.qwen3 import Qwen3TextReranker
        return Qwen3TextReranker(_model_id)

    if _backend == "cross-encoder":
        from circuitforge_core.reranker.adapters.cross_encoder import CrossEncoderTextReranker
        return CrossEncoderTextReranker(_model_id)

    if _backend == "cohere":
        from circuitforge_core.reranker.adapters.cohere import CohereTextReranker
        return CohereTextReranker(model=_model_id)

    if _backend == "remote":
        from circuitforge_core.reranker.adapters.remote import RemoteTextReranker
        return RemoteTextReranker(_model_id)

    raise ValueError(
        f"Unknown reranker backend {_backend!r}. "
        "Valid options: 'bge', 'qwen3', 'cross-encoder', 'cohere', 'remote', 'mock'."
    )


# ── Convenience functions (singleton path) ────────────────────────────────────


def rerank(
    query: str,
    candidates: Sequence[str],
    top_n: int = 0,
) -> list[RerankResult]:
    """
    Score and sort candidates against query using the process-level reranker.

    Returns a list of RerankResult sorted by score descending (rank 0 first).
    top_n=0 returns all candidates.

    For large corpora, prefer rerank_batch() on an explicit reranker instance
    to amortise model load time and batch the forward pass.
    """
    return _get_reranker().rerank(query, candidates, top_n=top_n)


def reset_reranker() -> None:
    """Reset the process-level singleton. Test teardown only."""
    global _reranker
    _reranker = None


__all__ = [
    "Reranker",
    "TextReranker",
    "RerankResult",
    "MockTextReranker",
    "make_reranker",
    "rerank",
    "reset_reranker",
]
