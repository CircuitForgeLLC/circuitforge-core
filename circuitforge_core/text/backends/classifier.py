# circuitforge_core/text/backends/classifier.py — HuggingFace token-classification backend
#
# BSL 1.1. Requires torch + transformers.
# Install: pip install circuitforge-core[text-transformers]
#
# Wraps pipeline("token-classification") for PII/entity detection.
# Returns spans with char offsets, entity labels, and confidence scores.
# Use make_classifier_backend() from base.py to instantiate.
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class ClassifierBackend:
    """
    HuggingFace token-classification backend for PII detection and entity labeling.

    Loads any token-classification model from HuggingFace Hub or a local checkpoint.
    Returns aggregated entity spans with char offsets — suitable for redaction or audit.

    Aggregation strategy "simple" merges consecutive BIO-tagged subwords into word-level
    spans and strips the B-/I- prefixes so callers see "NAME" not "B-NAME".

    Requires: pip install circuitforge-core[text-transformers]
    """

    def __init__(self, model_path: str) -> None:
        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for ClassifierBackend. "
                "Install with: pip install circuitforge-core[text-transformers]"
            ) from exc

        device = 0 if torch.cuda.is_available() else -1
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            device = 0

        logger.info("Loading classifier model %s on device %s", model_path, device)

        self._pipeline = hf_pipeline(
            "token-classification",
            model=model_path,
            aggregation_strategy="simple",
            device=device,
        )
        self._model_path = model_path

    @property
    def model_name(self) -> str:
        return self._model_path.split("/")[-1]

    @property
    def vram_mb(self) -> int:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() // (1024 * 1024)
        except Exception:
            pass
        return 0

    def classify(self, text: str) -> list[dict[str, Any]]:
        """
        Run token classification synchronously.

        Returns a list of entity dicts with keys:
          entity_group: str   — label without BIO prefix (e.g. "NAME", "EMAIL")
          score: float        — aggregated confidence
          word: str           — matched text span
          start: int          — char offset (start, inclusive)
          end: int            — char offset (end, exclusive)
        """
        results: list[dict[str, Any]] = self._pipeline(text)
        return results

    async def classify_async(self, text: str) -> list[dict[str, Any]]:
        """Async classify — runs pipeline in thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.classify, text)
