# circuitforge_core/hardware/vram_estimate.py
"""
Model VRAM fit estimation — cf-core #64.

`cf_core.hardware` can detect available VRAM but has no way to answer "can this
hardware run model X at quantization level Y?". This module closes that gap by
querying the HuggingFace Hub API for parameter count and architecture, then
applying the standard VRAM estimation formula:

    vram_gb = params * bytes_per_param(quant) + kv_cache_gb(ctx_len, arch) + overhead_gb

The formula is the reference algorithm used by LLMcalc
(https://github.com/Raskoll2/LLMcalc, no license — algorithm reference only,
not a dependency). No LLMcalc code is copied here.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

_HF_API_MODEL_URL = "https://huggingface.co/api/models/{model_id}"
_HF_CONFIG_URL = "https://huggingface.co/{model_id}/resolve/main/config.json"

_DEFAULT_OVERHEAD_GB = 0.6  # CUDA context + activation buffers, rough constant
_KV_CACHE_DTYPE_BYTES = 2  # KV cache is stored fp16 in the common case

# Bits per parameter for common quantization levels (weights-only, excludes KV cache).
_QUANT_BITS: dict[str, float] = {
    "fp32": 32.0,
    "fp16": 16.0,
    "bf16": 16.0,
    "int8": 8.0,
    "q8": 8.0,
    "q8_0": 8.0,
    "q6_k": 6.0,
    "q5_k_m": 5.0,
    "q5_0": 5.0,
    "int4": 4.0,
    "q4": 4.0,
    "q4_k_m": 4.0,
    "q4_0": 4.0,
    "q3_k_m": 3.0,
    "q2_k": 2.0,
}


class ModelVramLookupError(RuntimeError):
    """Raised when the HuggingFace Hub API can't supply data needed to estimate VRAM."""


@dataclass(frozen=True)
class VramEstimate:
    """Result of a `model_vram_estimate()` call."""

    hf_model_id: str
    quant_level: str
    params_billions: float
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_vram_gb: float
    fits: bool | None  # None when available_vram_mb wasn't supplied


def _quant_bits(quant_level: str) -> float:
    key = quant_level.strip().lower()
    try:
        return _QUANT_BITS[key]
    except KeyError:
        raise ValueError(
            f"Unknown quant_level {quant_level!r}. Known levels: {sorted(_QUANT_BITS)}"
        ) from None


def _fetch_param_count(hf_model_id: str, *, timeout: float) -> float:
    """Return total parameter count via the HF Hub API's safetensors metadata."""
    url = _HF_API_MODEL_URL.format(model_id=hf_model_id)
    try:
        resp = requests.get(url, params={"expand": ["safetensors"]}, timeout=timeout)
    except requests.RequestException as exc:
        raise ModelVramLookupError(
            f"HF Hub API request failed for {hf_model_id!r}: {exc}"
        ) from exc

    if resp.status_code != 200:
        raise ModelVramLookupError(
            f"HF Hub API returned {resp.status_code} for {hf_model_id!r}"
        )

    data = resp.json()
    total = (data.get("safetensors") or {}).get("total")
    if not total:
        raise ModelVramLookupError(
            f"{hf_model_id!r} has no safetensors parameter metadata on the HF Hub "
            "(model may not publish safetensors weights)"
        )
    return float(total)


def _fetch_arch_config(hf_model_id: str, *, timeout: float) -> dict:
    """Return the model's config.json (architecture fields used for KV cache sizing)."""
    url = _HF_CONFIG_URL.format(model_id=hf_model_id)
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise ModelVramLookupError(
            f"config.json fetch failed for {hf_model_id!r}: {exc}"
        ) from exc

    if resp.status_code != 200:
        raise ModelVramLookupError(
            f"config.json unavailable for {hf_model_id!r} (HTTP {resp.status_code})"
        )
    return resp.json()


def _kv_cache_gb(config: dict, ctx_len: int) -> float:
    """Estimate KV cache size in GB from architecture fields, 0.0 if unavailable."""
    num_layers = config.get("num_hidden_layers")
    hidden_size = config.get("hidden_size")
    num_heads = config.get("num_attention_heads")
    num_kv_heads = config.get("num_key_value_heads", num_heads)

    if not (num_layers and hidden_size and num_heads):
        # Non-standard config (missing architecture fields) — skip the KV
        # estimate rather than fail the whole call. Weights dominate VRAM use.
        return 0.0

    head_dim = hidden_size / num_heads
    bytes_total = 2 * num_layers * num_kv_heads * head_dim * ctx_len * _KV_CACHE_DTYPE_BYTES
    return bytes_total / 1e9


def model_vram_estimate(
    hf_model_id: str,
    quant_level: str,
    *,
    ctx_len: int = 4096,
    available_vram_mb: int | None = None,
    overhead_gb: float = _DEFAULT_OVERHEAD_GB,
    timeout: float = 10.0,
) -> VramEstimate:
    """
    Estimate VRAM required to run `hf_model_id` at `quant_level`.

    Args:
        hf_model_id: HuggingFace model repo ID, e.g. "Qwen/Qwen2.5-7B-Instruct".
        quant_level: One of the known quant levels (see `_QUANT_BITS`), e.g. "q4_k_m".
        ctx_len: Context length used for KV cache sizing.
        available_vram_mb: If given, populates `VramEstimate.fits`.
        overhead_gb: Fixed overhead for CUDA context / activation buffers.
        timeout: Per-request timeout in seconds for HF Hub API calls.

    Raises:
        ModelVramLookupError: HF Hub API request failed or returned unusable data.
        ValueError: `quant_level` isn't a recognized quantization level.
    """
    bits = _quant_bits(quant_level)
    params = _fetch_param_count(hf_model_id, timeout=timeout)
    weights_gb = (params * bits / 8) / 1e9

    try:
        config = _fetch_arch_config(hf_model_id, timeout=timeout)
        kv_gb = _kv_cache_gb(config, ctx_len)
    except ModelVramLookupError:
        # Architecture lookup is best-effort — weights_gb alone is still useful.
        kv_gb = 0.0

    total_gb = weights_gb + kv_gb + overhead_gb

    fits = None
    if available_vram_mb is not None:
        fits = total_gb <= (available_vram_mb / 1024)

    return VramEstimate(
        hf_model_id=hf_model_id,
        quant_level=quant_level,
        params_billions=params / 1e9,
        weights_gb=weights_gb,
        kv_cache_gb=kv_gb,
        overhead_gb=overhead_gb,
        total_vram_gb=total_gb,
        fits=fits,
    )
