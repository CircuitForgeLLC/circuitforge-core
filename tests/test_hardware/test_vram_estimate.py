"""Tests for circuitforge_core.hardware.vram_estimate (cf-core #64)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.hardware.vram_estimate import (
    ModelVramLookupError,
    model_vram_estimate,
)

_QWEN_CONFIG = {
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
}


def _api_response(status_code=200, safetensors_total=7_000_000_000):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"safetensors": {"total": safetensors_total}} if safetensors_total else {}
    return resp


def _config_response(status_code=200, config=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = config if config is not None else _QWEN_CONFIG
    return resp


class TestModelVramEstimate:
    def test_estimates_weights_gb_for_fp16(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=[_api_response(), _config_response()],
        ):
            est = model_vram_estimate("Qwen/Qwen2.5-7B-Instruct", "fp16")

        assert est.params_billions == pytest.approx(7.0)
        # 7e9 params * 16 bits / 8 bits-per-byte / 1e9 = 14 GB
        assert est.weights_gb == pytest.approx(14.0)

    def test_lower_bit_quant_uses_less_vram(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=[_api_response(), _config_response()],
        ):
            fp16 = model_vram_estimate("Qwen/Qwen2.5-7B-Instruct", "fp16")
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=[_api_response(), _config_response()],
        ):
            q4 = model_vram_estimate("Qwen/Qwen2.5-7B-Instruct", "q4_k_m")

        assert q4.weights_gb < fp16.weights_gb

    def test_kv_cache_included_when_config_available(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=[_api_response(), _config_response()],
        ):
            est = model_vram_estimate("Qwen/Qwen2.5-7B-Instruct", "fp16", ctx_len=8192)

        assert est.kv_cache_gb > 0.0
        assert est.total_vram_gb == pytest.approx(
            est.weights_gb + est.kv_cache_gb + est.overhead_gb
        )

    def test_kv_cache_zero_when_config_missing_arch_fields(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=[_api_response(), _config_response(config={"some_other_field": 1})],
        ):
            est = model_vram_estimate("weird/model", "fp16")

        assert est.kv_cache_gb == 0.0

    def test_kv_cache_zero_when_config_fetch_fails(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=[_api_response(), _config_response(status_code=404)],
        ):
            est = model_vram_estimate("Qwen/Qwen2.5-7B-Instruct", "fp16")

        assert est.kv_cache_gb == 0.0

    def test_fits_true_when_vram_sufficient(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=[_api_response(), _config_response()],
        ):
            est = model_vram_estimate(
                "Qwen/Qwen2.5-7B-Instruct", "q4_k_m", available_vram_mb=24_000
            )

        assert est.fits is True

    def test_fits_false_when_vram_insufficient(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=[_api_response(), _config_response()],
        ):
            est = model_vram_estimate(
                "Qwen/Qwen2.5-7B-Instruct", "fp32", available_vram_mb=4_000
            )

        assert est.fits is False

    def test_fits_none_when_available_vram_not_supplied(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=[_api_response(), _config_response()],
        ):
            est = model_vram_estimate("Qwen/Qwen2.5-7B-Instruct", "fp16")

        assert est.fits is None

    def test_unknown_quant_level_raises_value_error(self):
        with pytest.raises(ValueError):
            model_vram_estimate("Qwen/Qwen2.5-7B-Instruct", "not-a-real-quant")

    def test_quant_level_case_insensitive(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=[_api_response(), _config_response()],
        ):
            est = model_vram_estimate("Qwen/Qwen2.5-7B-Instruct", "FP16")

        assert est.weights_gb == pytest.approx(14.0)

    def test_raises_on_non_200_from_hf_api(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            return_value=_api_response(status_code=404),
        ):
            with pytest.raises(ModelVramLookupError):
                model_vram_estimate("nonexistent/model", "fp16")

    def test_raises_on_missing_safetensors_metadata(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            return_value=_api_response(safetensors_total=None),
        ):
            with pytest.raises(ModelVramLookupError):
                model_vram_estimate("gguf-only/model", "fp16")

    def test_raises_on_request_exception(self):
        import requests

        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=requests.ConnectionError("no network"),
        ):
            with pytest.raises(ModelVramLookupError):
                model_vram_estimate("Qwen/Qwen2.5-7B-Instruct", "fp16")

    def test_result_includes_hf_model_id_and_quant_level(self):
        with patch(
            "circuitforge_core.hardware.vram_estimate.requests.get",
            side_effect=[_api_response(), _config_response()],
        ):
            est = model_vram_estimate("Qwen/Qwen2.5-7B-Instruct", "q4_k_m")

        assert est.hf_model_id == "Qwen/Qwen2.5-7B-Instruct"
        assert est.quant_level == "q4_k_m"
