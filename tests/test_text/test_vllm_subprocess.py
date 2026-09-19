"""Tests for VllmSubprocessSupervisor — spawns and health-polls a vLLM child process."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.text.backends.vllm_subprocess import VllmSubprocessSupervisor


def _mock_health_response(status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = status_code == 200
    return resp


class TestVllmSubprocessSupervisorSpawn:
    def test_spawns_with_correct_model_and_port(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen") as mock_popen, \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            url = sup.ensure_running()

        assert url == "http://localhost:8300"
        args = mock_popen.call_args.args[0]
        assert args[0] == "/devl/miniconda3/envs/cf-vllm/bin/python"
        assert "-m" in args and "vllm.entrypoints.openai.api_server" in args
        assert "--model" in args
        assert args[args.index("--model") + 1] == "IFM/K2-Horizon-7B"
        assert "--port" in args
        assert args[args.index("--port") + 1] == "8300"

    def test_default_port_is_8300(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen"), patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B")
            url = sup.ensure_running()
        assert url == "http://localhost:8300"

    def test_uses_dtype_and_gpu_memory_utilization(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen") as mock_popen, \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor(
                "IFM/K2-Horizon-7B", port=8300, gpu_memory_utilization=0.85, dtype="bfloat16",
            )
            sup.ensure_running()
        args = mock_popen.call_args.args[0]
        assert args[args.index("--gpu-memory-utilization") + 1] == "0.85"
        assert args[args.index("--dtype") + 1] == "bfloat16"

    def test_always_targets_cuda_device_0(self, monkeypatch):
        """The child process always sees its assigned GPU as cuda:0 -- cf-orch
        already narrows visibility via CUDA_VISIBLE_DEVICES on cf-text's own
        process before this supervisor ever runs."""
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen") as mock_popen, \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
        args = mock_popen.call_args.args[0]
        assert args[args.index("--device") + 1] == "cuda:0"

    def test_missing_python_path_raises(self, monkeypatch):
        monkeypatch.delenv("CF_TEXT_VLLM_PYTHON", raising=False)
        sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
        with pytest.raises(RuntimeError, match="CF_TEXT_VLLM_PYTHON"):
            sup.ensure_running()


class TestVllmSubprocessSupervisorHealthPoll:
    def test_polls_until_healthy(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        responses = [_mock_health_response(503), _mock_health_response(503), _mock_health_response(200)]
        with patch("subprocess.Popen"), \
             patch("httpx.get", side_effect=responses) as mock_get, \
             patch("time.sleep") as mock_sleep:
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300, poll_interval_s=0.01)
            url = sup.ensure_running()
        assert url == "http://localhost:8300"
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    def test_health_check_hits_correct_url(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen"), \
             patch("httpx.get", return_value=_mock_health_response()) as mock_get:
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
        mock_get.assert_called_with("http://localhost:8300/health", timeout=2.0)

    def test_timeout_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen"), \
             patch("httpx.get", return_value=_mock_health_response(503)), \
             patch("time.sleep"), \
             patch("time.monotonic", side_effect=[0.0, 0.0, 400.0]):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300, health_timeout_s=300.0)
            with pytest.raises(RuntimeError, match="did not become healthy"):
                sup.ensure_running()

    def test_connection_error_is_treated_as_not_ready_yet(self, monkeypatch):
        """httpx raising (connection refused -- server not listening yet) must not
        crash the poll loop, only a real 200 response ends it."""
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        import httpx as _httpx
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen"), \
             patch("httpx.get", side_effect=[_httpx.ConnectError("refused"), _mock_health_response(200)]), \
             patch("time.sleep"):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            url = sup.ensure_running()
        assert url == "http://localhost:8300"


class TestVllmSubprocessSupervisorLifecycle:
    def test_ensure_running_is_idempotent(self, monkeypatch):
        """A second call while the process is already running must not spawn twice."""
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen") as mock_popen, \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
            sup.ensure_running()
        assert mock_popen.call_count == 1

    def test_terminate_stops_the_process(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
            sup.terminate()
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()

    def test_terminate_kills_if_terminate_times_out(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        import subprocess as _subprocess
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = _subprocess.TimeoutExpired(cmd="vllm", timeout=10.0)
        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
            sup.terminate()
        mock_proc.kill.assert_called_once()

    def test_terminate_before_ensure_running_is_a_noop(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
        sup.terminate()  # must not raise

    def test_spawn_registers_atexit_cleanup(self, monkeypatch):
        """Orphaned vLLM child processes are the exact failure mode found in
        cf-orch#112 (a coordinator process squatting on a port for a month
        because nothing ever cleaned it up). ensure_running() must register
        atexit cleanup so a killed cf-text process cannot leave vLLM behind."""
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen"), \
             patch("httpx.get", return_value=_mock_health_response()), \
             patch("atexit.register") as mock_atexit:
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
        mock_atexit.assert_called_once_with(sup.terminate)
