"""Tests for VllmSubprocessSupervisor -- spawns and health-polls a vLLM child process."""
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


def _mock_alive_proc() -> MagicMock:
    """A mock subprocess.Popen() return value whose poll() reports the process
    is still running (None), matching real Popen behavior for a live child.
    Needed for the _wait_for_health dead-process check to not trip on tests
    that are not exercising the crash path."""
    proc = MagicMock()
    proc.poll.return_value = None
    return proc


class TestVllmSubprocessSupervisorSpawn:
    def test_spawns_with_correct_model_and_port(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen", return_value=_mock_alive_proc()) as mock_popen, \
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
        with patch("subprocess.Popen", return_value=_mock_alive_proc()), patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B")
            url = sup.ensure_running()
        assert url == "http://localhost:8300"

    def test_uses_dtype_and_gpu_memory_utilization(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen", return_value=_mock_alive_proc()) as mock_popen, \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor(
                "IFM/K2-Horizon-7B", port=8300, gpu_memory_utilization=0.85, dtype="bfloat16",
            )
            sup.ensure_running()
        args = mock_popen.call_args.args[0]
        assert args[args.index("--gpu-memory-utilization") + 1] == "0.85"
        assert args[args.index("--dtype") + 1] == "bfloat16"

    def test_gpu_mem_util_env_override_used_when_not_passed_explicitly(self, monkeypatch):
        """A node whose GPU isn't fully free at baseline (found on Muninn's
        RTX 3090 -- ~2.6 GB reserved outside any single nvidia-smi
        compute-app entry) needs a lower ceiling than the 0.90 default."""
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        monkeypatch.setenv("CF_TEXT_VLLM_GPU_MEM_UTIL", "0.80")
        with patch("subprocess.Popen", return_value=_mock_alive_proc()) as mock_popen, \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
        args = mock_popen.call_args.args[0]
        assert args[args.index("--gpu-memory-utilization") + 1] == "0.8"

    def test_explicit_gpu_mem_util_wins_over_env(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        monkeypatch.setenv("CF_TEXT_VLLM_GPU_MEM_UTIL", "0.80")
        with patch("subprocess.Popen", return_value=_mock_alive_proc()) as mock_popen, \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300, gpu_memory_utilization=0.85)
            sup.ensure_running()
        args = mock_popen.call_args.args[0]
        assert args[args.index("--gpu-memory-utilization") + 1] == "0.85"

    def test_default_gpu_mem_util_is_090_when_nothing_set(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        monkeypatch.delenv("CF_TEXT_VLLM_GPU_MEM_UTIL", raising=False)
        with patch("subprocess.Popen", return_value=_mock_alive_proc()) as mock_popen, \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
        args = mock_popen.call_args.args[0]
        assert args[args.index("--gpu-memory-utilization") + 1] == "0.9"

    def test_does_not_pass_a_device_flag(self, monkeypatch):
        """vllm's api_server has no --device flag (confirmed against the
        installed vllm 0.19.1: passing one is a hard argparse error). GPU
        targeting relies entirely on CUDA_VISIBLE_DEVICES, which cf-orch
        already narrows on cf-text's own process before this supervisor
        ever runs, and which subprocess.Popen inherits by default."""
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen", return_value=_mock_alive_proc()) as mock_popen, \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
        args = mock_popen.call_args.args[0]
        assert "--device" not in args
        # No env= kwarg passed to Popen means the child inherits the parent's
        # environment (including CUDA_VISIBLE_DEVICES) unmodified.
        assert "env" not in mock_popen.call_args.kwargs

    def test_missing_python_path_raises(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.delenv("CF_TEXT_VLLM_PYTHON", raising=False)
        sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
        with pytest.raises(RuntimeError, match="CF_TEXT_VLLM_PYTHON"):
            sup.ensure_running()


class TestVllmSubprocessSupervisorHealthPoll:
    def test_polls_until_healthy(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        responses = [_mock_health_response(503), _mock_health_response(503), _mock_health_response(200)]
        with patch("subprocess.Popen", return_value=_mock_alive_proc()), \
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
        with patch("subprocess.Popen", return_value=_mock_alive_proc()), \
             patch("httpx.get", return_value=_mock_health_response()) as mock_get:
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
        mock_get.assert_called_with("http://localhost:8300/health", timeout=2.0)

    def test_timeout_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen", return_value=_mock_alive_proc()), \
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
        with patch("subprocess.Popen", return_value=_mock_alive_proc()), \
             patch("httpx.get", side_effect=[_httpx.ConnectError("refused"), _mock_health_response(200)]), \
             patch("time.sleep"):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            url = sup.ensure_running()
        assert url == "http://localhost:8300"

    def test_dead_process_fails_fast_with_exit_code(self, monkeypatch):
        """An instant-exit crash (bad CLI flag, CUDA OOM, etc.) must be detected via
        the child process's own exit code instead of burning the full health_timeout_s
        polling a port nothing is listening on."""
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("httpx.get") as mock_get, \
             patch("time.sleep") as mock_sleep:
            sup = VllmSubprocessSupervisor(
                "IFM/K2-Horizon-7B", port=8300, health_timeout_s=300.0
            )
            with pytest.raises(RuntimeError, match="exited with code 1"):
                sup.ensure_running()
        mock_get.assert_not_called()
        mock_sleep.assert_not_called()


class TestVllmSubprocessSupervisorLifecycle:
    def test_ensure_running_is_idempotent(self, monkeypatch):
        """A second call while the process is already running must not spawn twice."""
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        with patch("subprocess.Popen", return_value=_mock_alive_proc()) as mock_popen, \
             patch("httpx.get", return_value=_mock_health_response()):
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
            sup.ensure_running()
        assert mock_popen.call_count == 1

    def test_terminate_stops_the_process(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda *a, **k: None)
        monkeypatch.setenv("CF_TEXT_VLLM_PYTHON", "/devl/miniconda3/envs/cf-vllm/bin/python")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
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
        mock_proc.poll.return_value = None
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
        with patch("subprocess.Popen", return_value=_mock_alive_proc()), \
             patch("httpx.get", return_value=_mock_health_response()), \
             patch("atexit.register") as mock_atexit:
            sup = VllmSubprocessSupervisor("IFM/K2-Horizon-7B", port=8300)
            sup.ensure_running()
        mock_atexit.assert_called_once_with(sup.terminate)
