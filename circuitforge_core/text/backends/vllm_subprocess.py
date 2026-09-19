"""VllmSubprocessSupervisor -- spawns and health-supervises a vLLM child process.

cf-text's VllmBackend (vllm.py) proxies OpenAI-compatible requests to a vLLM
(VRAM-efficient LLM inference engine) server, but assumes one is already
running. This supervisor is what actually starts one, as a child process of
cf-text itself, using the isolated cf-vllm conda environment (vLLM hard-pins
a torch version incompatible with cf-text's own dependencies).

Environment:
  CF_TEXT_VLLM_PYTHON   Path to the cf-vllm environment's python. Required
                         when a subprocess actually needs spawning.
  CF_TEXT_VLLM_PORT     Fixed internal port for the spawned server (default
                         8300 -- distinct from the standalone vllm service's
                         8200 and cf-text's own 8008/8009 range).

MIT licensed.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import time

import httpx

_DEFAULT_PORT = 8300


class VllmSubprocessSupervisor:
    """Spawns `vllm.entrypoints.openai.api_server` as a child process and
    waits for it to report healthy.

    Not used when CF_TEXT_VLLM_URL is already set in the environment --
    that means an externally-managed vLLM server is in play and VllmBackend
    proxies to it directly, unchanged from before this class existed.
    """

    def __init__(
        self,
        model_id: str,
        *,
        python_path: str | None = None,
        port: int | None = None,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "float16",
        health_timeout_s: float = 300.0,
        poll_interval_s: float = 1.0,
    ) -> None:
        self._model_id = model_id
        self._python_path = python_path or os.environ.get("CF_TEXT_VLLM_PYTHON", "")
        self._port = port or int(os.environ.get("CF_TEXT_VLLM_PORT", _DEFAULT_PORT))
        self._gpu_memory_utilization = gpu_memory_utilization
        self._dtype = dtype
        self._health_timeout_s = health_timeout_s
        self._poll_interval_s = poll_interval_s
        self._proc: subprocess.Popen | None = None

    @property
    def _base_url(self) -> str:
        return f"http://localhost:{self._port}"

    def ensure_running(self) -> str:
        """Spawn the vLLM subprocess if not already running, wait for /health
        to succeed, and return the base URL. Idempotent: a second call while
        the process is already up returns immediately without spawning again.
        """
        if self._proc is None:
            self._spawn()
        self._wait_for_health()
        return self._base_url

    def _spawn(self) -> None:
        if not self._python_path:
            raise RuntimeError(
                "CF_TEXT_VLLM_PYTHON is not set -- cannot spawn a vLLM subprocess. "
                "Set it to the cf-vllm conda environment's python path."
            )
        args = [
            self._python_path,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", self._model_id,
            "--port", str(self._port),
            "--gpu-memory-utilization", str(self._gpu_memory_utilization),
            "--dtype", self._dtype,
            # cf-orch already narrows CUDA_VISIBLE_DEVICES on cf-text's own
            # process before this ever runs, so the assigned GPU is always
            # device 0 from here.
            "--device", "cuda:0",
        ]
        self._proc = subprocess.Popen(args)
        # Orphaned child processes are how cf-orch#112 happened (a coordinator
        # process squatted on a port for a month because nothing ever cleaned
        # it up). Register cleanup so a killed cf-text process cannot leave
        # this vLLM child running behind it.
        atexit.register(self.terminate)

    def _wait_for_health(self) -> None:
        deadline = time.monotonic() + self._health_timeout_s
        url = f"{self._base_url}/health"
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(self._poll_interval_s)
        raise RuntimeError(
            f"vLLM subprocess for {self._model_id!r} did not become healthy "
            f"within {self._health_timeout_s}s (checked {url})"
        )

    def terminate(self) -> None:
        """Stop the child process, if one was spawned. Safe to call even if
        ensure_running() was never called."""
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
