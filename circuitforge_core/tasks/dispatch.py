# circuitforge_core/tasks/dispatch.py
"""
Generic caller/args task dispatch — cf-core #67.

Products (e.g. pagepiper) were already importing `dispatch_task`/
`get_task_status` from `circuitforge_core.tasks` expecting a
"product/task_name" + kwargs-dict interface, distinct from the VRAM-budgeted
`TaskScheduler` in `scheduler.py` (which is keyed by task_id/job_id/params
against a specific SQLite `background_tasks` table). Neither function
existed, so every call silently hit the `except Exception` fallback.

This is the free-tier (in-process, single-node) implementation: a runnable
is registered under a name once at product startup, then dispatched by that
name any number of times. There is no coordinator here — cross-node
distribution via the separate circuitforge-orch package would need a new
generic task-dispatch endpoint on that coordinator; this module doesn't
attempt that, and is written so a future coordinator-backed implementation
can slot in behind the same two function signatures.

Usage::

    from circuitforge_core.tasks import register_task_runner, dispatch_task, get_task_status

    register_task_runner("pagepiper/ingest_pdf", run_ingest_pdf)

    task_id = dispatch_task("pagepiper/ingest_pdf", {"doc_id": "...", "file_path": "..."})
    get_task_status(task_id)  # {"status": "running", "progress": 0, "error": None}
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

_registry_lock = threading.Lock()
_task_runners: dict[str, Callable[..., None]] = {}

_status_lock = threading.Lock()
_task_status: dict[str, dict[str, Any]] = {}

_executor_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(thread_name_prefix="cf-task-dispatch")
        return _executor


def register_task_runner(caller: str, fn: Callable[..., None]) -> None:
    """
    Register a runnable under `caller` (e.g. "pagepiper/ingest_pdf").

    Must be called once at product startup before dispatch_task() is used
    with the same `caller` string. `fn` is called as `fn(**args)`.
    """
    with _registry_lock:
        _task_runners[caller] = fn


def unregister_task_runner(caller: str) -> None:
    """Remove a registration. TEST TEARDOWN ONLY."""
    with _registry_lock:
        _task_runners.pop(caller, None)


def dispatch_task(caller: str, args: dict[str, Any]) -> str:
    """
    Dispatch the runnable registered under `caller` with `args` as kwargs,
    running it on a background thread. Returns immediately with a task_id.

    Raises LookupError if no runnable is registered for `caller` — callers
    that assume this always routes through a coordinator (the pattern
    pagepiper's `_dispatch_ingest` uses) should catch and fall back, exactly
    as they already do for the ImportError this used to raise.
    """
    with _registry_lock:
        fn = _task_runners.get(caller)
    if fn is None:
        raise LookupError(
            f"No task runner registered for {caller!r}. "
            "Call register_task_runner() before dispatch_task()."
        )

    task_id = str(uuid4())
    with _status_lock:
        _task_status[task_id] = {"status": "queued", "progress": 0, "error": None}

    def _run() -> None:
        with _status_lock:
            _task_status[task_id] = {"status": "running", "progress": 0, "error": None}
        try:
            fn(**args)
        except Exception as exc:
            logger.exception("Task %s (%s) failed", task_id, caller)
            with _status_lock:
                _task_status[task_id] = {"status": "error", "progress": None, "error": str(exc)}
        else:
            with _status_lock:
                _task_status[task_id] = {"status": "complete", "progress": 100, "error": None}

    _get_executor().submit(_run)
    return task_id


def get_task_status(task_id: str) -> dict[str, Any]:
    """
    Return `{"status", "progress", "error"}` for `task_id`.

    Raises KeyError if `task_id` is unknown — never dispatched, or the
    process restarted since (status is in-memory only, not persisted).
    """
    with _status_lock:
        status = _task_status.get(task_id)
    if status is None:
        raise KeyError(f"Unknown task_id: {task_id!r}")
    return dict(status)


def reset_dispatch_registry() -> None:
    """Clear all registrations and status, and shut down the executor. TEST TEARDOWN ONLY."""
    global _executor
    with _registry_lock:
        _task_runners.clear()
    with _status_lock:
        _task_status.clear()
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=True)
            _executor = None
