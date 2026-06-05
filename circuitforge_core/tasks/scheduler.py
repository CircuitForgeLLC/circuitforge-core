# circuitforge_core/tasks/scheduler.py
"""Task scheduler for CircuitForge products — MIT layer.

Provides a simple FIFO task queue with no coordinator dependency.

For coordinator-aware VRAM-budgeted scheduling on paid/premium tiers, install
circuitforge-orch and use OrchestratedScheduler instead.

Public API:
    TaskScheduler       — Protocol defining the scheduler interface
    LocalScheduler      — Simple FIFO queue implementation (MIT, no coordinator)
    detect_available_vram_gb() — Returns 999.0 (unlimited; no coordinator on free tier)
    get_scheduler()     — Lazy process-level singleton returning a LocalScheduler
    reset_scheduler()   — Test teardown only
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from collections import deque
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

RunTaskFn = Callable[["Path", int, str, int, Optional[str]], None]


class TaskSpec(NamedTuple):
    id: int
    job_id: int
    params: Optional[str]


_DEFAULT_MAX_QUEUE_DEPTH = 500


def detect_available_vram_gb() -> float:
    """Return available VRAM for task scheduling.

    Free tier (no coordinator): always returns 999.0 — no VRAM gating.
    For coordinator-aware VRAM detection use circuitforge_orch.scheduler.
    """
    return 999.0


@runtime_checkable
class TaskScheduler(Protocol):
    """Protocol for task schedulers across free and paid tiers.

    Both LocalScheduler (MIT) and OrchestratedScheduler (BSL, circuitforge-orch)
    implement this interface so products can inject either without API changes.
    """

    def enqueue(self, task_id: int, task_type: str, job_id: int, params: Optional[str]) -> bool:
        """Add a task to the queue. Returns True if enqueued, False if queue full."""
        ...

    def start(self) -> None:
        """Start the background worker thread."""
        ...

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the scheduler and wait for it to exit."""
        ...


class LocalScheduler:
    """Simple FIFO task scheduler with no coordinator dependency.

    Processes tasks serially per task type. No VRAM gating — all tasks run.
    Suitable for free tier (single-host, up to 2 GPUs, static config).

    Usage::

        sched = get_scheduler(
            db_path=Path("data/app.db"),
            run_task_fn=my_run_task,
            task_types=frozenset({"cover_letter", "research"}),
            vram_budgets={"cover_letter": 2.5, "research": 5.0},
        )
        enqueued = sched.enqueue(task_id, "cover_letter", job_id, params_json)
    """

    def __init__(
        self,
        db_path: Path,
        run_task_fn: RunTaskFn,
        task_types: frozenset[str],
        vram_budgets: dict[str, float],
        max_queue_depth: int = _DEFAULT_MAX_QUEUE_DEPTH,
    ) -> None:
        self._db_path = db_path
        self._run_task = run_task_fn
        self._task_types = frozenset(task_types)
        self._budgets: dict[str, float] = dict(vram_budgets)
        self._max_queue_depth = max_queue_depth

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._queues: dict[str, deque[TaskSpec]] = {}
        self._active: dict[str, threading.Thread] = {}
        self._thread: Optional[threading.Thread] = None

        self._load_queued_tasks()

    def enqueue(self, task_id: int, task_type: str, job_id: int, params: Optional[str]) -> bool:
        with self._lock:
            q = self._queues.setdefault(task_type, deque())
            if len(q) >= self._max_queue_depth:
                logger.warning(
                    "Queue depth limit for %s (max=%d) — task %d dropped",
                    task_type, self._max_queue_depth, task_id,
                )
                return False
            q.append(TaskSpec(task_id, job_id, params))
        self._wake.set()
        return True

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._scheduler_loop, name="task-scheduler", daemon=True
        )
        self._thread.start()
        with self._lock:
            if any(self._queues.values()):
                self._wake.set()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        with self._lock:
            workers = list(self._active.values())
        for worker in workers:
            worker.join(timeout=timeout)

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=30)
            self._wake.clear()
            with self._lock:
                for t, thread in list(self._active.items()):
                    if not thread.is_alive():
                        del self._active[t]
                candidates = sorted(
                    [t for t in self._queues if self._queues[t] and t not in self._active],
                    key=lambda t: len(self._queues[t]),
                    reverse=True,
                )
                for task_type in candidates:
                    thread = threading.Thread(
                        target=self._batch_worker,
                        args=(task_type,),
                        name=f"batch-{task_type}",
                        daemon=True,
                    )
                    self._active[task_type] = thread
                    thread.start()

    def _batch_worker(self, task_type: str) -> None:
        try:
            while True:
                with self._lock:
                    q = self._queues.get(task_type)
                    if not q:
                        break
                    task = q.popleft()
                try:
                    self._run_task(self._db_path, task.id, task_type, task.job_id, task.params)
                except Exception as exc:
                    # run_task_fn should handle its own exceptions. If it leaks one,
                    # log it so the task doesn't silently stay 'queued' with no trace.
                    logger.exception(
                        "Unhandled exception in batch worker task %d (%s): %s",
                        task.id, task_type, exc,
                    )
        finally:
            with self._lock:
                self._active.pop(task_type, None)
            self._wake.set()

    def _load_queued_tasks(self) -> None:
        if not self._task_types:
            return
        task_types_list = sorted(self._task_types)
        placeholders = ",".join("?" * len(task_types_list))
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    f"SELECT id, task_type, job_id, params FROM background_tasks"
                    f" WHERE status='queued' AND task_type IN ({placeholders})"
                    f" ORDER BY created_at ASC",
                    task_types_list,
                ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        for row_id, task_type, job_id, params in rows:
            q = self._queues.setdefault(task_type, deque())
            q.append(TaskSpec(row_id, job_id, params))
        if rows:
            logger.info("Scheduler: resumed %d queued task(s) from prior run", len(rows))


# ── Process-level singleton ────────────────────────────────────────────────────

_scheduler: Optional[LocalScheduler] = None
_scheduler_lock = threading.Lock()


def get_scheduler(
    db_path: Optional[Path] = None,
    run_task_fn: Optional[RunTaskFn] = None,
    task_types: Optional[frozenset[str]] = None,
    vram_budgets: Optional[dict[str, float]] = None,
    max_queue_depth: int = _DEFAULT_MAX_QUEUE_DEPTH,
    coordinator_url: str = "http://localhost:7700",
    service_name: str = "peregrine",
) -> LocalScheduler:
    """Return the process-level LocalScheduler singleton.

    ``run_task_fn``, ``task_types``, ``vram_budgets``, and ``db_path`` are
    required on the first call; ignored on subsequent calls.

    ``coordinator_url`` and ``service_name`` are accepted but ignored —
    LocalScheduler has no coordinator. They exist for API compatibility with
    OrchestratedScheduler call sites.
    """
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    if run_task_fn is None or task_types is None or vram_budgets is None or db_path is None:
        raise ValueError(
            "db_path, run_task_fn, task_types, and vram_budgets are required "
            "on the first call to get_scheduler()"
        )
    candidate = LocalScheduler(
        db_path=db_path,
        run_task_fn=run_task_fn,
        task_types=task_types,
        vram_budgets=vram_budgets,
        max_queue_depth=max_queue_depth,
    )
    candidate.start()
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = candidate
        else:
            candidate.shutdown()
    return _scheduler


def reset_scheduler() -> None:
    """Shut down and clear the singleton. TEST TEARDOWN ONLY."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None:
            _scheduler.shutdown()
            _scheduler = None
