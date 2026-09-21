"""Tests for TaskScheduler Protocol + LocalScheduler (MIT, no coordinator)."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from circuitforge_core.tasks.scheduler import (
    LocalScheduler,
    TaskScheduler,
    TaskSpec,
    detect_available_vram_gb,
    get_scheduler,
    reset_scheduler,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.db"
    with sqlite3.connect(p) as conn:
        conn.execute(
            "CREATE TABLE background_tasks "
            "(id INTEGER PRIMARY KEY, task_type TEXT, job_id INTEGER, "
            "params TEXT, status TEXT DEFAULT 'queued', created_at TEXT DEFAULT '')"
        )
    return p


@pytest.fixture(autouse=True)
def clean_singleton():
    yield
    reset_scheduler()


def make_run_fn(results: list):
    def run(db_path, task_id, task_type, job_id, params):
        results.append((task_type, task_id))
        time.sleep(0.01)
    return run


def test_local_scheduler_implements_protocol():
    assert isinstance(LocalScheduler.__new__(LocalScheduler), TaskScheduler)


def test_detect_available_vram_returns_unlimited():
    assert detect_available_vram_gb() == 999.0


def test_enqueue_and_execute(db_path):
    results = []
    sched = LocalScheduler(
        db_path=db_path,
        run_task_fn=make_run_fn(results),
        task_types=frozenset({"cover_letter"}),
        vram_budgets={"cover_letter": 0.0},
    )
    sched.start()
    sched.enqueue(1, "cover_letter", 1, None, db_path)
    time.sleep(0.3)
    sched.shutdown()
    assert ("cover_letter", 1) in results


def test_fifo_ordering(db_path):
    results = []
    sched = LocalScheduler(
        db_path=db_path,
        run_task_fn=make_run_fn(results),
        task_types=frozenset({"t"}),
        vram_budgets={"t": 0.0},
    )
    sched.start()
    sched.enqueue(1, "t", 1, None, db_path)
    sched.enqueue(2, "t", 1, None, db_path)
    sched.enqueue(3, "t", 1, None, db_path)
    time.sleep(0.5)
    sched.shutdown()
    assert [r[1] for r in results] == [1, 2, 3]


def test_queue_depth_limit(db_path):
    sched = LocalScheduler(
        db_path=db_path,
        run_task_fn=make_run_fn([]),
        task_types=frozenset({"t"}),
        vram_budgets={"t": 0.0},
        max_queue_depth=2,
    )
    assert sched.enqueue(1, "t", 1, None, db_path) is True
    assert sched.enqueue(2, "t", 1, None, db_path) is True
    assert sched.enqueue(3, "t", 1, None, db_path) is False


def test_get_scheduler_singleton(db_path):
    results = []
    s1 = get_scheduler(
        db_path=db_path,
        run_task_fn=make_run_fn(results),
        task_types=frozenset({"t"}),
        vram_budgets={"t": 0.0},
    )
    s2 = get_scheduler(db_path=db_path)
    assert s1 is s2
    s1.shutdown()


def test_local_scheduler_no_httpx_dependency():
    """LocalScheduler must not import httpx — not in MIT core's hard deps."""
    import ast, inspect
    from circuitforge_core.tasks import scheduler as sched_mod
    src = inspect.getsource(sched_mod)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in getattr(node, 'names', [])]
            module = getattr(node, 'module', '') or ''
            assert 'httpx' not in names and 'httpx' not in module, \
                "LocalScheduler must not import httpx"


def test_load_queued_tasks_on_startup(db_path):
    """Tasks with status='queued' in the DB at startup are loaded and run."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO background_tasks (id, task_type, job_id, status) VALUES (99, 't', 1, 'queued')"
        )
    results = []
    sched = LocalScheduler(
        db_path=db_path,
        run_task_fn=make_run_fn(results),
        task_types=frozenset({"t"}),
        vram_budgets={"t": 0.0},
    )
    sched.start()
    time.sleep(0.3)
    sched.shutdown()
    assert ("t", 99) in results


def test_task_spec_has_db_path_field():
    spec = TaskSpec(id=1, job_id=2, params=None, db_path=Path("/tmp/a.db"))
    assert spec.db_path == Path("/tmp/a.db")


def test_two_tasks_with_different_db_paths_each_run_against_their_own(tmp_path):
    """The core bug this fix exists for: two tasks enqueued with different
    db_path values must each be processed with THEIR OWN db_path, not the
    scheduler's construction-time db_path."""
    seen_db_paths = []
    done = __import__("threading").Event()
    lock = __import__("threading").Lock()

    def fake_run_task(db_path, task_id, task_type, job_id, params):
        with lock:
            seen_db_paths.append(db_path)
            if len(seen_db_paths) == 2:
                done.set()

    db_a = tmp_path / "tenant-a.db"
    db_b = tmp_path / "tenant-b.db"

    sched = LocalScheduler(
        db_path=None,
        run_task_fn=fake_run_task,
        task_types=frozenset({"cover_letter"}),
        vram_budgets={"cover_letter": 1.0},
    )
    sched.start()
    try:
        sched.enqueue(1, "cover_letter", 100, None, db_a)
        sched.enqueue(2, "cover_letter", 200, None, db_b)
        assert done.wait(timeout=5), "tasks did not complete in time"
    finally:
        sched.shutdown()

    assert db_a in seen_db_paths
    assert db_b in seen_db_paths
    assert len(seen_db_paths) == 2


def test_local_scheduler_db_path_none_skips_resume_scan(tmp_path):
    """db_path=None at construction must not raise and must not attempt to
    scan any database for queued tasks to resume."""
    sched = LocalScheduler(
        db_path=None,
        run_task_fn=lambda *a: None,
        task_types=frozenset({"cover_letter"}),
        vram_budgets={"cover_letter": 1.0},
    )
    assert sched._queues == {}


def test_get_scheduler_no_longer_requires_db_path():
    """get_scheduler() must not raise ValueError for a missing db_path on
    first call -- only run_task_fn/task_types/vram_budgets remain required."""
    reset_scheduler()
    try:
        sched = get_scheduler(
            run_task_fn=lambda *a: None,
            task_types=frozenset({"cover_letter"}),
            vram_budgets={"cover_letter": 1.0},
        )
        assert sched is not None
    finally:
        reset_scheduler()
