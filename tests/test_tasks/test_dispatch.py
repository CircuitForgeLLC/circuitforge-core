"""Tests for circuitforge_core.tasks.dispatch (cf-core #67)."""
from __future__ import annotations

import threading
import time

import pytest

from circuitforge_core.tasks.dispatch import (
    dispatch_task,
    get_task_status,
    register_task_runner,
    reset_dispatch_registry,
    unregister_task_runner,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_dispatch_registry()
    yield
    reset_dispatch_registry()


def _wait_for_status(task_id, target_statuses, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get_task_status(task_id)
        if status["status"] in target_statuses:
            return status
        time.sleep(0.01)
    raise AssertionError(f"task {task_id} did not reach {target_statuses} in time")


class TestDispatchTask:
    def test_raises_lookup_error_when_unregistered(self):
        with pytest.raises(LookupError):
            dispatch_task("pagepiper/ingest_pdf", {"doc_id": "1"})

    def test_returns_a_task_id_string(self):
        register_task_runner("test/noop", lambda **kwargs: None)
        task_id = dispatch_task("test/noop", {})
        assert isinstance(task_id, str) and task_id

    def test_two_dispatches_get_different_task_ids(self):
        register_task_runner("test/noop", lambda **kwargs: None)
        id1 = dispatch_task("test/noop", {})
        id2 = dispatch_task("test/noop", {})
        assert id1 != id2

    def test_task_runs_registered_fn_with_args_as_kwargs(self):
        received = {}
        event = threading.Event()

        def runner(*, doc_id, file_path):
            received["doc_id"] = doc_id
            received["file_path"] = file_path
            event.set()

        register_task_runner("pagepiper/ingest_pdf", runner)
        task_id = dispatch_task(
            "pagepiper/ingest_pdf", {"doc_id": "abc", "file_path": "/tmp/x.pdf"}
        )
        assert event.wait(timeout=2.0)
        assert received == {"doc_id": "abc", "file_path": "/tmp/x.pdf"}
        _wait_for_status(task_id, {"complete"})

    def test_task_status_reaches_complete_on_success(self):
        register_task_runner("test/ok", lambda **kwargs: None)
        task_id = dispatch_task("test/ok", {})
        status = _wait_for_status(task_id, {"complete", "error"})
        assert status == {"status": "complete", "progress": 100, "error": None}

    def test_task_status_reaches_error_on_exception(self):
        def failing(**kwargs):
            raise ValueError("boom")

        register_task_runner("test/fail", failing)
        task_id = dispatch_task("test/fail", {})
        status = _wait_for_status(task_id, {"complete", "error"})
        assert status["status"] == "error"
        assert status["error"] == "boom"

    def test_unregister_removes_runner(self):
        register_task_runner("test/temp", lambda **kwargs: None)
        unregister_task_runner("test/temp")
        with pytest.raises(LookupError):
            dispatch_task("test/temp", {})


class TestGetTaskStatus:
    def test_raises_key_error_for_unknown_task_id(self):
        with pytest.raises(KeyError):
            get_task_status("not-a-real-task-id")

    def test_status_immediately_after_dispatch_is_queued_or_running(self):
        gate = threading.Event()

        def slow(**kwargs):
            gate.wait(timeout=2.0)

        register_task_runner("test/slow", slow)
        try:
            task_id = dispatch_task("test/slow", {})
            status = get_task_status(task_id)
            assert status["status"] in ("queued", "running")
        finally:
            gate.set()

    def test_returns_a_copy_not_the_internal_dict(self):
        register_task_runner("test/ok", lambda **kwargs: None)
        task_id = dispatch_task("test/ok", {})
        _wait_for_status(task_id, {"complete"})
        status = get_task_status(task_id)
        status["status"] = "mutated"
        assert get_task_status(task_id)["status"] == "complete"
