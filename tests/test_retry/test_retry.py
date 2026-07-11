"""Tests for circuitforge_core.retry (cf-core #65)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from circuitforge_core.retry import (
    disable_retries,
    enable_retries_globally,
    on_exception,
    retry,
)


class FlakyError(Exception):
    pass


def _flaky_after(n_failures):
    """Return a function that raises FlakyError n_failures times, then succeeds."""
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        if state["calls"] <= n_failures:
            raise FlakyError(f"attempt {state['calls']}")
        return state["calls"]

    fn.state = state
    return fn


@pytest.fixture(autouse=True)
def _ensure_retries_enabled():
    """Some tests disable retries; always restore the global state after."""
    yield
    enable_retries_globally()


class TestOnException:
    def test_succeeds_without_retry_when_no_exception(self):
        calls = {"n": 0}

        @on_exception(FlakyError, max_tries=3)
        def fn():
            calls["n"] += 1
            return "ok"

        assert fn() == "ok"
        assert calls["n"] == 1

    def test_retries_until_success(self):
        fn = _flaky_after(2)
        wrapped = on_exception(FlakyError, max_tries=5, sleep=lambda s: None)(fn)

        assert wrapped() == 3
        assert fn.state["calls"] == 3

    def test_gives_up_after_max_tries(self):
        fn = _flaky_after(10)
        wrapped = on_exception(FlakyError, max_tries=3, sleep=lambda s: None)(fn)

        with pytest.raises(FlakyError):
            wrapped()
        assert fn.state["calls"] == 3

    def test_only_retries_specified_exception_type(self):
        @on_exception(FlakyError, max_tries=3, sleep=lambda s: None)
        def fn():
            raise ValueError("not a FlakyError")

        with pytest.raises(ValueError):
            fn()

    def test_disable_retries_context_manager_calls_once(self):
        fn = _flaky_after(10)
        wrapped = on_exception(FlakyError, max_tries=5, sleep=lambda s: None)(fn)

        with disable_retries():
            with pytest.raises(FlakyError):
                wrapped()

        assert fn.state["calls"] == 1

    def test_retries_resume_after_disable_retries_context_exits(self):
        fn = _flaky_after(2)
        wrapped = on_exception(FlakyError, max_tries=5, sleep=lambda s: None)(fn)

        with disable_retries():
            pass  # just exercise enter/exit without calling wrapped()

        assert wrapped() == 3


class TestRetry:
    def test_retry_calls_target_with_args_and_kwargs(self):
        def fn(a, b, *, c):
            return a + b + c

        assert retry(fn, 1, 2, c=3, max_tries=1) == 6

    def test_retry_retries_on_failure(self):
        fn = _flaky_after(2)

        with patch("time.sleep"):
            assert retry(fn, max_tries=5) == 3
        assert fn.state["calls"] == 3

    def test_retry_raises_after_exhausting_max_tries(self):
        fn = _flaky_after(10)

        with patch("time.sleep"), pytest.raises(FlakyError):
            retry(fn, exception=FlakyError, max_tries=2)
