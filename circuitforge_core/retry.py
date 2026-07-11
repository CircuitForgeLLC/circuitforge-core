# circuitforge_core/retry.py
"""
circuitforge_core.retry — thin standard-retry wrapper over backon (cf-core #65).

Standardizes retry/backoff behavior across cf-core modules and products that
make external calls (LLM endpoints, affiliate links, reranker APIs, federated
ActivityPub delivery). Wraps `backon` (MIT, zero stdlib dependencies) rather
than reimplementing retry logic, and rather than each module hand-rolling its
own ad-hoc retry loop.

Typical usage::

    from circuitforge_core.retry import on_exception
    import requests

    @on_exception(requests.RequestException, max_tries=3)
    def fetch(url):
        return requests.get(url, timeout=5)

Tests should disable retries globally rather than waiting out real backoff
delays::

    from circuitforge_core.retry import disable_retries

    def test_fetch_raises_immediately_on_failure():
        with disable_retries():
            with pytest.raises(requests.RequestException):
                fetch("http://unreachable")
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import backon
from backon import disable as disable_retries_globally
from backon import disable_retries, enable_retries
from backon import enable as enable_retries_globally

T = TypeVar("T")

__all__ = [
    "on_exception",
    "retry",
    "disable_retries",
    "enable_retries",
    "disable_retries_globally",
    "enable_retries_globally",
]

_DEFAULT_MAX_TRIES = 3
_DEFAULT_MAX_TIME = 30.0


def on_exception(
    exception: type[Exception] | tuple[type[Exception], ...] = Exception,
    *,
    max_tries: int = _DEFAULT_MAX_TRIES,
    max_time: float | None = _DEFAULT_MAX_TIME,
    **backon_kwargs,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator: retry a function on `exception` using full-jitter exponential
    backoff — CF's standard defaults for external API calls.

    Extra keyword arguments are forwarded to `backon.on_exception` (e.g.
    `on_backoff=` for a logging callback, `sleep=` to override the sleep
    function in tests).
    """
    return backon.on_exception(
        backon.expo(),
        exception,
        max_tries=max_tries,
        max_time=max_time,
        logger="circuitforge_core.retry",
        **backon_kwargs,
    )


def retry(
    target: Callable[..., T],
    *args,
    exception: type[Exception] | tuple[type[Exception], ...] = Exception,
    max_tries: int = _DEFAULT_MAX_TRIES,
    max_time: float | None = _DEFAULT_MAX_TIME,
    **kwargs,
) -> T:
    """
    Call `target(*args, **kwargs)` with CF's standard retry/backoff behavior,
    for wrapping an already-defined callable without decorator syntax.
    """
    wrapped = on_exception(exception, max_tries=max_tries, max_time=max_time)(target)
    return wrapped(*args, **kwargs)
