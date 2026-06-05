"""MQTT topic router with wildcard pattern matching.

MIT licensed.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from circuitforge_core.mqtt.models import MQTTMessage

logger = logging.getLogger(__name__)

Handler = Callable[[MQTTMessage], Coroutine[Any, Any, None]]


def matches(pattern: str, topic: str) -> bool:
    """Return True if topic matches the MQTT wildcard pattern.

    MQTT wildcard rules:
    - '+' matches exactly one topic level (segment between '/' separators)
    - '#' matches zero or more levels and MUST appear at the end of the pattern
    - All other characters match literally

    Examples:
        matches("sensor/+/temp", "sensor/room1/temp")       → True
        matches("sensor/+/temp", "sensor/a/b/temp")         → False
        matches("sensor/#", "sensor/room1/temp")            → True
        matches("sensor/#", "sensor")                       → True  (# = zero levels)
        matches("#", "any/topic/here")                      → True
        matches("a/b/c", "a/b/c")                          → True
    """
    # TODO: implement wildcard matching
    # Hint: split both pattern and topic on '/' and walk them in parallel.
    # Handle '#' early (if it appears, everything past that point in topic matches).
    # '+' must cover exactly one (non-empty) level.
    raise NotImplementedError("matches() is not yet implemented")


class TopicRouter:
    """Register async handlers for MQTT topic patterns and dispatch messages."""

    def __init__(self) -> None:
        self._routes: list[tuple[str, Handler]] = []

    @property
    def patterns(self) -> list[str]:
        return [p for p, _ in self._routes]

    def register(self, pattern: str, handler: Handler) -> None:
        """Add a handler for the given topic pattern."""
        self._routes.append((pattern, handler))

    def on(self, pattern: str) -> Callable[[Handler], Handler]:
        """Decorator: @router.on("sensor/#") async def handle(msg): ..."""
        def decorator(fn: Handler) -> Handler:
            self.register(pattern, fn)
            return fn
        return decorator

    async def dispatch(self, message: MQTTMessage) -> None:
        """Call all handlers whose pattern matches message.topic."""
        for pattern, handler in self._routes:
            try:
                if matches(pattern, message.topic):
                    if inspect.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
            except Exception:
                logger.exception("Handler for %r raised on topic %r", pattern, message.topic)
