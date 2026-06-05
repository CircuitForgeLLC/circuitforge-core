"""Async MQTT client wrapper around aiomqtt.

MIT licensed.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from circuitforge_core.mqtt.models import MQTTConfig, MQTTMessage
from circuitforge_core.mqtt.router import TopicRouter

logger = logging.getLogger(__name__)


class MQTTClient:
    """Async MQTT client that subscribes to topics and dispatches messages.

    Usage (with a router)::

        cfg = MQTTConfig(host="localhost")
        client = MQTTClient(cfg)

        @client.on("msh/#")
        async def handle_mesh(msg: MQTTMessage):
            print(msg.topic, msg.text())

        await client.run()

    Usage (iterate raw messages)::

        async with MQTTClient(cfg) as messages:
            async for msg in messages:
                print(msg.topic)
    """

    def __init__(self, config: MQTTConfig, router: TopicRouter | None = None) -> None:
        self._config = config
        self._router = router or TopicRouter()

    def on(self, pattern: str):
        """Shorthand decorator — forwards to the internal router."""
        return self._router.on(pattern)

    async def run(self) -> None:
        """Subscribe to all registered patterns and dispatch until cancelled.

        Reconnects automatically if the connection drops.
        """
        try:
            import aiomqtt
        except ImportError as exc:
            raise ImportError(
                "aiomqtt is required for MQTTClient. "
                "Install with: pip install circuitforge-core[mqtt]"
            ) from exc

        cfg = self._config
        while True:
            try:
                kwargs: dict[str, Any] = {
                    "hostname": cfg.host,
                    "port": cfg.port,
                    "keepalive": cfg.keepalive,
                    "tls_params": aiomqtt.TLSParameters() if cfg.tls else None,
                }
                if cfg.client_id:
                    kwargs["identifier"] = cfg.client_id
                if cfg.username is not None:
                    kwargs["username"] = cfg.username
                if cfg.password is not None:
                    kwargs["password"] = cfg.password

                async with aiomqtt.Client(**kwargs) as ac:
                    patterns = self._router.patterns
                    if not patterns:
                        logger.warning("MQTTClient started with no subscriptions")
                    for p in patterns:
                        await ac.subscribe(p)
                        logger.debug("Subscribed to %r on %s:%d", p, cfg.host, cfg.port)
                    logger.info("MQTT connected to %s:%d", cfg.host, cfg.port)

                    async for raw in ac.messages:
                        msg = MQTTMessage(
                            topic=str(raw.topic),
                            payload=raw.payload if isinstance(raw.payload, bytes) else str(raw.payload).encode(),
                            qos=raw.qos,
                            retain=raw.retain,
                            received_at=datetime.now(tz=timezone.utc),
                        )
                        await self._router.dispatch(msg)

            except asyncio.CancelledError:
                logger.info("MQTTClient cancelled")
                raise
            except Exception as exc:
                logger.warning(
                    "MQTT connection to %s:%d failed (%s), retrying in %.0fs",
                    cfg.host, cfg.port, exc, cfg.reconnect_interval,
                )
                await asyncio.sleep(cfg.reconnect_interval)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncIterator[MQTTMessage]]:
        """Context manager that yields an async iterator of raw messages.

        Useful when the caller wants to do its own routing::

            async with client.connect() as messages:
                async for msg in messages:
                    ...
        """
        try:
            import aiomqtt
        except ImportError as exc:
            raise ImportError(
                "aiomqtt is required. Install with: pip install circuitforge-core[mqtt]"
            ) from exc

        cfg = self._config
        kwargs: dict[str, Any] = {
            "hostname": cfg.host,
            "port": cfg.port,
            "keepalive": cfg.keepalive,
            "tls_params": aiomqtt.TLSParameters() if cfg.tls else None,
        }
        if cfg.client_id:
            kwargs["identifier"] = cfg.client_id
        if cfg.username is not None:
            kwargs["username"] = cfg.username
        if cfg.password is not None:
            kwargs["password"] = cfg.password

        async with aiomqtt.Client(**kwargs) as ac:
            for p in self._router.patterns:
                await ac.subscribe(p)

            async def _iter() -> AsyncIterator[MQTTMessage]:
                async for raw in ac.messages:
                    yield MQTTMessage(
                        topic=str(raw.topic),
                        payload=raw.payload if isinstance(raw.payload, bytes) else str(raw.payload).encode(),
                        qos=raw.qos,
                        retain=raw.retain,
                        received_at=datetime.now(tz=timezone.utc),
                    )

            yield _iter()
