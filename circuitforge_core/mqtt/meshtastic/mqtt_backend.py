"""Meshtastic MQTT bridge backend.

Subscribes to the JSON MQTT topics that Meshtastic firmware publishes when
the MQTT uplink is enabled on a node.

Topic schema (Meshtastic firmware >=2.1):
    msh/{region}/{gateway}/2/json/{portnum}/{fromId}

The payload is a JSON object. Examples by type:

Text message:
    {"channel":0,"from":123456789,"id":987,"payload":{"text":"hello"},
     "sender":"!07558d85","timestamp":1716200000,"to":4294967295,"type":"sendtext"}

Position:
    {"channel":0,"from":123456789,"payload":{"altitude":50,
     "latitude_i":374208130,"longitude_i":-1220848320,"time":1716200000},
     "type":"position"}

Node info:
    {"channel":0,"from":123456789,"payload":{"hardware":43,
     "id":"!07558d85","longname":"Alan Node","shortname":"AN"},
     "type":"nodeinfo"}

Telemetry:
    {"channel":0,"from":123456789,"payload":{"battery_level":82,
     "voltage":4.09,"channel_utilization":0.5,"air_util_tx":0.01,
     "time":1716200000},"type":"telemetry"}

MIT licensed.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from circuitforge_core.mqtt.client import MQTTClient
from circuitforge_core.mqtt.meshtastic.interface import MeshtasticInterface
from circuitforge_core.mqtt.meshtastic.models import (
    MeshtasticPacket,
    MeshtasticPosition,
    MeshtasticTelemetry,
)
from circuitforge_core.mqtt.models import MQTTConfig, MQTTMessage

logger = logging.getLogger(__name__)

# latitude_i / longitude_i are stored as integer × 1e7 in Meshtastic protobuf.
_COORD_SCALE = 1e-7


def _parse_packet(raw_json: str | bytes, topic: str) -> MeshtasticPacket | None:
    """Parse a Meshtastic MQTT JSON payload into a MeshtasticPacket.

    Returns None if the payload cannot be parsed or is an encrypted packet
    (payload is a base64 blob instead of a dict).
    """
    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.debug("Non-JSON Meshtastic payload on topic %r", topic)
        return None

    payload = obj.get("payload")
    if not isinstance(payload, dict):
        # Encrypted packet — payload is a base64 string; skip.
        return None

    from_num: int = obj.get("from", 0)
    sender: str = obj.get("sender", f"!{from_num:08x}")
    channel: int = obj.get("channel", 0)
    to_num: int = obj.get("to", 0xFFFFFFFF)
    raw_ts: int | None = payload.get("time") or obj.get("timestamp")
    received_at = (
        datetime.fromtimestamp(raw_ts, tz=timezone.utc) if raw_ts else datetime.now(tz=timezone.utc)
    )

    ptype: str = obj.get("type", "unknown").lower()

    if ptype in ("sendtext", "text"):
        return MeshtasticPacket(
            packet_type="text",
            from_id=sender,
            from_num=from_num,
            to_num=to_num,
            channel=channel,
            received_at=received_at,
            text=payload.get("text", ""),
            raw=obj,
        )

    if ptype == "position":
        lat_i: int | None = payload.get("latitude_i")
        lon_i: int | None = payload.get("longitude_i")
        return MeshtasticPacket(
            packet_type="position",
            from_id=sender,
            from_num=from_num,
            to_num=to_num,
            channel=channel,
            received_at=received_at,
            position=MeshtasticPosition(
                latitude=lat_i * _COORD_SCALE if lat_i is not None else None,
                longitude=lon_i * _COORD_SCALE if lon_i is not None else None,
                altitude_m=payload.get("altitude"),
                timestamp=received_at,
            ),
            raw=obj,
        )

    if ptype == "nodeinfo":
        return MeshtasticPacket(
            packet_type="nodeinfo",
            from_id=sender,
            from_num=from_num,
            to_num=to_num,
            channel=channel,
            received_at=received_at,
            node_longname=payload.get("longname"),
            node_shortname=payload.get("shortname"),
            hardware=payload.get("hardware"),
            raw=obj,
        )

    if ptype == "telemetry":
        return MeshtasticPacket(
            packet_type="telemetry",
            from_id=sender,
            from_num=from_num,
            to_num=to_num,
            channel=channel,
            received_at=received_at,
            telemetry=MeshtasticTelemetry(
                battery_level=payload.get("battery_level"),
                voltage=payload.get("voltage"),
                channel_util=payload.get("channel_utilization"),
                air_util_tx=payload.get("air_util_tx"),
            ),
            raw=obj,
        )

    # Routing, admin, and other packet types — return minimal packet.
    return MeshtasticPacket(
        packet_type="unknown",
        from_id=sender,
        from_num=from_num,
        to_num=to_num,
        channel=channel,
        received_at=received_at,
        raw=obj,
    )


class MQTTMeshtasticBackend(MeshtasticInterface):
    """Receive Meshtastic packets via a Meshtastic MQTT bridge.

    Requires a Meshtastic node with the MQTT uplink enabled, publishing to
    the configured broker. Set ``topic_prefix`` to match the region prefix
    configured on the node (default ``msh/#`` matches all regions).

    Args:
        mqtt_config: broker connection settings
        topic_prefix: MQTT topic pattern to subscribe to (default ``msh/#``)
    """

    def __init__(
        self,
        mqtt_config: MQTTConfig,
        topic_prefix: str = "msh/#",
    ) -> None:
        self._mqtt_config = mqtt_config
        self._topic_prefix = topic_prefix

    async def packets(self) -> AsyncIterator[MeshtasticPacket]:
        client = MQTTClient(self._mqtt_config)

        queue: asyncio.Queue[MeshtasticPacket] = asyncio.Queue()

        @client.on(self._topic_prefix)
        async def _handle(msg: MQTTMessage) -> None:
            pkt = _parse_packet(msg.payload, msg.topic)
            if pkt is not None:
                await queue.put(pkt)

        runner = asyncio.create_task(client.run())
        try:
            while True:
                yield await queue.get()
        finally:
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass

    async def send_text(
        self,
        text: str,
        dest_id: int = 0xFFFFFFFF,
        channel: int = 0,
    ) -> None:
        """Publishing back to MQTT is not supported by this backend.

        Meshtastic nodes consume from MQTT in a different topic namespace;
        use the serial backend or a direct Meshtastic MQTT channel config
        for two-way messaging.
        """
        raise NotImplementedError(
            "MQTTMeshtasticBackend is receive-only. "
            "Use SerialMeshtasticBackend for send support."
        )
