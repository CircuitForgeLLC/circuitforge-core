"""Meshtastic serial/TCP backend using the meshtastic Python library.

Connects directly to a Meshtastic node over serial port or TCP (e.g.
when a node exposes Meshtastic's native TCP API on port 4403).

The ``meshtastic`` library is synchronous and uses threading + PyPubSub
for callbacks. This backend bridges into asyncio via an asyncio.Queue:
the sync callback puts packets on the queue, and ``packets()`` awaits
items from it.

MIT licensed.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from circuitforge_core.mqtt.meshtastic.interface import MeshtasticInterface
from circuitforge_core.mqtt.meshtastic.models import (
    MeshtasticPacket,
    MeshtasticPosition,
    MeshtasticTelemetry,
)

logger = logging.getLogger(__name__)

_COORD_SCALE = 1e-7


def _packet_from_decoded(decoded: dict, from_id: int) -> MeshtasticPacket:
    """Convert a meshtastic-library decoded packet dict to MeshtasticPacket."""
    portnum: str = decoded.get("portnum", "UNKNOWN_APP")
    sender = f"!{from_id:08x}"
    to_num: int = decoded.get("to", 0xFFFFFFFF)
    channel: int = decoded.get("channel", 0)
    now = datetime.now(tz=timezone.utc)

    if portnum == "TEXT_MESSAGE_APP":
        return MeshtasticPacket(
            packet_type="text",
            from_id=sender,
            from_num=from_id,
            to_num=to_num,
            channel=channel,
            received_at=now,
            text=decoded.get("decoded", {}).get("text", ""),
            raw=decoded,
        )

    if portnum == "POSITION_APP":
        pos = decoded.get("decoded", {}).get("position", {})
        lat_i = pos.get("latitudeI")
        lon_i = pos.get("longitudeI")
        alt = pos.get("altitude")
        return MeshtasticPacket(
            packet_type="position",
            from_id=sender,
            from_num=from_id,
            to_num=to_num,
            channel=channel,
            received_at=now,
            position=MeshtasticPosition(
                latitude=lat_i * _COORD_SCALE if lat_i is not None else None,
                longitude=lon_i * _COORD_SCALE if lon_i is not None else None,
                altitude_m=alt,
                timestamp=now,
            ),
            raw=decoded,
        )

    if portnum == "NODEINFO_APP":
        info = decoded.get("decoded", {}).get("user", {})
        return MeshtasticPacket(
            packet_type="nodeinfo",
            from_id=sender,
            from_num=from_id,
            to_num=to_num,
            channel=channel,
            received_at=now,
            node_longname=info.get("longName"),
            node_shortname=info.get("shortName"),
            hardware=info.get("hwModel"),
            raw=decoded,
        )

    if portnum == "TELEMETRY_APP":
        telem = decoded.get("decoded", {}).get("telemetry", {})
        dev = telem.get("deviceMetrics", {})
        return MeshtasticPacket(
            packet_type="telemetry",
            from_id=sender,
            from_num=from_id,
            to_num=to_num,
            channel=channel,
            received_at=now,
            telemetry=MeshtasticTelemetry(
                battery_level=dev.get("batteryLevel"),
                voltage=dev.get("voltage"),
                channel_util=dev.get("channelUtilization"),
                air_util_tx=dev.get("airUtilTx"),
            ),
            raw=decoded,
        )

    return MeshtasticPacket(
        packet_type="unknown",
        from_id=sender,
        from_num=from_id,
        to_num=to_num,
        channel=channel,
        received_at=now,
        raw=decoded,
    )


class SerialMeshtasticBackend(MeshtasticInterface):
    """Receive and send Meshtastic packets via serial port or TCP.

    Args:
        dev_path: serial device path (e.g. ``/dev/ttyUSB0``) or ``None``
            to auto-detect the first connected Meshtastic device.
        tcp_host: hostname for TCP connection. If set, ``dev_path`` is ignored
            and a TCP connection to port 4403 is used.
        tcp_port: TCP port (default 4403).
    """

    def __init__(
        self,
        dev_path: str | None = None,
        tcp_host: str | None = None,
        tcp_port: int = 4403,
    ) -> None:
        self._dev_path = dev_path
        self._tcp_host = tcp_host
        self._tcp_port = tcp_port

    def _make_interface(self):
        try:
            import meshtastic.serial_interface
            import meshtastic.tcp_interface
        except ImportError as exc:
            raise ImportError(
                "meshtastic is required for SerialMeshtasticBackend. "
                "Install with: pip install circuitforge-core[meshtastic-serial]"
            ) from exc

        if self._tcp_host:
            return meshtastic.tcp_interface.TCPInterface(
                hostname=self._tcp_host,
                portNumber=self._tcp_port,
            )
        return meshtastic.serial_interface.SerialInterface(devPath=self._dev_path)

    async def packets(self) -> AsyncIterator[MeshtasticPacket]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[MeshtasticPacket | None] = asyncio.Queue()

        def _on_receive(packet: dict, interface) -> None:
            try:
                from_id: int = packet.get("from", 0)
                pkt = _packet_from_decoded(packet, from_id)
                loop.call_soon_threadsafe(queue.put_nowait, pkt)
            except Exception:
                logger.exception("Error decoding Meshtastic serial packet")

        def _on_connection_closed(interface) -> None:
            logger.warning("Meshtastic serial connection closed")
            loop.call_soon_threadsafe(queue.put_nowait, None)

        iface = await loop.run_in_executor(None, self._make_interface)

        try:
            from pubsub import pub
            pub.subscribe(_on_receive, "meshtastic.receive")
            pub.subscribe(_on_connection_closed, "meshtastic.connection.lost")
        except ImportError:
            await loop.run_in_executor(None, iface.close)
            raise ImportError(
                "pypubsub is required for SerialMeshtasticBackend. "
                "Install with: pip install circuitforge-core[meshtastic-serial]"
            )

        try:
            while True:
                pkt = await queue.get()
                if pkt is None:
                    break
                yield pkt
        finally:
            pub.unsubscribe(_on_receive, "meshtastic.receive")
            pub.unsubscribe(_on_connection_closed, "meshtastic.connection.lost")
            await loop.run_in_executor(None, iface.close)

    async def send_text(
        self,
        text: str,
        dest_id: int = 0xFFFFFFFF,
        channel: int = 0,
    ) -> None:
        loop = asyncio.get_running_loop()
        iface = await loop.run_in_executor(None, self._make_interface)
        try:
            await loop.run_in_executor(
                None,
                lambda: iface.sendText(text, destinationId=dest_id, channelIndex=channel),
            )
        finally:
            await loop.run_in_executor(None, iface.close)
