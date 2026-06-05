"""Abstract interface for Meshtastic backends.

MIT licensed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class MeshtasticInterface(ABC):
    """Async interface for receiving and sending Meshtastic packets.

    Two concrete backends exist:

    - MQTTMeshtasticBackend  — subscribes to a Meshtastic MQTT bridge
    - SerialMeshtasticBackend — connects directly via the meshtastic Python library
    """

    @abstractmethod
    def packets(self) -> AsyncIterator:
        """Async generator of MeshtasticPacket objects.

        Yields packets as they arrive. Runs until cancelled.
        Concrete types are ``MeshtasticPacket`` from
        ``circuitforge_core.mqtt.meshtastic.models``.
        """

    @abstractmethod
    async def send_text(
        self,
        text: str,
        dest_id: int = 0xFFFFFFFF,
        channel: int = 0,
    ) -> None:
        """Send a text message to dest_id (default: broadcast)."""
