"""Meshtastic adapter for circuitforge-core.

Two backends are available:

- ``MQTTMeshtasticBackend``  — subscribes to a Meshtastic MQTT bridge
- ``SerialMeshtasticBackend`` — direct serial/TCP connection via the
  ``meshtastic`` Python library

Use ``make_backend()`` for config-driven selection.

MIT licensed.
"""
from __future__ import annotations

from circuitforge_core.mqtt.meshtastic.interface import MeshtasticInterface
from circuitforge_core.mqtt.meshtastic.models import (
    MeshtasticPacket,
    MeshtasticPosition,
    MeshtasticTelemetry,
)
from circuitforge_core.mqtt.meshtastic.mqtt_backend import MQTTMeshtasticBackend
from circuitforge_core.mqtt.meshtastic.serial_backend import SerialMeshtasticBackend
from circuitforge_core.mqtt.models import MQTTConfig


def make_backend(config: dict) -> MeshtasticInterface:
    """Construct a Meshtastic backend from a config dict.

    Config keys:
        backend (str): ``"mqtt"`` or ``"serial"`` (required)

        For ``"mqtt"`` backend:
            broker_host (str): MQTT broker hostname
            broker_port (int): MQTT broker port (default 1883)
            broker_username (str|None): optional
            broker_password (str|None): optional
            topic_prefix (str): topic to subscribe to (default ``msh/#``)

        For ``"serial"`` backend:
            dev_path (str|None): serial device, e.g. ``/dev/ttyUSB0``
            tcp_host (str|None): TCP hostname for TCP mode
            tcp_port (int): TCP port (default 4403)
    """
    backend = config.get("backend", "mqtt").lower()

    if backend == "mqtt":
        mqtt_cfg = MQTTConfig(
            host=config["broker_host"],
            port=int(config.get("broker_port", 1883)),
            username=config.get("broker_username"),
            password=config.get("broker_password"),
        )
        return MQTTMeshtasticBackend(
            mqtt_config=mqtt_cfg,
            topic_prefix=config.get("topic_prefix", "msh/#"),
        )

    if backend == "serial":
        return SerialMeshtasticBackend(
            dev_path=config.get("dev_path"),
            tcp_host=config.get("tcp_host"),
            tcp_port=int(config.get("tcp_port", 4403)),
        )

    raise ValueError(f"Unknown Meshtastic backend: {backend!r}. Must be 'mqtt' or 'serial'.")


__all__ = [
    "MeshtasticInterface",
    "MeshtasticPacket",
    "MeshtasticPosition",
    "MeshtasticTelemetry",
    "MQTTMeshtasticBackend",
    "SerialMeshtasticBackend",
    "make_backend",
]
