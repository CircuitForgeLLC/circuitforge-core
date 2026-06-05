"""circuitforge_core.mqtt — async MQTT client with topic routing and
Meshtastic adapter support.

MIT licensed.

Quick start::

    from circuitforge_core.mqtt import MQTTClient, MQTTConfig

    cfg = MQTTConfig(host="localhost")
    client = MQTTClient(cfg)

    @client.on("sensors/#")
    async def handle(msg):
        print(msg.topic, msg.text())

    await client.run()

For Meshtastic::

    from circuitforge_core.mqtt.meshtastic import make_backend

    backend = make_backend({
        "backend": "mqtt",
        "broker_host": "mqtt.example.com",
        "topic_prefix": "msh/#",
    })
    async for pkt in backend.packets():
        print(pkt.summary())
"""

from circuitforge_core.mqtt.client import MQTTClient
from circuitforge_core.mqtt.models import MQTTConfig, MQTTMessage
from circuitforge_core.mqtt.router import TopicRouter, matches

__all__ = [
    "MQTTClient",
    "MQTTConfig",
    "MQTTMessage",
    "TopicRouter",
    "matches",
]
