# circuitforge_core.mqtt

Async MQTT messaging and Meshtastic mesh radio integration. MIT licensed.

## What are you connecting to?

Choose your backend before installing:

| Backend | When to use |
|---|---|
| MQTT broker | You have a running MQTT broker (Mosquitto, HiveMQ, etc.) and want to send/receive structured messages over TCP |
| Meshtastic serial | You have a Meshtastic-compatible radio connected via USB and want to send messages over LoRa mesh |

These are independent backends, not sequential steps. Pick one.

---

## MQTT broker path

### Install

```bash
pip install "circuitforge-core[mqtt]"
```

### Quick start

```python
import asyncio
from circuitforge_core.mqtt.client import MQTTClient
from circuitforge_core.mqtt.models import MQTTConfig

cfg = MQTTConfig(host="localhost", port=1883)
client = MQTTClient(cfg)

@client.on("sensor/#")
async def handle_sensor(msg):
    print(msg.topic, msg.text())

asyncio.run(client.run())
```

`client.run()` subscribes to all registered patterns and reconnects automatically if the connection drops.

### Iterating raw messages

```python
from circuitforge_core.mqtt.client import MQTTClient
from circuitforge_core.mqtt.models import MQTTConfig

cfg = MQTTConfig(host="localhost")
client = MQTTClient(cfg)

async with client as messages:
    async for msg in messages:
        print(msg.topic, msg.payload)
```

### MQTTConfig

```python
from circuitforge_core.mqtt.models import MQTTConfig

cfg = MQTTConfig(
    host="localhost",     # required
    port=1883,            # default
    username=None,        # optional
    password=None,        # optional
    tls=False,            # set True for port 8883
    client_id=None,       # auto-generated if None
)
```

### Publishing

```python
await client.publish("sensor/room1/temp", payload=b"22.5")
```

---

## Meshtastic serial path

### Hardware required

A Meshtastic-compatible LoRa radio connected via USB serial. Supported boards include T-Beam, T-Echo, Heltec V3, RAK4631, and others listed at [meshtastic.org/docs/hardware](https://meshtastic.org/docs/hardware/).

### Install

```bash
pip install "circuitforge-core[meshtastic-serial]"
```

### Quick start

```python
import asyncio
from circuitforge_core.mqtt.meshtastic import MeshtasticSerialClient

async def main():
    async with MeshtasticSerialClient(port="/dev/ttyUSB0") as mesh:
        await mesh.send_text("hello mesh", channel=0)
        async for packet in mesh.packets():
            print(packet)

asyncio.run(main())
```

### Port detection

If you are unsure of the device path:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
# or on macOS:
ls /dev/cu.*
```

---

## TopicRouter

`TopicRouter` lets you register pattern-matched handlers for MQTT topics.

```python
from circuitforge_core.mqtt.router import TopicRouter

router = TopicRouter()

@router.on("sensor/+/temp")
async def handle_temp(msg):
    print(msg.topic, msg.text())

@router.on("alerts/#")
async def handle_alert(msg):
    print("alert:", msg.text())
```

Pass the router to `MQTTClient`:

```python
client = MQTTClient(cfg, router=router)
await client.run()
```

!!! warning "Known issue: `matches()` not yet implemented"
    The `matches()` function used internally by `TopicRouter` to route messages to handlers raises `NotImplementedError`. Dispatching to handlers via pattern matching will fail at runtime.

    **Workaround:** Use the raw message iteration path (`async with client as messages`) and match topics manually:

    ```python
    async with client as messages:
        async for msg in messages:
            if msg.topic.startswith("sensor/"):
                await handle_sensor(msg)
    ```

    Tracked at [circuitforge-core#TBD] — `matches()` is marked TODO in `router.py`.

---

## MQTTMessage

```python
from circuitforge_core.mqtt.models import MQTTMessage

msg.topic       # str — full topic string
msg.payload     # bytes — raw payload
msg.text()      # str — payload decoded as UTF-8
msg.json()      # Any — payload parsed as JSON
msg.received_at # datetime — UTC timestamp
```

---

## Install extras

| Extra | What it installs |
|---|---|
| `mqtt` | `aiomqtt` — MQTT broker connectivity |
| `meshtastic-serial` | `meshtastic`, `pypubsub` — USB serial radio |
| `meshtastic-service` | Both of the above + FastAPI + uvicorn |

```bash
# MQTT broker only
pip install "circuitforge-core[mqtt]"

# Meshtastic serial only
pip install "circuitforge-core[meshtastic-serial]"

# Both + FastAPI service layer
pip install "circuitforge-core[meshtastic-service]"
```
