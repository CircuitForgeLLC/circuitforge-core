"""circuitforge_core.memory — persistent knowledge graph via mnemo sidecar.

MIT licensed.

Requires the mnemo sidecar to be running (https://github.com/zaydmulani09/mnemo).
If the sidecar is not available, all operations silently no-op so products
can call memory methods unconditionally.

Quick start (in a FastAPI lifespan)::

    from circuitforge_core.memory import MemoryClient, MemoryConfig

    memory = MemoryClient(MemoryConfig.from_env())

    @asynccontextmanager
    async def lifespan(app):
        await memory.connect()
        yield
        await memory.close()

    # In a route:
    await memory.remember("User avoids shellfish", source="dietary-prefs")
    context = await memory.recall("What are this user's food restrictions?")

Docker Compose setup::

    services:
      mnemo:
        image: ghcr.io/zaydmulani09/mnemo:latest
        ports: ["8080:8080"]
        environment:
          MNEMO_LLM_PROVIDER: ollama
          MNEMO_LLM_BASE_URL: http://ollama:11434/v1
          MNEMO_LLM_MODEL: llama3
        volumes:
          - mnemo-data:/data

Environment variables (for MemoryConfig.from_env())::

    MNEMO_HOST     — default: localhost
    MNEMO_PORT     — default: 8080
    MNEMO_TIMEOUT  — default: 10.0
"""

from circuitforge_core.memory.client import MemoryClient, MemoryUnavailableError
from circuitforge_core.memory.models import MemoryConfig, MemoryEntity, MemoryStats

__all__ = [
    "MemoryClient",
    "MemoryConfig",
    "MemoryEntity",
    "MemoryStats",
    "MemoryUnavailableError",
]
