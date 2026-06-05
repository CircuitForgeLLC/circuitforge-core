# circuitforge_core.memory

Persistent knowledge graph for CF products, backed by the
[mnemo](https://github.com/zaydmulani09/mnemo) sidecar.

## What it does

mnemo runs as a sidecar process alongside a product's FastAPI backend. It:

- Extracts named entities and relationships from text you feed it
- Persists them in a local SQLite database with WAL mode
- Returns a formatted context block for prompt injection in under 5ms

`cf_core.memory` wraps mnemo's Python SDK with CF-standard config,
graceful degradation (no-ops when the sidecar is absent), and
exponential backoff with automatic reconnect after transient failures.

## Install

```bash
pip install circuitforge-core[memory]
```

## Docker Compose setup

Add the `mnemo` service to your product's `compose.yml` alongside `ollama`.
Peregrine is the reference implementation — copy the block from
`peregrine/compose.yml`:

```yaml
services:

  mnemo:
    image: ghcr.io/zaydmulani09/mnemo:latest
    ports:
      - "${MNEMO_PORT:-8080}:8080"
    volumes:
      - mnemo-data:/data
    environment:
      - MNEMO_DB_PATH=/data/mnemo.db
      - MNEMO_LLM_PROVIDER=${MNEMO_LLM_PROVIDER:-ollama}
      - MNEMO_LLM_BASE_URL=${MNEMO_LLM_BASE_URL:-http://ollama:11434/v1}
      - MNEMO_LLM_API_KEY=${MNEMO_LLM_API_KEY:-ollama}
      - MNEMO_LLM_MODEL=${MNEMO_LLM_MODEL:-llama3.2:3b}
    depends_on:
      - ollama
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:8080/health"]
      interval: 15s
      timeout: 5s
      retries: 3
    profiles: [memory]
    restart: unless-stopped

volumes:
  mnemo-data:
```

Add these to the product's api service environment:

```yaml
    environment:
      - MNEMO_HOST=${MNEMO_HOST:-mnemo}
      - MNEMO_PORT=${MNEMO_PORT:-8080}
```

Launch with:

```bash
docker compose --profile memory --profile cpu up -d
# or alongside a GPU profile:
docker compose --profile memory --profile single-gpu up -d
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MNEMO_HOST` | `localhost` | Sidecar hostname (use `mnemo` in Docker) |
| `MNEMO_PORT` | `8080` | Sidecar port |
| `MNEMO_TIMEOUT` | `10.0` | HTTP timeout in seconds |

The sidecar itself is configured via `MNEMO_LLM_*` env vars (see compose block above).

## FastAPI integration

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from circuitforge_core.memory import MemoryClient, MemoryConfig

memory = MemoryClient(MemoryConfig.from_env())

@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory.connect()   # no-op + warning if sidecar absent
    yield
    await memory.close()

app = FastAPI(lifespan=lifespan)
```

## API

```python
# Store a text fragment (conversation turn, fact, user preference, etc.)
await memory.remember("User avoids shellfish and prefers dark mode", source="settings")

# Retrieve a prompt-ready context block
context = await memory.recall("What are this user's dietary restrictions?")
system_prompt = f"You are a helpful assistant.\n\n{context}"

# List extracted entities
entities = await memory.entities(limit=20)

# Stats snapshot
stats = await memory.stats()   # MemoryStats | None

# Wipe everything (irreversible)
await memory.wipe()
```

All methods return empty values (`False`, `""`, `[]`, `None`) when the
sidecar is not available — no try/except needed in product code.

## Resilience model

| Event | Behaviour |
|---|---|
| Sidecar absent at startup | `connect()` logs once, enters no-op mode |
| First call failure | Warning logged, 5s backoff scheduled |
| Nth consecutive failure | Backoff doubles each time (5→10→20→40→60s cap) |
| After `_MAX_FAILURES` (3) | Client marked unavailable; all calls no-op |
| Cooldown elapses | Next call silently attempts reconnect |
| Successful call | Failure counter and retry timer reset |
| `strict=True` | `MemoryUnavailableError` raised instead of no-op |

## Chunking note

mnemo stores each `remember()` call as a single chunk — it does **not**
automatically split large texts. For best retrieval quality, chunk on the
caller side before ingesting:

```python
# Good: one turn per ingest call
for turn in conversation_turns:
    await memory.remember(turn, source="chat", session_id=session_id)

# Avoid: one giant blob
await memory.remember(entire_conversation_as_one_string)
```
