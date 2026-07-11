# task_bridge

Shared MIT data contract for pushing tasks from a CF product into an external scheduler. Pilot consumer: Kiwi's pantry expiry alerts pushed into [Focus Flow](https://git.opensourcesolarpunk.com/Circuit-Forge/focus-flow), Ashley Venn's ADHD-first scheduler (AGPL-3.0, external project).

## Why this lives in cf-core

Focus Flow is AGPL-3.0; CF's AI features are BSL 1.1. To avoid any question of AGPL reach extending into CF's licensed code, the two sides never share a process or artifact:

- The **data contract** (this module) is MIT, a pure data definition with no product-specific logic.
- Each **product's exporter/receiver** lives in that product's own repo, under its own licensing.
- The two sides talk only over local HTTP, never a shared library or process.

Design spec: `circuitforge-plans/shared/superpowers/specs/2026-07-05-focus-flow-task-bridge-design.md`.

## Usage

```python
from circuitforge_core.task_bridge import ExternalTask, push_tasks

task = ExternalTask(
    source_product="kiwi",
    external_id="kiwi:item:1234",   # stable, idempotent per source item — upserts, not duplicates
    title="Use up milk",
    due_at="2026-07-07T00:00:00Z",  # ISO 8601 UTC
    notes="Opened 2026-07-01",
)

push_tasks("http://127.0.0.1:8512/import/tasks", token, [task])
```

`kind` is always `"flexible"` in schema v1 — Focus Flow's calm task kind. External sources can never set `"inflexible"`, `"critical"`, `"locked"`, or `"surprise"`, so no CF product can inject urgency into another product's UX.

`status` is `"active"` (default) or `"cancelled"` — push a `"cancelled"` record when the source item is deleted or edited away before the external scheduler marks it complete.

## API

- `ExternalTask(source_product, external_id, title, due_at, notes=None, kind="flexible", status="active")` — frozen dataclass; validates required fields and `kind`/`status` in `__post_init__`. `.to_dict()` serializes to the wire format.
- `push_tasks(endpoint, token, tasks, *, timeout=10.0)` — POSTs `{"tasks": [...]}` with `Authorization: Bearer <token>`. Raises `TaskBridgeError` on transport failure or non-2xx response.

## What this module does *not* do

No transport server, no auth/token generation or pairing logic, no product-specific behavior (diffing, callback handling, inventory updates). Those live entirely on the consuming product's side — see Kiwi's `focus_flow` service for the reference implementation.

Install: `pip install circuitforge-core[task-bridge]` (pulls in `httpx`).
