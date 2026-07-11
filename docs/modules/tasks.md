# tasks

VRAM-aware background task scheduler. Manages a queue of LLM inference jobs and coordinates VRAM allocation with the cf-orch coordinator before executing each task.

```python
from circuitforge_core.tasks import TaskScheduler, get_scheduler, reset_scheduler
```

## Why VRAM-aware scheduling

Running multiple LLM inference jobs concurrently on a single GPU causes OOM errors and corrupted outputs. The scheduler serializes LLM work per service and negotiates with the cf-orch coordinator so tasks across multiple products don't compete for the same VRAM budget.

## Core API

### `get_scheduler() -> TaskScheduler`

Returns the singleton scheduler for the current process. Creates it on first call.

### `reset_scheduler()`

Tears down the scheduler (releases VRAM leases, cancels pending tasks). Called during FastAPI lifespan teardown.

```python
# In FastAPI lifespan
from circuitforge_core.tasks import get_scheduler, reset_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = get_scheduler()
    yield
    reset_scheduler()
```

### `scheduler.submit(task_type, payload, vram_gb) -> str`

Enqueues a task. Returns the task ID. The scheduler acquires a VRAM lease from the coordinator before executing.

```python
task_id = await scheduler.submit(
    task_type="recipe_llm",
    payload={"pantry_ids": [1, 2, 3]},
    vram_gb=4.0,
)
```

### `scheduler.result(task_id) -> TaskResult | None`

Polls for a completed result. Returns `None` if still running.

## VRAM budgets

Each product defines its VRAM budgets in `compose.yml` / `compose.override.yml`:

```yaml
environment:
  VRAM_BUDGET_RECIPE_LLM: "4.0"
  VRAM_BUDGET_EXPIRY_LLM: "2.0"
```

These map to task types in the scheduler. If the coordinator is unavailable (local dev without cf-orch), the scheduler falls back to sequential local execution.

## Shim pattern

Products that need to re-export scheduler functions for backward compatibility use a shim:

```python
# myproduct/app/tasks/scheduler.py
from circuitforge_core.tasks.scheduler import (
    get_scheduler as _base_get_scheduler,
    reset_scheduler,          # re-export for lifespan teardown
)

def get_scheduler():
    """Product-specific scheduler with service name injected."""
    return _base_get_scheduler(service_name="myproduct")
```

Always re-export `reset_scheduler` from the shim so the FastAPI lifespan can import it from one place.

## Generic caller/args dispatch

Separate from the VRAM-budgeted scheduler above: `dispatch_task()`/`get_task_status()` are a generic, product-agnostic pair for running a named callable in the background and polling its status, keyed by a `"product/task_name"` string rather than the scheduler's `task_id`/`job_id`/SQLite-table shape.

```python
from circuitforge_core.tasks import register_task_runner, dispatch_task, get_task_status

# Once, at product startup:
register_task_runner("pagepiper/ingest_pdf", run_ingest_pdf)

# Anywhere a task needs dispatching:
task_id = dispatch_task("pagepiper/ingest_pdf", {"doc_id": "...", "file_path": "..."})
get_task_status(task_id)  # {"status": "running", "progress": 0, "error": None}
```

Free-tier, in-process, single-node — runs on a background thread pool, no cross-node distribution. `dispatch_task()` raises `LookupError` for an unregistered `caller`, so products with an `except Exception: ...` fallback to local execution keep working unchanged if they forget to register a runner.

`register_task_runner`/`dispatch_task`/`get_task_status` route through `circuitforge_core.tasks.dispatch` internally — nothing shared with `scheduler.py`'s VRAM-aware queue.
