# circuitforge_core/task_bridge/__init__.py
"""
task_bridge — shared MIT data contract for pushing tasks from a CF product
into an external scheduler (pilot consumer: Kiwi -> Focus Flow, AGPL-3.0).

Pure data contract + reusable push helper. No transport server, no auth/token
generation logic (that lives on the receiving side), no product-specific
behavior — keeps AGPL and BSL code from ever sharing a process or artifact.

Typical usage::

    from circuitforge_core.task_bridge import ExternalTask, push_tasks

    task = ExternalTask(
        source_product="kiwi",
        external_id="kiwi:item:1234",
        title="Use up milk",
        due_at="2026-07-07T00:00:00Z",
        notes="Opened 2026-07-01",
    )
    push_tasks("http://127.0.0.1:8512/import/tasks", token, [task])

Design spec: circuitforge-plans/shared/superpowers/specs/2026-07-05-focus-flow-task-bridge-design.md
"""
from .client import TaskBridgeError, push_tasks
from .models import SCHEMA_VERSION, ExternalTask

__all__ = [
    "ExternalTask",
    "SCHEMA_VERSION",
    "TaskBridgeError",
    "push_tasks",
]
