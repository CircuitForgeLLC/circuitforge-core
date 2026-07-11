"""Push client for the cf-core task_bridge module.

MIT licensed. Thin httpx wrapper any CF product reuses to push a batch of
ExternalTask records to a configured local HTTP endpoint. No transport
server, no auth/token generation logic — that lives on the receiving side
(e.g. Focus Flow's importer). See:
circuitforge-plans/shared/superpowers/specs/2026-07-05-focus-flow-task-bridge-design.md
"""
from __future__ import annotations

from collections.abc import Sequence

import httpx

from circuitforge_core.task_bridge.models import ExternalTask


class TaskBridgeError(RuntimeError):
    """Raised when pushing tasks to the configured endpoint fails."""


def push_tasks(
    endpoint: str,
    token: str,
    tasks: Sequence[ExternalTask],
    *,
    timeout: float = 10.0,
) -> httpx.Response:
    """
    POST a batch of ExternalTask records to `endpoint` as `{"tasks": [...]}`,
    authenticated with a bearer token.

    Raises:
        TaskBridgeError: the request failed to send, or the endpoint returned
            a non-2xx status.
    """
    payload = {"tasks": [t.to_dict() for t in tasks]}
    try:
        resp = httpx.post(
            endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise TaskBridgeError(f"task_bridge push to {endpoint!r} failed: {exc}") from exc

    if resp.status_code >= 300:
        raise TaskBridgeError(
            f"task_bridge push to {endpoint!r} returned {resp.status_code}: {resp.text}"
        )

    return resp
