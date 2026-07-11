"""Tests for circuitforge_core.task_bridge.client (cf-core #66)."""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from circuitforge_core.task_bridge.client import TaskBridgeError, push_tasks
from circuitforge_core.task_bridge.models import ExternalTask


def _task(**overrides) -> ExternalTask:
    fields = {
        "source_product": "kiwi",
        "external_id": "kiwi:item:1234",
        "title": "Use up milk",
        "due_at": "2026-07-07T00:00:00Z",
    }
    fields.update(overrides)
    return ExternalTask(**fields)


@contextmanager
def _mocked_httpx_post(handler):
    """Patch httpx.post to route through an httpx.MockTransport handler."""
    transport = httpx.MockTransport(handler)

    def fake_post(url, *, json=None, headers=None, timeout=None):
        with httpx.Client(transport=transport) as client:
            return client.post(url, json=json, headers=headers, timeout=timeout)

    import circuitforge_core.task_bridge.client as client_module

    original = client_module.httpx.post
    client_module.httpx.post = fake_post
    try:
        yield
    finally:
        client_module.httpx.post = original


class TestPushTasksUnit:
    def test_sends_conformant_payload(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"accepted": 1})

        with _mocked_httpx_post(handler):
            resp = push_tasks(
                "http://127.0.0.1:9999/import/tasks", "tok123", [_task()]
            )

        assert resp.status_code == 200
        assert captured["auth"] == "Bearer tok123"
        assert captured["body"] == {"tasks": [_task().to_dict()]}

    def test_pushes_multiple_tasks_in_one_batch(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"accepted": 2})

        tasks = [_task(external_id="kiwi:item:1"), _task(external_id="kiwi:item:2")]
        with _mocked_httpx_post(handler):
            push_tasks("http://127.0.0.1:9999/import/tasks", "tok", tasks)

        assert len(captured["body"]["tasks"]) == 2

    def test_raises_on_non_2xx_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="bad token")

        with _mocked_httpx_post(handler):
            with pytest.raises(TaskBridgeError):
                push_tasks("http://127.0.0.1:9999/import/tasks", "bad", [_task()])

    def test_raises_on_transport_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with _mocked_httpx_post(handler):
            with pytest.raises(TaskBridgeError):
                push_tasks("http://127.0.0.1:9999/import/tasks", "tok", [_task()])

    def test_empty_task_list_sends_empty_batch(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"accepted": 0})

        with _mocked_httpx_post(handler):
            push_tasks("http://127.0.0.1:9999/import/tasks", "tok", [])

        assert captured["body"] == {"tasks": []}


class _FakeImporterHandler(BaseHTTPRequestHandler):
    """A minimal stand-in for Focus Flow's import API."""

    received: list[dict] = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        _FakeImporterHandler.received.append(
            {"body": body, "auth": self.headers.get("Authorization")}
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"accepted": len(body["tasks"])}).encode())

    def log_message(self, format, *args):
        pass  # silence test output


class TestPushTasksContract:
    """Real local HTTP server standing in for Focus Flow's importer."""

    def test_client_emits_conformant_payload_over_real_socket(self):
        _FakeImporterHandler.received = []
        server = HTTPServer(("127.0.0.1", 0), _FakeImporterHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            task = _task(notes="Opened 2026-07-01")
            resp = push_tasks(
                f"http://127.0.0.1:{port}/import/tasks", "pairing-token-abc", [task]
            )
            thread.join(timeout=5)

            assert resp.status_code == 200
            assert resp.json() == {"accepted": 1}
            assert len(_FakeImporterHandler.received) == 1
            received = _FakeImporterHandler.received[0]
            assert received["auth"] == "Bearer pairing-token-abc"
            assert received["body"] == {"tasks": [task.to_dict()]}
        finally:
            server.server_close()
