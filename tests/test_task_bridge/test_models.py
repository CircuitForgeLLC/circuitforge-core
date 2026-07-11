"""Tests for circuitforge_core.task_bridge.models (cf-core #66). No network."""
from __future__ import annotations

import pytest

from circuitforge_core.task_bridge.models import SCHEMA_VERSION, ExternalTask


def _task(**overrides) -> ExternalTask:
    fields = {
        "source_product": "kiwi",
        "external_id": "kiwi:item:1234",
        "title": "Use up milk",
        "due_at": "2026-07-07T00:00:00Z",
    }
    fields.update(overrides)
    return ExternalTask(**fields)


class TestExternalTaskDefaults:
    def test_default_kind_is_flexible(self):
        assert _task().kind == "flexible"

    def test_default_status_is_active(self):
        assert _task().status == "active"

    def test_default_schema_version(self):
        assert _task().schema_version == SCHEMA_VERSION == 1

    def test_notes_defaults_to_none(self):
        assert _task().notes is None

    def test_frozen(self):
        task = _task()
        with pytest.raises((AttributeError, TypeError)):
            task.title = "other"  # type: ignore


class TestExternalTaskValidation:
    def test_rejects_empty_source_product(self):
        with pytest.raises(ValueError):
            _task(source_product="")

    def test_rejects_empty_external_id(self):
        with pytest.raises(ValueError):
            _task(external_id="")

    def test_rejects_empty_title(self):
        with pytest.raises(ValueError):
            _task(title="")

    def test_rejects_non_flexible_kind(self):
        with pytest.raises(ValueError):
            _task(kind="critical")

    def test_rejects_unknown_status(self):
        with pytest.raises(ValueError):
            _task(status="done")

    def test_accepts_cancelled_status(self):
        assert _task(status="cancelled").status == "cancelled"


class TestExternalTaskToDict:
    def test_to_dict_contains_all_fields(self):
        task = _task(notes="Opened 2026-07-01")
        d = task.to_dict()
        assert d == {
            "source_product": "kiwi",
            "external_id": "kiwi:item:1234",
            "title": "Use up milk",
            "due_at": "2026-07-07T00:00:00Z",
            "notes": "Opened 2026-07-01",
            "kind": "flexible",
            "status": "active",
            "schema_version": 1,
        }

    def test_to_dict_is_json_serializable(self):
        import json

        task = _task()
        json.dumps(task.to_dict())  # must not raise
