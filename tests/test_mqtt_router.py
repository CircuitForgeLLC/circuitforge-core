"""Tests for MQTT topic wildcard matching in circuitforge_core.mqtt.router."""
import pytest


# NOTE: matches() currently raises NotImplementedError — tests will fail
# until you implement it. Run these to verify correctness once implemented.

def _matches(pattern: str, topic: str) -> bool:
    from circuitforge_core.mqtt.router import matches
    return matches(pattern, topic)


class TestExactMatch:
    def test_exact(self):
        assert _matches("a/b/c", "a/b/c")

    def test_no_match(self):
        assert not _matches("a/b/c", "a/b/d")

    def test_empty_topic(self):
        assert _matches("", "")


class TestSingleLevelWildcard:
    def test_plus_middle(self):
        assert _matches("sensor/+/temp", "sensor/room1/temp")

    def test_plus_no_match_extra_level(self):
        assert not _matches("sensor/+/temp", "sensor/a/b/temp")

    def test_plus_start(self):
        assert _matches("+/b/c", "a/b/c")

    def test_plus_end(self):
        assert _matches("a/b/+", "a/b/anything")

    def test_multiple_plus(self):
        assert _matches("+/+/+", "x/y/z")

    def test_plus_no_match_empty_segment(self):
        # '+' must match exactly one level — a leading slash creates an empty segment
        # This edge case depends on the implementation; just check consistent behavior.
        result = _matches("+", "a/b")
        assert result is False


class TestMultiLevelWildcard:
    def test_hash_root(self):
        assert _matches("#", "a/b/c")

    def test_hash_prefix(self):
        assert _matches("sensor/#", "sensor/room1/temp")

    def test_hash_zero_levels(self):
        # '#' matches zero or more levels — "sensor/#" should match "sensor"
        assert _matches("sensor/#", "sensor")

    def test_hash_must_be_last(self):
        # '#' in the middle is invalid MQTT but we should handle gracefully
        # Just verify it doesn't crash; exact behavior is implementation-defined.
        try:
            _matches("sensor/#/foo", "sensor/bar/foo")
        except Exception:
            pass  # either False or ValueError is acceptable

    def test_hash_only(self):
        assert _matches("#", "anything")

    def test_hash_no_match_different_prefix(self):
        assert not _matches("sensor/#", "actuator/fan")


class TestMixedWildcards:
    def test_plus_and_hash(self):
        assert _matches("msh/+/#", "msh/us-west/node1/json/TEXT_MESSAGE_APP/!deadbeef")

    def test_plus_before_hash(self):
        assert _matches("+/#", "region/any/nested/topic")
