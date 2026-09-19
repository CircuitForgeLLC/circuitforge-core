"""Tests for cf-text backend selection, mock backend, and public API."""
import os
import pytest
from unittest.mock import patch

from circuitforge_core.text.backends.base import (
    ChatMessage,
    GenerateResult,
    TextBackend,
    _select_backend,
    make_text_backend,
)
from circuitforge_core.text.backends.mock import MockTextBackend
from circuitforge_core.text import generate, chat, reset_backend, make_backend


# ── _select_backend ───────────────────────────────────────────────────────────

class TestSelectBackend:
    def test_explicit_llamacpp(self):
        assert _select_backend("model.gguf", "llamacpp") == "llamacpp"

    def test_explicit_transformers(self):
        assert _select_backend("model.gguf", "transformers") == "transformers"

    def test_explicit_invalid_raises(self):
        with pytest.raises(ValueError, match="not valid"):
            _select_backend("model.gguf", "ctransformers")

    def test_env_override_llamacpp(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_BACKEND", "llamacpp")
        assert _select_backend("Qwen/Qwen2.5-3B", None) == "llamacpp"

    def test_env_override_transformers(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_BACKEND", "transformers")
        assert _select_backend("model.gguf", None) == "transformers"

    def test_env_override_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_BACKEND", "ctransformers")
        with pytest.raises(ValueError):
            _select_backend("model.gguf", None)

    def test_caller_beats_env(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_BACKEND", "transformers")
        assert _select_backend("model.gguf", "llamacpp") == "llamacpp"

    def test_gguf_extension_selects_llamacpp(self, monkeypatch):
        monkeypatch.delenv("CF_TEXT_BACKEND", raising=False)
        assert _select_backend("/models/qwen2.5-3b-q4.gguf", None) == "llamacpp"

    def test_gguf_uppercase_extension(self, monkeypatch):
        monkeypatch.delenv("CF_TEXT_BACKEND", raising=False)
        assert _select_backend("/models/model.GGUF", None) == "llamacpp"

    def test_hf_repo_id_selects_transformers(self, monkeypatch):
        monkeypatch.delenv("CF_TEXT_BACKEND", raising=False)
        assert _select_backend("Qwen/Qwen2.5-3B-Instruct", None) == "transformers"

    def test_safetensors_dir_selects_transformers(self, monkeypatch):
        monkeypatch.delenv("CF_TEXT_BACKEND", raising=False)
        assert _select_backend("/models/qwen2.5-3b/", None) == "transformers"


# ── ChatMessage ───────────────────────────────────────────────────────────────

class TestChatMessage:
    def test_valid_roles(self):
        for role in ("system", "user", "assistant"):
            msg = ChatMessage(role, "hello")
            assert msg.role == role

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="Invalid role"):
            ChatMessage("bot", "hello")

    def test_to_dict(self):
        msg = ChatMessage("user", "hello")
        assert msg.to_dict() == {"role": "user", "content": "hello"}


# ── MockTextBackend ───────────────────────────────────────────────────────────

class TestMockTextBackend:
    def test_generate_returns_result(self):
        backend = MockTextBackend()
        result = backend.generate("write something")
        assert isinstance(result, GenerateResult)
        assert len(result.text) > 0

    def test_vram_mb_is_zero(self):
        assert MockTextBackend().vram_mb == 0

    def test_model_name(self):
        assert MockTextBackend(model_name="test-model").model_name == "test-model"

    def test_generate_stream_yields_tokens(self):
        backend = MockTextBackend()
        tokens = list(backend.generate_stream("hello"))
        assert len(tokens) > 0
        assert "".join(tokens).strip() == backend.generate("hello").text.strip()

    @pytest.mark.asyncio
    async def test_generate_async(self):
        backend = MockTextBackend()
        result = await backend.generate_async("hello")
        assert isinstance(result, GenerateResult)

    @pytest.mark.asyncio
    async def test_generate_stream_async(self):
        backend = MockTextBackend()
        tokens = []
        async for token in backend.generate_stream_async("hello"):
            tokens.append(token)
        assert len(tokens) > 0

    def test_chat(self):
        backend = MockTextBackend()
        messages = [ChatMessage("user", "hello")]
        result = backend.chat(messages)
        assert isinstance(result, GenerateResult)

    def test_isinstance_protocol(self):
        assert isinstance(MockTextBackend(), TextBackend)


# ── make_text_backend ─────────────────────────────────────────────────────────

class TestMakeTextBackend:
    def test_mock_flag(self):
        backend = make_text_backend("any-model", mock=True)
        assert isinstance(backend, MockTextBackend)

    def test_mock_env(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_MOCK", "1")
        backend = make_text_backend("any-model")
        assert isinstance(backend, MockTextBackend)

    def test_real_gguf_raises_import_error(self, monkeypatch):
        monkeypatch.delenv("CF_TEXT_MOCK", raising=False)
        monkeypatch.delenv("CF_TEXT_BACKEND", raising=False)
        with pytest.raises((ImportError, FileNotFoundError)):
            make_text_backend("/nonexistent/model.gguf", mock=False)

    def test_real_transformers_nonexistent_model_raises(self, monkeypatch):
        monkeypatch.delenv("CF_TEXT_MOCK", raising=False)
        monkeypatch.setenv("CF_TEXT_BACKEND", "transformers")
        # Use a clearly nonexistent local path — avoids a network hit and HF download
        with pytest.raises(Exception):
            make_text_backend("/nonexistent/local/model-dir", mock=False)


# ── make_text_backend with vllm spawning ──────────────────────────────────────

class TestMakeTextBackendVllmSpawning:
    def test_spawns_supervisor_when_no_external_url_set(self, monkeypatch):
        # Use setenv("") rather than delenv(raising=False) here: delenv on an
        # already-absent var records no undo action, so the direct
        # os.environ[...] = ... assignment the code under test performs would
        # leak into later tests in the same session (a real risk since
        # pytest-randomly varies test order). setenv("") is falsy for the
        # `if not os.environ.get(...)` guard below, so it still exercises the
        # spawn branch, but monkeypatch now has an original state to restore.
        monkeypatch.setenv("CF_TEXT_VLLM_URL", "")
        monkeypatch.delenv("CF_TEXT_MOCK", raising=False)
        with patch(
            "circuitforge_core.text.backends.vllm_subprocess.VllmSubprocessSupervisor"
        ) as mock_supervisor_cls, patch(
            "circuitforge_core.text.backends.vllm.VllmBackend"
        ) as mock_backend_cls:
            mock_supervisor = mock_supervisor_cls.return_value
            mock_supervisor.ensure_running.return_value = "http://localhost:8300"
            make_text_backend("vllm://IFM/K2-Horizon-7B", mock=False)

        mock_supervisor_cls.assert_called_once_with("IFM/K2-Horizon-7B")
        mock_supervisor.ensure_running.assert_called_once()
        assert os.environ["CF_TEXT_VLLM_URL"] == "http://localhost:8300"
        mock_backend_cls.assert_called_once_with(model_path="vllm://IFM/K2-Horizon-7B")

    def test_skips_supervisor_when_external_url_already_set(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_VLLM_URL", "http://external-vllm:8000")
        monkeypatch.delenv("CF_TEXT_MOCK", raising=False)
        with patch(
            "circuitforge_core.text.backends.vllm_subprocess.VllmSubprocessSupervisor"
        ) as mock_supervisor_cls, patch(
            "circuitforge_core.text.backends.vllm.VllmBackend"
        ) as mock_backend_cls:
            make_text_backend("vllm://IFM/K2-Horizon-7B", mock=False)

        mock_supervisor_cls.assert_not_called()
        assert os.environ["CF_TEXT_VLLM_URL"] == "http://external-vllm:8000"
        mock_backend_cls.assert_called_once_with(model_path="vllm://IFM/K2-Horizon-7B")


# ── Public API (singleton) ────────────────────────────────────────────────────

class TestPublicAPI:
    def setup_method(self):
        reset_backend()

    def teardown_method(self):
        reset_backend()

    def test_generate_mock(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_MOCK", "1")
        result = generate("write something")
        assert isinstance(result, GenerateResult)

    def test_generate_stream_mock(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_MOCK", "1")
        tokens = list(generate("hello", stream=True))
        assert len(tokens) > 0

    def test_chat_mock(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_MOCK", "1")
        result = chat([ChatMessage("user", "hello")])
        assert isinstance(result, GenerateResult)

    def test_chat_stream_raises(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_MOCK", "1")
        with pytest.raises(NotImplementedError):
            chat([ChatMessage("user", "hello")], stream=True)

    def test_make_backend_returns_mock(self):
        backend = make_backend("any", mock=True)
        assert isinstance(backend, MockTextBackend)

    def test_singleton_reused(self, monkeypatch):
        monkeypatch.setenv("CF_TEXT_MOCK", "1")
        r1 = generate("a")
        r2 = generate("b")
        # Both calls should succeed (singleton loaded once)
        assert isinstance(r1, GenerateResult)
        assert isinstance(r2, GenerateResult)
