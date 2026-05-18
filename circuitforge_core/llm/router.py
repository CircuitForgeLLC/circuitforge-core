"""
LLM abstraction layer with priority fallback chain.

Reads config from ~/.config/circuitforge/llm.yaml (or the path passed to
LLMRouter.__init__). Tries backends in fallback_order; skips unreachable or
disabled entries and falls back to the next until one succeeds.

## Backend types

**openai_compat** — OpenAI-compatible /v1/chat/completions endpoint.
  Used for: Ollama, vLLM, GitHub Copilot wrapper, Claude Code wrapper,
  and the cf-orch trunk services (cf-text, cf-voice).

  With a cf_orch block the router first allocates via cf-orch, which
  starts the service on-demand and returns its URL. Without cf_orch the
  router does a static reachability check against base_url.

**anthropic** — Direct Anthropic API via the anthropic SDK.

**vision_service** — cf-vision managed service (moondream2 / SigLIP).
  Posts to /analyze; only used when images= is provided to complete().
  Supports cf_orch allocation to start cf-vision on-demand.

## Trunk services (The Orchard architecture)

These services live in cf-orch as branches; cf-core wires them as backends.
Products declare them in llm.yaml using the openai_compat type plus a
cf_orch block — the router handles allocation and URL injection transparently.

  cf-text   — Local transformer inference (/v1/chat/completions, port 8008).
               Default model set by default_model in the node's service
               profile; override via model_candidates in the cf_orch block.

  cf-voice  — STT/TTS pipeline endpoint (/v1/chat/completions, port 8009).
               Same allocation pattern as cf-text.

  cf-vision — Vision inference (moondream2 / SigLIP), vision_service type.
               Used via the vision_fallback_order when images are present.

## Config auto-detection (no llm.yaml)

When llm.yaml is absent, the router builds a minimal config from environment
variables: ANTHROPIC_API_KEY, OPENAI_API_KEY / OPENAI_BASE_URL, OLLAMA_HOST.
Ollama on localhost:11434 is always included as the lowest-cost local fallback.
"""

import logging
import os
import yaml
import requests
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".config" / "circuitforge" / "llm.yaml"


class LLMRouter:
    def __init__(self, config_path: Path | dict = CONFIG_PATH):
        self._ollama_tags_cache: dict[str, set[str]] = {}
        if isinstance(config_path, dict):
            self.config = config_path
        elif config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            env_config = self._auto_config_from_env()
            if env_config is None:
                raise FileNotFoundError(
                    f"{config_path} not found and no LLM env vars detected. "
                    "Either copy llm.yaml.example to ~/.config/circuitforge/llm.yaml, "
                    "or set ANTHROPIC_API_KEY, OPENAI_API_KEY, or OLLAMA_HOST."
                )
            logger.info(
                "[LLMRouter] No llm.yaml found — using env-var auto-config "
                "(backends: %s)",
                ", ".join(env_config["fallback_order"]),
            )
            self.config = env_config

    @staticmethod
    def _auto_config_from_env() -> dict | None:
        """Build a minimal LLM config from well-known environment variables.

        Priority order (highest to lowest):
          1. ANTHROPIC_API_KEY  → anthropic backend
          2. OPENAI_API_KEY     → openai-compat → api.openai.com (or OPENAI_BASE_URL)
          3. OLLAMA_HOST        → openai-compat → local Ollama (always included as last resort)

        Returns None only when none of these are set and Ollama is not configured,
        so the caller can decide whether to raise or surface a user-facing message.
        """
        backends: dict = {}
        fallback_order: list[str] = []

        if os.environ.get("ANTHROPIC_API_KEY"):
            backends["anthropic"] = {
                "type": "anthropic",
                "enabled": True,
                "model": os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                "api_key_env": "ANTHROPIC_API_KEY",
                "supports_images": True,
            }
            fallback_order.append("anthropic")

        if os.environ.get("OPENAI_API_KEY"):
            backends["openai"] = {
                "type": "openai_compat",
                "enabled": True,
                "base_url": os.environ.get(
                    "OPENAI_BASE_URL", "https://api.openai.com/v1"
                ),
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "supports_images": True,
            }
            fallback_order.append("openai")

        # Ollama — always added when any config exists, as the lowest-cost local fallback.
        # Unreachable Ollama is harmless — _is_reachable() skips it gracefully.
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not ollama_host.startswith("http"):
            ollama_host = f"http://{ollama_host}"
        backends["ollama"] = {
            "type": "openai_compat",
            "enabled": True,
            "base_url": ollama_host.rstrip("/") + "/v1",
            "model": os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
            "api_key": "any",
            "supports_images": False,
        }
        fallback_order.append("ollama")

        # Return None if only ollama is in the list AND no explicit host was set —
        # that means the user set nothing at all, not even OLLAMA_HOST.
        if fallback_order == ["ollama"] and "OLLAMA_HOST" not in os.environ:
            return None

        return {"backends": backends, "fallback_order": fallback_order}

    def _is_reachable(self, base_url: str) -> bool:
        """Quick health-check ping. Returns True if backend is up."""
        health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
        try:
            resp = requests.get(health_url, timeout=2)
            return resp.status_code < 500
        except Exception:
            return False

    def _check_ollama_model_pulled(self, base_url: str, model: str) -> None:
        """Raise RuntimeError with actionable message if model is not pulled in Ollama.

        Silently skips the check if the /api/tags endpoint is unavailable (e.g. vLLM).
        Results are cached per base_url for the lifetime of this router instance.
        """
        tags_url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
        if not hasattr(self, "_ollama_tags_cache"):
            self._ollama_tags_cache = {}
        if base_url not in self._ollama_tags_cache:
            try:
                resp = requests.get(tags_url, timeout=3)
                if resp.status_code != 200:
                    return
                pulled = {
                    m["name"].split(":")[0]
                    for m in resp.json().get("models", [])
                }
                self._ollama_tags_cache[base_url] = pulled
            except Exception:
                return  # can't verify — let the actual embed call fail naturally
        pulled_models = self._ollama_tags_cache.get(base_url)
        if pulled_models is None:
            return
        model_base = model.split(":")[0]
        if model_base not in pulled_models:
            raise RuntimeError(
                f'Ollama embedding model "{model}" is not pulled.\n'
                f"Fix: ollama pull {model}"
            )

    def _resolve_model(self, client: OpenAI, model: str) -> str:
        """Resolve __auto__ to the first model served by vLLM."""
        if model != "__auto__":
            return model
        models = client.models.list()
        return models.data[0].id

    def _try_cf_orch_alloc(self, backend: dict) -> "tuple | None":
        """
        If backend config has a cf_orch block and CF_ORCH_URL is set (env takes
        precedence over yaml url), allocate via cf-orch and return (ctx, alloc).

        Two allocation modes:
        - task-based (preferred): cf_orch block has `product` + `task` keys.
          Calls POST /api/inference/task; coordinator resolves model/node from
          assignments.yaml. No hardcoded model IDs in product config.
        - service-based (legacy): cf_orch block has `service` key.
          Calls allocate(service=...) directly.

        Returns None if not configured or allocation fails.
        Caller MUST call ctx.__exit__(None, None, None) in a finally block.
        """
        import os

        orch_cfg = backend.get("cf_orch")
        if not orch_cfg:
            return None
        orch_url = os.environ.get("CF_ORCH_URL", orch_cfg.get("url", ""))
        if not orch_url:
            return None
        try:
            from circuitforge_orch.client import CFOrchClient

            client = CFOrchClient(orch_url)
            ttl_s = float(orch_cfg.get("ttl_s", 3600.0))

            # Task-based allocation: product+task → coordinator resolves model/node.
            task = orch_cfg.get("task")
            product = orch_cfg.get("product") or os.environ.get("CF_APP_NAME") or None
            if task and product:
                ctx = client.task_allocate(product, task, ttl_s=ttl_s)
                alloc = ctx.__enter__()
                return (ctx, alloc)

            # Service-based allocation (legacy path).
            cf_app = os.environ.get("CF_APP_NAME") or None
            caller = f"{cf_app}.llm-router" if cf_app else "llm-router"
            ctx = client.allocate(
                orch_cfg.get("service", "vllm"),
                model_candidates=orch_cfg.get("model_candidates", []),
                ttl_s=ttl_s,
                caller=caller,
                pipeline=cf_app,
            )
            alloc = ctx.__enter__()
            return (ctx, alloc)
        except Exception as exc:
            logger.warning(
                "[LLMRouter] cf_orch allocation failed, using base_url directly: %s",
                exc,
            )
            return None

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model_override: str | None = None,
        fallback_order: list[str] | None = None,
        images: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a completion. Tries each backend in fallback_order.

        model_override: when set, replaces the configured model for
        openai_compat backends (e.g. pass a research-specific ollama model).
        fallback_order: when set, overrides config fallback_order for this
        call (e.g. pass config["research_fallback_order"] for research tasks).
        images: optional list of base64-encoded PNG/JPG strings. When provided,
        backends without supports_images=true are skipped. vision_service backends
        are only tried when images is provided.
        Raises RuntimeError if all backends are exhausted.
        """
        if os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes"):
            raise RuntimeError(
                "AI inference is disabled in the public demo. "
                "Run your own instance to use AI features."
            )
        order = (
            fallback_order
            if fallback_order is not None
            else self.config["fallback_order"]
        )
        for name in order:
            backend = self.config["backends"][name]

            if not backend.get("enabled", True):
                print(f"[LLMRouter] {name}: disabled, skipping")
                continue

            supports_images = backend.get("supports_images", False)
            is_vision_service = backend["type"] == "vision_service"

            # vision_service only used when images provided
            if is_vision_service and not images:
                print(f"[LLMRouter] {name}: vision_service skipped (no images)")
                continue

            # non-vision backends skipped when images provided and they don't support it
            if images and not supports_images and not is_vision_service:
                print(f"[LLMRouter] {name}: no image support, skipping")
                continue

            if is_vision_service:
                # cf_orch: try allocation first (same pattern as openai_compat).
                # Allocation can start the vision service on-demand on the cluster.
                orch_ctx = orch_alloc = None
                orch_result = self._try_cf_orch_alloc(backend)
                if orch_result is not None:
                    orch_ctx, orch_alloc = orch_result
                    backend = {**backend, "base_url": orch_alloc.url}
                elif not self._is_reachable(backend["base_url"]):
                    print(f"[LLMRouter] {name}: unreachable, skipping")
                    continue
                try:
                    resp = requests.post(
                        backend["base_url"].rstrip("/") + "/analyze",
                        json={
                            "prompt": prompt,
                            "image_base64": images[0] if images else "",
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    print(f"[LLMRouter] Used backend: {name} (vision_service)")
                    return resp.json()["text"]
                except Exception as e:
                    print(f"[LLMRouter] {name}: error — {e}, trying next")
                    continue
                finally:
                    if orch_ctx is not None:
                        orch_ctx.__exit__(None, None, None)

            elif backend["type"] == "openai_compat":
                # cf_orch: try allocation first — this may start the service on-demand.
                # Do NOT reachability-check before allocating; the service may be stopped
                # and the allocation is what starts it.
                orch_ctx = orch_alloc = None
                orch_result = self._try_cf_orch_alloc(backend)
                if orch_result is not None:
                    orch_ctx, orch_alloc = orch_result
                    backend = {**backend, "base_url": orch_alloc.url + "/v1"}
                elif not self._is_reachable(backend["base_url"]):
                    # Static backend (no cf-orch) — skip if not reachable.
                    print(f"[LLMRouter] {name}: unreachable, skipping")
                    continue
                try:
                    client = OpenAI(
                        base_url=backend["base_url"],
                        api_key=backend.get("api_key") or "any",
                    )
                    raw_model = model_override or backend["model"]
                    model = self._resolve_model(client, raw_model)
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    if images and supports_images:
                        content = [{"type": "text", "text": prompt}]
                        for img in images:
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img}"
                                    },
                                }
                            )
                        messages.append({"role": "user", "content": content})
                    else:
                        messages.append({"role": "user", "content": prompt})

                    create_kwargs: dict = {"model": model, "messages": messages}
                    if max_tokens is not None:
                        create_kwargs["max_tokens"] = max_tokens
                    resp = client.chat.completions.create(**create_kwargs)
                    print(f"[LLMRouter] Used backend: {name} ({model})")
                    return resp.choices[0].message.content

                except Exception as e:
                    print(f"[LLMRouter] {name}: error — {e}, trying next")
                    continue
                finally:
                    if orch_ctx is not None:
                        try:
                            orch_ctx.__exit__(None, None, None)
                        except Exception:
                            pass

            elif backend["type"] == "anthropic":
                api_key = os.environ.get(backend["api_key_env"], "")
                if not api_key:
                    print(
                        f"[LLMRouter] {name}: {backend['api_key_env']} not set, skipping"
                    )
                    continue
                try:
                    import anthropic as _anthropic

                    client = _anthropic.Anthropic(api_key=api_key)
                    if images and supports_images:
                        content = []
                        for img in images:
                            content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": img,
                                    },
                                }
                            )
                        content.append({"type": "text", "text": prompt})
                    else:
                        content = prompt
                    kwargs: dict = {
                        "model": backend["model"],
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": content}],
                    }
                    if system:
                        kwargs["system"] = system
                    msg = client.messages.create(**kwargs)
                    print(f"[LLMRouter] Used backend: {name}")
                    return msg.content[0].text
                except Exception as e:
                    print(f"[LLMRouter] {name}: error — {e}, trying next")
                    continue

        raise RuntimeError("All LLM backends exhausted")

    def embed(
        self,
        texts: list[str],
        model_override: str | None = None,
        fallback_order: list[str] | None = None,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Only openai_compat backends are tried — Ollama and vLLM expose
        /v1/embeddings; anthropic and vision_service do not.

        Uses ``embedding_model`` from backend config when present;
        falls back to ``model`` (the chat model) otherwise.

        Args:
            texts:          Texts to embed (batched in a single API call).
            model_override: Override the embedding model for this call.
            fallback_order: Override the backend fallback order for this call.

        Returns:
            List of float vectors, one per input text, in input order.

        Raises:
            RuntimeError: If all eligible backends are exhausted.
        """
        if os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes"):
            raise RuntimeError(
                "AI inference is disabled in the public demo. "
                "Run your own instance to use AI features."
            )
        order = (
            fallback_order
            if fallback_order is not None
            else self.config["fallback_order"]
        )
        for name in order:
            backend = self.config["backends"][name]
            if not backend.get("enabled", True):
                continue
            if backend["type"] != "openai_compat":
                continue

            orch_ctx = orch_alloc = None
            orch_result = self._try_cf_orch_alloc(backend)
            if orch_result is not None:
                orch_ctx, orch_alloc = orch_result
                backend = {**backend, "base_url": orch_alloc.url + "/v1"}
            elif not self._is_reachable(backend["base_url"]):
                print(f"[LLMRouter] {name}: unreachable, skipping")
                continue

            embed_model = model_override or backend.get(
                "embedding_model", backend["model"]
            )
            self._check_ollama_model_pulled(backend["base_url"], embed_model)

            try:
                client = OpenAI(
                    base_url=backend["base_url"],
                    api_key=backend.get("api_key") or "any",
                )
                model = embed_model
                resp = client.embeddings.create(model=model, input=texts)
                print(f"[LLMRouter] embed: used backend {name} ({model})")
                return [item.embedding for item in resp.data]
            except Exception as e:
                print(f"[LLMRouter] {name}: embed error — {e}, trying next")
                continue
            finally:
                if orch_ctx is not None:
                    try:
                        orch_ctx.__exit__(None, None, None)
                    except Exception:
                        pass

        raise RuntimeError("All LLM backends exhausted for embed()")


# Module-level singleton for convenience
_router: LLMRouter | None = None


def complete(prompt: str, system: str | None = None) -> str:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router.complete(prompt, system)
