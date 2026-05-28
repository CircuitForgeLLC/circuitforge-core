<p align="center">
  <img src="docs/cf-logo.png" alt="CircuitForge logo" width="120" />
</p>

<h1 align="center">circuitforge-core</h1>

<p align="center">Shared Python scaffold for privacy-first, self-hosted AI tools</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/version-0.20.0-blue.svg" alt="v0.20.0" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+" />
  <a href="https://git.opensourcesolarpunk.com/Circuit-Forge/circuitforge-core"><img src="https://img.shields.io/badge/repo-Forgejo-orange.svg" alt="Forgejo" /></a>
</p>

---

## Why circuitforge-core?

- **Local inference first.** The LLM router defaults to Ollama on localhost. Cloud APIs are a configurable fallback, not the default path. No telemetry, no round-trips you didn't ask for.
- **VRAM-aware scheduling.** The task scheduler and resource coordinator track GPU memory across concurrent services, allocate slots before loading models, and evict backends gracefully when VRAM is scarce.
- **Consistent tier system across products.** One `tiers` module handles Free / Paid / Premium / Ultra tiers, BYOK (bring your own key) unlocks, and local-vision capability gates — the same way in every product.
- **Uniform developer experience.** DB migrations, config validation, document ingestion, process management, and preference storage all share a single, tested implementation. Products extend, not reimplement.

---

## Install

```bash
# From PyPI
pip install circuitforge-core

# Editable install from source (recommended for product development)
pip install -e /path/to/circuitforge-core

# With optional extras
pip install circuitforge-core[pdf]                  # PDF/DOCX/OCR document ingestion
pip install circuitforge-core[vector]               # SQLite-vec vector store
pip install circuitforge-core[text-transformers]    # Local transformer inference (cf-text)
pip install circuitforge-core[stt-faster-whisper]   # Speech-to-text via Faster Whisper
pip install circuitforge-core[tts-chatterbox]       # Text-to-speech via Chatterbox
pip install circuitforge-core[reranker-qwen3]       # Reranking via Qwen3
pip install circuitforge-core[community]            # PostgreSQL-backed community store
pip install circuitforge-core[manage]               # cf-manage CLI (Typer)
pip install circuitforge-core[dev]                  # All dev dependencies
```

---

## Modules

| Module | Status | Description |
|---|---|---|
| `db` | Implemented | SQLite connection factory and migration runner |
| `llm` | Implemented | LLM router with priority fallback chain (Ollama, vLLM, Anthropic, OpenAI-compatible) |
| `tiers` | Implemented | Tier system with BYOK and local-vision unlocks (Free / Paid / Premium / Ultra) |
| `config` | Implemented | Env validation and `.env` loader with startup fail-fast |
| `hardware` | Implemented | GPU/CPU detection, VRAM profiling, backend profile generation |
| `documents` | Implemented | PDF, DOCX, and image OCR ingestion into `StructuredDocument` |
| `affiliates` | Implemented | Affiliate URL wrapping with per-user opt-out and env-var fallback |
| `preferences` | Implemented | User preference store — local YAML with pluggable backend; dot-path get/set |
| `tasks` | Implemented | VRAM-aware LLM task scheduler; shared slot manager across services |
| `manage` | Implemented | Cross-platform product process manager (Docker and native modes) |
| `resources` | Implemented | VRAM allocation, eviction engine, GPU profile registry |
| `text` | Implemented | Text processing utilities and local transformer inference service |
| `activitypub` | Implemented | ActivityPub actor, inbox, delivery, and Lemmy federation primitives |
| `audio` | Implemented | Audio buffer, format conversion, resampling, and VAD (voice activity detection) gate |
| `stt` | Implemented | Speech-to-text service (Faster Whisper backend) |
| `tts` | Implemented | Text-to-speech service (Chatterbox backend) |
| `musicgen` | Implemented | Music generation service (AudioCraft/MusicGen backend) |
| `reranker` | Implemented | Result reranking — BGE, Qwen3, cross-encoder, and Cohere adapters |
| `vector` | Implemented | SQLite-vec vector store with pluggable embedding backend |
| `api` | Implemented | Shared API helpers — corrections and feedback endpoints |
| `community` | Implemented | Community feed and social store (PostgreSQL-backed) |
| `platforms` | Implemented | Platform-specific integrations (eBay) |
| `cloud_session` | Implemented | Cloud session management primitives |
| `input` | Implemented | Input handling — MediaPipe gesture recognition |
| `job_quality` | Implemented | Job listing quality scoring and signal extraction |
| `vision` | Stub | Vision router (moondream2 / SigLIP dispatch — planned) |
| `wizard` | Stub | First-run wizard base class — products subclass `BaseWizard` |
| `pipeline` | Stub | Staging queue base — products provide concrete schema |

---

## Usage: LLM Router

The LLM router reads a config file at `~/.config/circuitforge/llm.yaml`, tries each backend in fallback order, and skips unreachable or disabled entries transparently.

```python
from circuitforge_core.llm import LLMRouter

# Auto-detects from env vars when llm.yaml is absent:
# ANTHROPIC_API_KEY, OPENAI_API_KEY / OPENAI_BASE_URL, OLLAMA_HOST
router = LLMRouter()

response = router.complete(
    messages=[{"role": "user", "content": "Summarize this in one sentence."}],
    system="You are a concise assistant.",
)
print(response)
```

**Example `llm.yaml`** (Ollama local, Anthropic cloud fallback):

```yaml
fallback_order:
  - ollama
  - anthropic

backends:
  ollama:
    type: openai_compat
    enabled: true
    base_url: http://localhost:11434/v1
    model: llama3.2:3b

  anthropic:
    type: anthropic
    enabled: true
    model: claude-haiku-4-5-20251001
    api_key_env: ANTHROPIC_API_KEY
    supports_images: true
```

---

## Usage: Database + Migrations

```python
from circuitforge_core.db import get_connection, run_migrations
from pathlib import Path

# Run product migrations on startup
run_migrations(db_path=Path("data/app.db"), migrations_dir=Path("db/migrations"))

# Get a connection anywhere in your app
with get_connection(Path("data/app.db")) as conn:
    conn.execute("INSERT INTO items (name) VALUES (?)", ("example",))
```

---

## Used by

| Product | Description |
|---|---|
| [peregrine](https://git.opensourcesolarpunk.com/Circuit-Forge/peregrine) | Job search — discovery, cover letters, interview prep |
| [snipe](https://git.opensourcesolarpunk.com/Circuit-Forge/snipe) | Auction sniping — eBay trust scoring, bid timing |
| [kiwi](https://git.opensourcesolarpunk.com/Circuit-Forge/kiwi) | Pantry tracker with barcode/receipt OCR and recipe suggestions |
| [avocet](https://git.opensourcesolarpunk.com/Circuit-Forge/avocet) | Email classifier training and benchmark harness |
| [osprey](https://git.opensourcesolarpunk.com/Circuit-Forge/osprey) | Government hold-line automation |
| [linnet](https://git.opensourcesolarpunk.com/Circuit-Forge/linnet) | Real-time tone annotation and voice transcription |
| pagepiper | PDF/rulebook RAG (retrieval-augmented generation) search |

---

## Contributing

circuitforge-core is MIT licensed. Contributions are welcome.

```bash
git clone https://git.opensourcesolarpunk.com/Circuit-Forge/circuitforge-core
cd circuitforge-core
pip install -e ".[dev]"
pytest
```

- New modules belong in `circuitforge_core/<module>/` as a package, not a flat file
- Keep modules focused — extract when a module exceeds 400 lines
- All public functions need type annotations
- Tests live in `tests/` — aim for 80% coverage on new code
- Use `ruff` for linting before submitting a PR

Open issues and PRs at: [git.opensourcesolarpunk.com/Circuit-Forge/circuitforge-core](https://git.opensourcesolarpunk.com/Circuit-Forge/circuitforge-core)

---

## License

MIT — see [LICENSE](LICENSE).

This is the fully open layer of the CircuitForge stack. Products built on top of circuitforge-core may carry different licenses (BSL 1.1 for AI features, proprietary for fine-tuned weights). The scaffold itself is and will remain MIT.

---

Humans own design, architecture, code review, testing, and verification. LLMs are part of our development workflow. [Our positions on LLM use →](https://circuitforge.tech/positions)
