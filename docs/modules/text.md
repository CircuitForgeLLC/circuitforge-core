# text

Text processing utilities. Normalization, truncation, chunking, and token estimation — shared across all products that manipulate text before or after LLM inference.

```python
from circuitforge_core.text import normalize, chunk, truncate, estimate_tokens
```

## `normalize(text: str) -> str`

Strips excess whitespace, normalizes unicode (NFC), and removes null bytes and control characters that can cause downstream issues with SQLite FTS5 or LLM tokenizers.

```python
from circuitforge_core.text import normalize

clean = normalize("  Hello\u00a0world\x00  ")
# → "Hello world"
```

## `truncate(text: str, max_tokens: int, model: str = "default") -> str`

Truncates text to approximately `max_tokens` tokens, breaking at sentence or paragraph boundaries where possible. Uses a simple byte-based heuristic (1 token ≈ 4 bytes) unless a specific model tokenizer is requested.

```python
excerpt = truncate(long_doc, max_tokens=2048)
```

## `chunk(text: str, chunk_size: int, overlap: int = 0) -> list[str]`

Splits text into overlapping chunks for RAG (retrieval-augmented generation) pipelines. Respects paragraph boundaries.

```python
chunks = chunk(article_text, chunk_size=512, overlap=64)
```

## `estimate_tokens(text: str, model: str = "default") -> int`

Estimates token count without loading a full tokenizer. Accurate enough for context window budget planning (within ~10%).

## FTS5 helpers

SQLite FTS5 has quirks with special characters in MATCH expressions. The `text` module provides helpers used by the recipe engine and other FTS5 consumers:

```python
from circuitforge_core.text import fts_quote, strip_apostrophes

# Always double-quote FTS5 terms — bare tokens break on brand names
query = " ".join(fts_quote(term) for term in tokens)
# → '"chicken" "breast" "lemon"'

# Strip apostrophes before FTS5 queries
clean = strip_apostrophes("O'Doul's")
# → "ODoulS"
```

!!! warning "FTS5 gotcha"
    Always quote ALL terms in MATCH expressions. Bare tokens break on brand names (e.g., `O'Doul's`), plant-based ingredient names, and anything with punctuation.

---

## LLM inference service

`circuitforge_core.text.app` is a self-contained FastAPI inference server. It exposes a local LLM (or PII classifier) over HTTP so that products can call it via `CF_TEXT_URL` without bundling heavy ML dependencies themselves.

### What are you running?

Three independent paths — pick one before installing:

| Path | Use case | Extra |
|---|---|---|
| **LLM inference** | Chat, completion, summarisation using a GGUF or HuggingFace model | `text-llamacpp` or `text-transformers` |
| **VLM inference** | Vision-language model that accepts images alongside text | `text-llamacpp` (GGUF with `--mmproj`) or `text-transformers` |
| **Classifier / PII filter** | NER-based PII detection and redaction | `text-transformers` |

---

### LLM inference (GGUF via llama.cpp)

```bash
pip install "circuitforge-core[text-llamacpp]"
```

```bash
python -m circuitforge_core.text.app \
    --model /path/to/model.gguf \
    --port 8006 \
    --gpu-id 0
```

4-bit quantisation (GGUF files ending in `q4_k_m`, `q4_0`, etc.) runs well on 6–8GB VRAM. Full-precision (`f16`) requires more.

Multi-GPU (splits across two GPUs via `device_map=auto`):

```bash
python -m circuitforge_core.text.app \
    --model /path/to/large-model \
    --port 8006 \
    --gpu-ids 0,1
```

---

### LLM inference (HuggingFace transformers)

```bash
pip install "circuitforge-core[text-transformers]"
# 4-bit quantisation (bitsandbytes):
pip install "circuitforge-core[text-transformers-4bit]"
```

```bash
python -m circuitforge_core.text.app \
    --model /path/to/model-or-hf-repo \
    --backend transformers \
    --port 8006
```

---

### VLM inference (GGUF with mmproj)

LLaVA-style models (LLaVA, BakLLaVA, llava-phi) require a separate projector file (`--mmproj`):

```bash
python -m circuitforge_core.text.app \
    --model /path/to/llava-model.gguf \
    --mmproj /path/to/mmproj.gguf \
    --port 8006 \
    --gpu-id 0
```

Embedded VLMs (Qwen2-VL, MiniCPM-V, Moondream) have the projector baked in — no `--mmproj` needed.

Sending images via the multimodal API:

```json
POST /chat
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<b64>"}},
        {"type": "text", "text": "What is in this document?"}
      ]
    }
  ]
}
```

Sending an image to a text-only model returns HTTP 422.

---

### Classifier / PII filter

```bash
pip install "circuitforge-core[text-transformers]"
```

```bash
python -m circuitforge_core.text.app \
    --backend classifier \
    --model dslim/bert-base-NER \
    --port 8006
```

Recommended model for English PII detection: `dslim/bert-base-NER`. Substituting other HuggingFace NER models is supported.

Calling the filter endpoint:

```json
POST /filter
{
  "text": "Please contact John Smith at john@example.com.",
  "mode": "redact"
}
```

Modes: `redact` (replace spans with `[REDACTED]`), `detect` (return boolean), `spans` (return span list with labels and confidence).

---

### Mock mode (no model required)

```bash
CF_TEXT_MOCK=1 python -m circuitforge_core.text.app --port 8006
```

Returns deterministic canned responses for all endpoints. No GPU, no model download. Suitable for CI and integration testing.

---

### Configuration

| Variable | Default | Description |
|---|---|---|
| `CF_TEXT_URL` | — | URL products use to reach cf-text (e.g. `http://localhost:8006`) |
| `CF_TEXT_MOCK` | — | Set to `1` to enable mock mode |

CLI flags: `--model`, `--backend` (`llamacpp`/`transformers`/`classifier`/`mock`), `--port`, `--gpu-id`, `--gpu-ids`, `--mmproj`.

---

### API endpoints

| Endpoint | Backend | Description |
|---|---|---|
| `GET /health` | all | `{"status":"ok","model":str,"backend":str,"vram_mb":int}` |
| `POST /generate` | text-gen | Single prompt completion |
| `POST /chat` | text-gen | OpenAI-compatible chat (supports multimodal content blocks) |
| `POST /v1/chat/completions` | text-gen | OpenAI-compatible alias for `/chat` |
| `POST /filter` | classifier | PII detection and redaction |

---

### Connecting from a product

```bash
CF_TEXT_URL=http://localhost:8006
```

Products using cf-core's LLM router pick this up automatically when the `text` backend is enabled in `config/llm.yaml`.
