# circuitforge_core.video

Video captioning and temporal grounding service using [Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B) (Apache 2.0).

## What it does

- **Caption:** Produces a scene summary and a timestamped list of detected events for a video file.
- **Find:** Grounds a natural-language event description to a time span within the video.

## Prerequisites

### Hardware

| GPU VRAM | Result |
|----------|--------|
| 16GB+ | Recommended for full-precision inference |
| 12GB | Minimum for most videos |
| Under 12GB | OOM likely on longer clips |

CPU mode is not supported — Marlin-2B requires a CUDA-capable GPU.

### CUDA version

```bash
nvidia-smi | grep "CUDA Version"
```

| CUDA version | Install path |
|---|---|
| 12.x or earlier | Standard install — see below |
| 13.x (RTX 50-series / Blackwell) | PyTorch nightly required — see below |

### Security note

Marlin-2B requires `trust_remote_code=True`. Review the model's `modeling_marlin.py` on HuggingFace before deploying on a production node. The model is Apache 2.0 and the source is auditable at [huggingface.co/NemoStation/Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B).

---

## Install

Standard (CUDA 12.x):

```bash
pip install "circuitforge-core[video-service]"
```

CUDA 13.x (RTX 50-series / Blackwell) — PyTorch nightly required:

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cu130 torch torchvision
pip install "circuitforge-core[video-service]" --no-deps
pip install transformers>=5.7.0 torchcodec "qwen-vl-utils>=0.0.14" av Pillow accelerate fastapi "uvicorn[standard]"
```

---

## Running the service

Download the model to a local path first (one-time, approximately 4–6 GB):

```bash
huggingface-cli download NemoStation/Marlin-2B --local-dir /path/to/models/Marlin-2B
```

Start the service:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID python -m circuitforge_core.video.app \
    --model /path/to/models/Marlin-2B \
    --port 8016 \
    --gpu-id 0
```

The service blocks at startup until the model is loaded, then prints ready status. Confirm:

```bash
curl http://localhost:8016/health
# {"status": "ok", "model": "/path/to/models/Marlin-2B", "vram_mb": ...}
```

Point products at the service with:

```bash
CF_VIDEO_URL=http://localhost:8016
```

---

## API reference

### `GET /health`

Returns 200 when model is loaded. `vram_mb` is the GPU memory in use.

```json
{"status": "ok", "model": "/models/Marlin-2B", "vram_mb": 4200}
```

### `POST /caption`

Generate a scene summary and timestamped events for a video.

> **Important:** `video_path` must be an absolute path on the machine running cf-video — not the calling machine. If cf-video runs in Docker, mount your video directory into the container and use the container-side path.

**Request:**

```json
{
  "video_path": "/absolute/path/to/video.mp4",
  "max_new_tokens": 2048
}
```

**Response:**

```json
{
  "scene": "A kitchen scene where someone prepares pasta.",
  "events": [
    {"start": 0.0, "end": 4.5, "description": "Filling pot with water"},
    {"start": 4.5, "end": 12.0, "description": "Boiling water on stovetop"}
  ],
  "caption": "Kitchen cooking scene with pasta preparation steps.",
  "model": "/models/Marlin-2B"
}
```

### `POST /find`

Ground a natural-language event description to a time span.

**Request:**

```json
{
  "video_path": "/absolute/path/to/video.mp4",
  "event": "person adds salt to the water",
  "max_new_tokens": 256
}
```

**Response:**

```json
{
  "span": [8.2, 10.6],
  "format_ok": true,
  "raw": "[8.2, 10.6]",
  "model": "/models/Marlin-2B"
}
```

`span` is `null` when the model cannot ground the event in the video. `format_ok` indicates whether the model produced a parseable time range.

---

## Docker Compose setup

```yaml
# compose.yml excerpt
services:
  cf-video:
    image: ghcr.io/circuit-forge/cf-video:latest   # or build locally
    network_mode: host
    environment:
      CF_VIDEO_MODEL: /models/Marlin-2B
      CF_VIDEO_PORT: "8016"
    volumes:
      - /path/to/models/Marlin-2B:/models/Marlin-2B:ro
      - /path/to/your/videos:/videos:ro              # mount video storage
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

Pass video paths relative to the container mount:

```json
{"video_path": "/videos/my-video.mp4", "event": "person enters room"}
```

---

## Troubleshooting

**`CUDA out of memory`**
Marlin-2B requires 12GB+ VRAM. No CPU fallback is available.

**`No such file or directory: /home/user/video.mp4`**
`video_path` is resolved on the server, not the client. If cf-video runs in Docker, you must mount the directory containing the video into the container and use the container-side path.

**CUDA version mismatch**
RTX 50-series (Blackwell) cards use CUDA 13. Standard PyTorch stable does not support CUDA 13 — install PyTorch nightly as described in Prerequisites.

**`trust_remote_code` errors**
Make sure `transformers >= 5.7.0` is installed. Older versions do not support the Marlin architecture registration.

---

## License

- cf-video service code: MIT — CircuitForge LLC
- Marlin-2B model: [Apache 2.0](https://huggingface.co/NemoStation/Marlin-2B) — NemoStation
