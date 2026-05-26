"""
circuitforge_core.video — cf-video service: video VLM inference via Marlin-2B.

Exposes a FastAPI process (managed by cf-orch) with endpoints:
  GET  /health    → {"status": "ok", "model": str, "vram_mb": int}
  POST /caption   → CaptionResult (scene description + timestamped events)
  POST /find      → FindResult (temporal grounding span for a natural-language event)

Run as:
    python -m circuitforge_core.video.app --model /path/to/NemoStation--Marlin-2B --port 8016 --gpu-id 0
"""
