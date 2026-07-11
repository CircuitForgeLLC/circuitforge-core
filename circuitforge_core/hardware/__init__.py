# circuitforge_core/hardware/__init__.py
"""
Hardware detection and LLM profile generation.

Typical usage::

    from circuitforge_core.hardware import detect_hardware, generate_profile
    import yaml

    spec = detect_hardware()
    config = generate_profile(spec)
    print(yaml.dump(config.to_dict()))
    print("Recommended profile:", config.profile_name)
"""
from .detect import detect_hardware, detect_hardware_json
from .generator import generate_profile
from .models import HardwareSpec, LLMBackendConfig, LLMConfig
from .tiers import VRAM_TIERS, VramTier, select_tier
from .vram_estimate import ModelVramLookupError, VramEstimate, model_vram_estimate

__all__ = [
    "detect_hardware",
    "detect_hardware_json",
    "generate_profile",
    "HardwareSpec",
    "LLMBackendConfig",
    "LLMConfig",
    "VRAM_TIERS",
    "VramTier",
    "select_tier",
    "ModelVramLookupError",
    "VramEstimate",
    "model_vram_estimate",
]
