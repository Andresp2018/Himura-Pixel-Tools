"""Recommended model registry (spec.model_registry_recommendations).

Each entry mirrors the spec's structure: id, role, url, type, why, and notes.
Used by the downloader and surfaced in the model-install UI.
"""

from __future__ import annotations

from typing import Literal

RegType = Literal["base", "vae", "lora", "controlnet", "ip_adapter", "motion", "segmentation", "upscale"]

# Flat registry combining every stack section of the spec.
REGISTRY: list[dict] = [
    # primary_generation_stack
    {"id": "stabilityai/stable-diffusion-xl-base-1.0", "type": "base",
     "role": "Base text-to-image and image-to-image model",
     "license": "CreativeML Open RAIL++-M",
     "base_compatibility": ["sdxl"],
     "notes": "Use at 768 or 1024 internal source size, then pixel-process to exact canvas."},
    {"id": "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors",
     "type": "lora",
     "role": "SDXL-Lightning 4-step acceleration LoRA",
     "license": "openrail++",
     "base_compatibility": ["sdxl"],
     "notes": "NOT a standalone base â€” apply on top of an SDXL base for 4-step fast drafts "
              "(set as the pixel/accel LoRA, guidance ~1). The full SDXL-Lightning repo ships "
              "only UNet/LoRA accelerator files, so it can't be selected as a base model."},
    {"id": "latent-consistency/lcm-lora-sdxl", "type": "lora",
     "role": "Acceleration adapter (LCM)",
     "license": "MIT",
     "base_compatibility": ["sdxl"],
     "notes": "Reduces SDXL inference steps."},

    # pixel-art style LoRAs (research: nerijs Pixel Art XL + FLUX pixel LoRAs).
    # Set one as the active pixel LoRA in Settings to push the base model toward
    # blocky, anti-alias-free output before the true-pixel pipeline runs.
    {"id": "nerijs/pixel-art-xl", "type": "lora",
     "role": "Foundational SDXL pixel-art style LoRA",
     "license": "Apache-2.0",
     "base_compatibility": ["sdxl"],
     "trigger": "pixel",
     "notes": "The standard SDXL pixel LoRA (Civitai 120096). Generate at 1024 then "
              "downscale ~8x; works well with no special trigger. Pair with the fp16-fix VAE."},
    {"id": "prithivMLmods/Retro-Pixel-Flux-LoRA", "type": "lora",
     "role": "FLUX retro 8-bit pixel-art LoRA",
     "license": "creativeml-openrail-m",
     "base_compatibility": ["flux_optional_future"],
     "trigger": "Retro Pixel",
     "notes": "FLUX.1-dev pixel LoRA; trigger 'Retro Pixel' for stringent 8-bit limitations."},

    # Community Civitai pixel-art LoRAs (downloaded via the Civitai API).
    {"id": "https://civitai.com/models/945266/pixel-game-assets-flux-by-dever", "type": "lora",
     "role": "FLUX pixel game-assets LoRA (Dever)",
     "license": "see Civitai model page",
     "base_compatibility": ["flux_optional_future"],
     "trigger": "dvr-pixel-flux",
     "notes": "Civitai 945266. Game item/asset pixel art for FLUX.1-dev."},
    {"id": "https://civitai.com/models/1776030/perfect-pixelart-x8-16bit-style", "type": "lora",
     "role": "16-bit pixel-art style LoRA (Perfect PixelArt x8)",
     "license": "see Civitai model page",
     "base_compatibility": ["sdxl"],
     "trigger": "pixelart, 16bit style",
     "notes": "Civitai 1776030 (Illustrious/SDXL). Designed for an 8x downscale to true pixels."},
    {"id": "https://civitai.com/models/1936887/game-character-sprites-assets-generator-retro-rpg-video-game-dev-2d-pixel-art",
     "type": "lora",
     "role": "Character sprite-sheet / asset generator LoRA",
     "license": "see Civitai model page",
     "base_compatibility": ["sdxl"],
     "trigger": "pixel_character_sprite, sprite sheet, multiple views",
     "notes": "Civitai 1936887 (Illustrious/SDXL). Multi-view retro RPG character sprites."},
    {"id": "https://civitai.com/models/165876/2d-pixel-toolkit-2d", "type": "lora",
     "role": "2D pixel toolkit sprite LoRA",
     "license": "see Civitai model page",
     "base_compatibility": ["sd15"],
     "trigger": "pixel, pixel art, pixelart",
     "notes": "Civitai 165876 (SD 1.5). General 2D pixel sprites."},
    {"id": "black-forest-labs/FLUX.1-dev", "type": "base",
     "role": "FLUX.1 Dev base model for FLUX.1 LoRAs",
     "license": "flux-1-dev-non-commercial-license",
     "base_compatibility": ["flux_optional_future"],
     "notes": "Use this base when selecting FLUX.1 pixel-art LoRAs such as Retro-Pixel-Flux-LoRA or the Dever Civitai LoRA. May require Hugging Face access to the model."},
    {"id": "https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_K_M.gguf", "type": "base",
     "role": "Low-VRAM FLUX.2 Klein 4B GGUF transformer",
     "license": "Apache-2.0",
     "base_compatibility": ["flux_optional_future"],
     "notes": "Q4_K_M is the practical first choice for 6GB VRAM; requires diffusers GGUF support and black-forest-labs/FLUX.2-klein-4B text/vae assets."},
    {"id": "leejet/ideogram-4-GGUF", "type": "base",
     "role": "Ideogram 4 GGUF bundle for stable-diffusion.cpp",
     "license": "see Hugging Face model card",
     "base_compatibility": ["ideogram_gguf"],
     "notes": "Downloads ideogram4-Q4_0.gguf and ideogram4_uncond-Q4_0.gguf. The current in-app Diffusers runtime indexes and validates these files, while generation needs the planned stable-diffusion.cpp execution backend."},
    {"id": "madebyollin/sdxl-vae-fp16-fix", "type": "vae",
     "role": "VAE for fp16 stability",
     "license": "MIT",
     "base_compatibility": ["sdxl"],
     "notes": "Default VAE for SDXL profiles; avoids fp16 NaN/quality issues."},

    # consistency_and_control_stack
    {"id": "h94/IP-Adapter", "type": "ip_adapter",
     "role": "Reference-image identity/style lock",
     "license": "Apache-2.0",
     "base_compatibility": ["sdxl"],
     "notes": "Image prompt conditioning for consistent character/style references."},
    {"id": "diffusers/controlnet-canny-sdxl-1.0-small", "type": "controlnet",
     "role": "Small SDXL Canny/edge ControlNet",
     "license": "openrail++",
     "base_compatibility": ["sdxl"],
     "notes": "Lighter spatial control for silhouettes, item shapes, frame cleanup."},
    {"id": "thibaud/controlnet-openpose-sdxl-1.0", "type": "controlnet",
     "role": "Pose-conditioned character frame generation",
     "license": "see model card",
     "base_compatibility": ["sdxl"],
     "notes": "Pose control for humanoid characters and animation keyframes."},
    {"id": "xinsir/controlnet-openpose-sdxl-1.0", "type": "controlnet",
     "role": "Alternative pose ControlNet (Apache-2.0)",
     "license": "Apache-2.0",
     "base_compatibility": ["sdxl"],
     "notes": "Benchmark against thibaud/openpose for pixel-art outputs."},

    # FLUX conditioning stack (research: add ControlNet + IP-Adapter to FLUX).
    # These target FLUX.1-dev; FLUX.2-klein adapter support depends on community
    # models. The loader auto-uses FluxControlNetModel / the XLabs FLUX IP-Adapter
    # when the active base is a FLUX model.
    {"id": "InstantX/FLUX.1-dev-Controlnet-Union", "type": "controlnet",
     "role": "FLUX union ControlNet (pose/canny/depth/tile)",
     "license": "flux-1-dev-non-commercial-license",
     "base_compatibility": ["flux_optional_future"],
     "notes": "Multi-mode FLUX.1 ControlNet (includes pose). Loaded via FluxControlNetModel."},
    {"id": "XLabs-AI/flux-ip-adapter", "type": "ip_adapter",
     "role": "FLUX reference-image identity/style adapter",
     "license": "flux-1-dev-non-commercial-license",
     "base_compatibility": ["flux_optional_future"],
     "notes": "FLUX IP-Adapter (uses openai/clip-vit-large-patch14 image encoder). "
              "Loaded automatically when the active base is FLUX."},

    # animation_stack
    {"id": "guoyww/animatediff-motion-adapter-sdxl-beta", "type": "motion",
     "role": "Optional motion prior for rough animation drafts",
     "license": "Apache-2.0",
     "base_compatibility": ["sdxl"],
     "notes": "Draft motion only; final frames go through deterministic alignment."},

    # segmentation (background removal)
    {"id": "ZhengPeng7/BiRefNet_lite", "type": "segmentation",
     "role": "Background removal",
     "license": "MIT",
     "base_compatibility": ["sdxl", "sd15"],
     "notes": "Used by true-pixel pipeline step 2 when model-based masking is enabled."},
]

# Convenience: what to grab with ``--all`` on first run. Includes the
# consistency/control/segmentation stack so characters stay on-model, rotate,
# and get clean transparent backgrounds out of the box (pixellab-style).
DEFAULT_DOWNLOAD_SET = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "madebyollin/sdxl-vae-fp16-fix",
    "h94/IP-Adapter",
    "thibaud/controlnet-openpose-sdxl-1.0",
    "ZhengPeng7/BiRefNet_lite",
]

# A minimal fast set for a quick first experience.
FAST_DOWNLOAD_SET = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "madebyollin/sdxl-vae-fp16-fix",
    "latent-consistency/lcm-lora-sdxl",
]


def _normalize_registry_id(value: str) -> str:
    s = (value or "").strip()
    if s.startswith(("http://", "https://")) and "huggingface.co" in s.lower():
        path = s.split("huggingface.co/", 1)[1].split("?", 1)[0].strip("/")
        for marker in ("/resolve/", "/tree/", "/blob/"):
            if marker in path:
                path = path.split(marker, 1)[0]
        parts = path.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
    return s


def by_id(model_id: str) -> dict | None:
    wanted = _normalize_registry_id(model_id)
    for r in REGISTRY:
        if r["id"] == model_id or _normalize_registry_id(r["id"]) == wanted:
            return r
    return None




