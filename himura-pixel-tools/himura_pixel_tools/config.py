"""Central configuration and path resolution for Himura Pixel Tools.

All user data lives under a single platform-appropriate user directory, with a
layout that matches the spec's ``model_store_layout`` and ``database_schema``.
Nothing here touches the network.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Version / identity ────────────────────────────────────────────────────────

APP_NAME = "Himura Pixel Tools"
APP_VERSION = "1.0.0"
API_BASE_DEFAULT = "http://127.0.0.1:8765"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MCP_PATH = "/mcp"

# ── Path resolution ───────────────────────────────────────────────────────────

# Project root = the folder that contains the himura_pixel_tools package.
# Everything (models, HF cache, DB, outputs) lives INSIDE the project folder so
# the whole app is fully self-contained and portable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    """Resolve the data root.

    Order of precedence:
      1. HIMURA_DATA_DIR env var (absolute path override)
      2. <project_root>/himura_data   (default — inside the project folder)

    This keeps models, HF cache, DB and outputs all inside the project folder
    instead of scattering them across %LOCALAPPDATA% and ~/.cache.
    """
    env = os.environ.get("HIMURA_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_PROJECT_ROOT / "himura_data").resolve()


DATA_ROOT = _data_root()
MODELS_ROOT = DATA_ROOT / "models"
PROJECTS_ROOT = DATA_ROOT / "projects"
DB_PATH = DATA_ROOT / "himura.db"
CONFIG_PATH = DATA_ROOT / "config.json"
CACHE_ROOT = DATA_ROOT / "cache"
LOGS_ROOT = DATA_ROOT / "logs"
TOKEN_PATH = DATA_ROOT / "mcp_token"  # bearer token for HTTP MCP

# Hugging Face cache is redirected INSIDE the project folder too, so model
# downloads never leak into the global ~/.cache/huggingface. Set at import time
# (before huggingface_hub is imported anywhere).
HF_HOME = DATA_ROOT / "hf_cache"
HF_HUB_CACHE = HF_HOME / "hub"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
os.environ.setdefault("DIFFUSERS_CACHE", str(HF_HOME / "diffusers"))

# Model store subfolders, matching the spec's model_store_layout.
MODEL_SUBDIRS = {
    "base": "Base diffusion models, e.g. SDXL",
    "vae": "VAEs",
    "lora": "style, pixel-art, and character LoRAs",
    "controlnet": "pose, canny, lineart, depth control models",
    "ip_adapter": "reference image adapters and image encoders",
    "motion": "animation/motion adapters",
    "segmentation": "background removal and mask models",
    "upscale": "optional pixel-preview upscalers",
}


def ensure_dirs() -> None:
    """Create the user-data directory tree. Safe to call repeatedly."""
    for p in (DATA_ROOT, MODELS_ROOT, PROJECTS_ROOT, CACHE_ROOT, LOGS_ROOT):
        p.mkdir(parents=True, exist_ok=True)
    for sub in MODEL_SUBDIRS:
        (MODELS_ROOT / sub).mkdir(parents=True, exist_ok=True)


def model_dir(kind: str) -> Path:
    """Return the model store directory for a model type."""
    kind = kind.lower()
    if kind not in MODEL_SUBDIRS:
        raise ValueError(f"Unknown model kind: {kind!r}")
    d = MODELS_ROOT / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Runtime config (persisted JSON) ───────────────────────────────────────────


@dataclass
class RuntimeConfig:
    """Persisted runtime preferences, editable from the desktop UI."""

    precision: str = "fp16"           # fp16 | bf16 | fp32
    attention_slicing: bool = True
    vae_tiling: bool = True
    sequential_cpu_offload_when_low_vram: bool = True
    max_parallel_jobs: int = 1         # spec: keep 1 on RTX 3060 12GB
    default_quality_preset: str = "quality"   # quality | fast (LCM/Lightning)
    auto_download_models_at_startup: bool = False
    default_output_root: str = ""
    mirror_outputs_to_final: bool = True
    mcp_http_enabled: bool = True
    mcp_require_token: bool = True
    last_base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # ── generation provider ──────────────────────────────────────────────
    # "local" runs the on-GPU SDXL/FLUX pipeline (default). "gemini" is an
    # OPTIONAL external image source: only used when a Gemini API key is set,
    # and only for the source image — the same local true-pixel pipeline still
    # quantizes/snaps/segments it, so output is identical-format pixel art.
    generation_provider: str = "local"       # local | gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-image"

    # Optional Civitai API token — only needed to download LoRAs/models from
    # civitai.com that require authentication. Public models download without it.
    civitai_api_key: str = ""

    # ── optional pixel-art LoRA (research: nerijs Pixel Art XL etc.) ──────
    # When set to an installed LoRA model id, it is auto-attached before each
    # local SDXL/FLUX generation to push the base model toward blocky output.
    pixel_lora_id: str = ""
    pixel_lora_weight: float = 1.0

    extras: dict = field(default_factory=dict)

    def gemini_enabled(self) -> bool:
        """True only when the user explicitly chose Gemini AND provided a key."""
        return (self.generation_provider or "local").lower() == "gemini" \
            and bool((self.gemini_api_key or "").strip())

    @classmethod
    def load(cls) -> "RuntimeConfig":
        ensure_dirs()
        if CONFIG_PATH.exists():
            try:
                raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__ or k == "extras"}
                    extras = known.pop("extras", raw.get("extras", {}))
                    cfg = cls(**known)
                    if isinstance(extras, dict):
                        cfg.extras = extras
                    return cfg
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        ensure_dirs()
        CONFIG_PATH.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ── Exact-size targets (from spec.core_product_features) ─────────────────────

EXACT_SIZE_TARGETS = [
    {"name": "tiny_icon", "width": 16, "height": 16},
    {"name": "small_icon", "width": 24, "height": 24},
    {"name": "rpg_item", "width": 32, "height": 32},
    {"name": "rpg_character", "width": 48, "height": 48},
    {"name": "standard_character", "width": 64, "height": 64},
    {"name": "large_character", "width": 96, "height": 96},
    {"name": "portrait_or_ui", "width": 128, "height": 128},
]

ASSET_TYPES = [
    "item", "prop", "character", "building", "character_turnaround",
    "character_portrait", "character_animation", "tileset_top_down",
    "tileset_sidescroller", "isometric_tile", "background_scene",
    "ui_element", "ui_pack", "inpaint_edit", "rotation_sheet",
]

# Backdrop hint. We no longer ask for a colored chroma backdrop (the model often
# ignored it, leaving a scene behind). We ask for a plain, uncluttered backdrop
# and rely on the segmentation model to actually cut the subject out.
SOLID_BG_HINT = (
    "isolated on a plain flat background, no scenery, no ground, no floor, "
    "no platform, no cast shadow"
)

# Per-asset-type pipeline profiles. Each category is steered as its own kind of
# asset — the right *view* (orthographic vs isometric), framing, alignment and a
# category-specific negative — instead of one generic "make pixel art" path.
#   pos     : positive phrase describing the subject + view
#   neg     : category-specific negative terms
#   align   : "bottom" (feet/base planted) | "center"
#   margin  : subject margin inside the canvas
#   isolated: single subject that must be cut out of its background
_GROUND_NEG = "ground, floor, platform, pedestal, base, grass, dirt, terrain, cast shadow, drop shadow, contact shadow"
_ORTHO_NEG = "isometric, 3d render, perspective view, diagonal view, depth, vanishing point"
_TOPDOWN_NEG = "isometric, diamond tile, side view, side faces, front view, horizon, perspective, vanishing point, walls"
_ISO_NEG = "top-down overhead map texture, flat square texture, straight front view, side-scroller platform, horizon"
ASSET_TYPE_PROFILES = {
    "item": {"pos": "a single game item icon, orthographic front view, centered, isolated object",
             "neg": _ORTHO_NEG + ", " + _GROUND_NEG + ", character, scene, multiple items",
             "align": "center", "margin": 0.12, "isolated": True},
    "prop": {"pos": "a single game prop, orthographic front view, centered, isolated object",
             "neg": _ORTHO_NEG + ", " + _GROUND_NEG + ", character, scene",
             "align": "center", "margin": 0.10, "isolated": True},
    "character": {"pos": "a single full-body character sprite, orthographic front view, "
                         "standing upright, facing the camera, full body visible, centered",
                  "neg": _ORTHO_NEG + ", " + _GROUND_NEG + ", multiple characters, cropped, cut off, portrait, bust",
                  "align": "bottom", "margin": 0.08, "isolated": True},
    "building": {"pos": "a single building, flat orthographic front elevation, 2D front-facing "
                        "facade, straight-on front view, standalone structure, centered",
                 "neg": _ORTHO_NEG + ", top-down, aerial, scene, city, landscape, multiple buildings",
                 "align": "bottom", "margin": 0.06, "isolated": True},
    "ui_element": {"pos": "clean empty game UI component, flat front view, readable border, no character",
                   "neg": _ORTHO_NEG + ", character, face, creature, item icon, scene, landscape, clutter, tiny sprites",
                   "align": "center", "margin": 0.04, "isolated": True},
    "background_scene": {"pos": "a game background scene, scenery, environment art",
                         "neg": "", "align": "center", "margin": 0.0, "isolated": False},
    "character_turnaround": {"pos": "a single full-body character sprite, full body visible, centered",
                             "neg": _ORTHO_NEG + ", " + _GROUND_NEG + ", multiple characters, cropped, portrait",
                             "align": "bottom", "margin": 0.08, "isolated": True},
    "character_animation": {"pos": "a single full-body character sprite, full body visible, centered",
                            "neg": _ORTHO_NEG + ", " + _GROUND_NEG + ", multiple characters, cropped",
                            "align": "bottom", "margin": 0.08, "isolated": True},
    "character_portrait": {"pos": "a character portrait, head and shoulders bust, face close-up, "
                                  "front view, centered",
                           "neg": "full body, legs, feet, distant, tiny, sprite sheet",
                           "align": "center", "margin": 0.04, "isolated": True},
    "map_object": {"pos": "a single top-down map object, strict overhead orthographic view, "
                          "camera looking straight down, centered, isolated",
                   "neg": _TOPDOWN_NEG + ", " + _GROUND_NEG + ", character, scene",
                   "align": "center", "margin": 0.08, "isolated": True},
    "isometric_tile": {"pos": "a single true isometric 2:1 diamond tile, visible top face and "
                              "short side faces, axonometric game tile, transparent corners",
                       "neg": _ISO_NEG + ", square tile, flat map texture",
                       "align": "center", "margin": 0.02, "isolated": True},
    "tileset_top_down": {"pos": "strict top-down orthographic overhead game tile texture, "
                                "camera looking straight down, square seamless map tile",
                         "neg": _TOPDOWN_NEG,
                         "align": "center", "margin": 0.0, "isolated": False},
    "tiles_pro": {"pos": "strict game-ready tile, exact camera view requested, clean readable material",
                  "neg": "wrong camera view, mixed perspective, blurry seams, text, watermark",
                  "align": "center", "margin": 0.0, "isolated": False},
}

# Default profile for any type not listed above.
_DEFAULT_PROFILE = {"pos": "centered, isolated", "neg": "", "align": "center",
                    "margin": 0.06, "isolated": True}

# Back-compat: the set of single-subject types (used for segmentation toggles).
ISOLATED_ASSET_TYPES = {k for k, v in ASSET_TYPE_PROFILES.items() if v["isolated"]}
ISOLATED_NEGATIVE = ("multiple characters, group, scene, landscape, background scenery, "
                     "tilemap, level map, collage, montage, frame border, grid")


def asset_profile(asset_type: str) -> dict:
    """Return the pipeline profile (view/framing/alignment) for an asset type."""
    return ASSET_TYPE_PROFILES.get(asset_type, _DEFAULT_PROFILE)


def _trim_words(value: str, max_words: int) -> str:
    words = str(value or "").replace("\n", " ").split()
    return " ".join(words[:max_words])


def build_prompt(prompt: str, asset_type: str, *, solid_bg: bool = True) -> str:
    """Compose a compact positive prompt that keeps pixel-style terms in range.

    SDXL CLIP truncates around 77 tokens. Put the essential pixel-art/style and
    camera terms first, then cap user text so the suffix is not silently lost.
    """
    prof = asset_profile(asset_type)
    style = "pixel art, game sprite, limited palette, hard edges"
    user = _trim_words(str(prompt or "").strip().rstrip(","), 34)
    parts = [style, user, prof["pos"]]
    if solid_bg and prof["isolated"]:
        parts.append("plain background, no scenery, no floor, no shadow")
    return ", ".join(p for p in parts if p)


def build_negative(negative: str | None, asset_type: str) -> str:
    """Compose the negative prompt: base + category-specific + isolation guards."""
    prof = asset_profile(asset_type)
    parts = [(negative or DEFAULT_NEGATIVE_PROMPT).strip().rstrip(",")]
    if prof["neg"]:
        parts.append(prof["neg"])
    if prof["isolated"]:
        parts.append(ISOLATED_NEGATIVE)
    return ", ".join(p for p in parts if p)

EXPORT_FORMATS = [
    "PNG exact production asset",
    "PNG 4x preview using nearest-neighbor only",
    "PNG 8x preview using nearest-neighbor only",
    "sprite sheet PNG",
    "GIF preview",
    "WebP preview",
    "Aseprite-compatible JSON metadata",
    "Unity metadata JSON",
    "Godot metadata JSON",
    "Phaser/TexturePacker-style atlas JSON",
    "ZIP export pack",
]

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, smooth, antialiased, anti-aliasing, soft edges, gradient, "
    "sloppy, messy, noisy, jpeg artifacts, realistic, photographic, 3d render, "
    "low quality, extra limbs, deformed, watermark, text"
)

DEFAULT_PIXEL_SUFFIX = ", pixel art, 8-bit style, game sprite, limited palette, hard edges"


def resolve_size(width, height):
    """Return integer (w, h); 'custom'/'user_defined'/'None' allowed."""
    try:
        w = int(width)
    except (TypeError, ValueError):
        w = 64
    try:
        h = int(height)
    except (TypeError, ValueError):
        h = 64
    return max(1, w), max(1, h)


def name_for_size(w: int, h: int) -> str:
    for t in EXACT_SIZE_TARGETS:
        if t["width"] == w and t["height"] == h:
            return t["name"]
    return "custom"
