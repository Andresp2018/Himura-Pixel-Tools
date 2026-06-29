"""Job / generation request & response schemas (spec.api_contract)."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "succeeded", "failed", "needs_review", "cancelled"]
AssetType = Literal[
    "item", "prop", "character", "building", "character_turnaround",
    "character_portrait", "character_animation", "tileset_top_down",
    "tileset_sidescroller", "isometric_tile", "background_scene",
    "ui_element", "ui_pack", "inpaint_edit", "rotation_sheet",
    "map_object", "tiles_pro",
]


class TargetSize(BaseModel):
    width: int | str = "user_defined"
    height: int | str = "user_defined"


class SavedOutputOptions(BaseModel):
    exclude_from_saved_outputs: bool = False
    # Optional per-job style LoRA (selected per tab in the UI). Falls back to the
    # global pixel_lora_id in Settings when unset. Must be base-compatible.
    lora_id: Optional[str] = None
    lora_weight: Optional[float] = None


class GenerateRequest(SavedOutputOptions):
    asset_type: AssetType = "item"
    prompt: str
    target_size: TargetSize = Field(default_factory=lambda: TargetSize(width=64, height=64))
    transparent: bool = True
    style_profile_id: Optional[str] = None
    character_profile_id: Optional[str] = None
    directions: Optional[int] = 4
    seed: Optional[int] = None
    palette_limit: Optional[int] = 24
    output_root: Optional[str] = None
    generate_preview_scale: list[int] = Field(default_factory=lambda: [4, 8])
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    quality_preset: Optional[str] = None   # quality | fast
    # research-driven palette controls
    palette_preset: Optional[str] = None   # gameboy | nes | c64 | pico8 | ...
    dither: Optional[str] = None           # none | bayer2 | bayer4 | bayer8
    dither_strength: Optional[float] = None  # 0..1
    protect_extremes: bool = False         # keep outline/glint colours


class JobOutputs(BaseModel):
    production_png: Optional[str] = None
    preview_png: Optional[str] = None
    preview_png_8x: Optional[str] = None
    sprite_sheet_png: Optional[str] = None
    metadata_json: Optional[str] = None
    gif_preview: Optional[str] = None
    webp_preview: Optional[str] = None
    zip_path: Optional[str] = None
    files: list[str] = Field(default_factory=list)
    final_files: list[str] = Field(default_factory=list)
    final_output_folder: Optional[str] = None
    named_files: dict[str, Optional[str]] = Field(default_factory=dict)
    turnaround_pngs: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Optional[str]] = Field(default_factory=dict)


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    outputs: JobOutputs = Field(default_factory=JobOutputs)
    warnings: list[str] = Field(default_factory=list)
    model_profile_id: Optional[str] = None
    seed: Optional[int] = None
    error: Optional[str] = None
    created_at: Optional[str] = None


class CreateCharacterRequest(SavedOutputOptions):
    name: str
    description: str
    width: int = 64
    height: int = 64
    directions: int = 4
    style_profile_id: Optional[str] = None
    output_folder: Optional[str] = None
    prompt: Optional[str] = None
    seed: Optional[int] = None


class TurnaroundRequest(SavedOutputOptions):
    character_profile_id: str
    directions: int = 4
    width: Optional[int] = None
    height: Optional[int] = None
    output_folder: Optional[str] = None


class PortraitRequest(SavedOutputOptions):
    character_profile_id: str
    width: int = 128
    height: int = 128
    palette_limit: int = 32
    expression: Optional[str] = None          # e.g. "neutral", "angry", "smiling"
    transparent: bool = True
    output_folder: Optional[str] = None
    seed: Optional[int] = None


class AnimateRequest(SavedOutputOptions):
    character_profile_id: str
    animation: str = "idle"
    directions: int = 4
    frames_per_direction: Optional[int] = None
    output_folder: Optional[str] = None


class InpaintRequest(SavedOutputOptions):
    asset_path: str
    mask_path: str
    edit_prompt: str
    character_profile_id: Optional[str] = None
    output_folder: Optional[str] = None
    strength: float = 0.95


class TilesetRequest(SavedOutputOptions):
    description: str
    tile_width: int = 16
    tile_height: int = 16
    tileset_type: Literal["top_down", "sidescroller", "isometric", "wang"] = "top_down"
    tile_count: Optional[int] = 8
    output_folder: Optional[str] = None
    seamless: bool = True


class ObjectRequest(SavedOutputOptions):
    description: str
    size: int = 64
    view: Literal["top-down", "sidescroller", "low top-down", "high top-down", "side"] = "top-down"
    output_folder: Optional[str] = None
    seed: Optional[int] = None


class ObjectRotationRequest(SavedOutputOptions):
    description: str
    size: int = 64
    view: Literal["low top-down", "high top-down", "side"] = "low top-down"
    output_folder: Optional[str] = None
    seed: Optional[int] = None


class ObjectStateRequest(SavedOutputOptions):
    object_id: Optional[str] = None
    edit_description: str
    source_image: Optional[str] = None      # data URL/base64 upload fallback
    output_folder: Optional[str] = None
    seed: Optional[int] = None


class MapObjectRequest(SavedOutputOptions):
    description: str
    width: int = 64
    height: int = 64
    view: Literal["low top-down", "high top-down", "side"] = "low top-down"
    detail: Optional[Literal["low detail", "medium detail", "high detail"]] = None
    output_folder: Optional[str] = None
    seed: Optional[int] = None


class UIAssetRequest(SavedOutputOptions):
    description: str
    name: Optional[str] = None
    width: int = 256
    height: int = 128
    color_palette: Optional[str] = None
    elements: list[str] = Field(default_factory=list)
    no_background: bool = True
    output_folder: Optional[str] = None
    seed: Optional[int] = None


class TopdownTilesetRequest(SavedOutputOptions):
    lower_description: str
    upper_description: str
    transition_description: Optional[str] = None
    tile_size: dict[str, int] = Field(default_factory=lambda: {"width": 16, "height": 16})
    transition_size: float = 0.5
    view: Literal["low top-down", "high top-down"] = "low top-down"
    mode: Literal["standard", "pro"] = "standard"
    output_folder: Optional[str] = None
    seed: Optional[int] = None


class SidescrollerTilesetRequest(SavedOutputOptions):
    lower_description: str
    transition_description: str
    tile_size: dict[str, int] = Field(default_factory=lambda: {"width": 16, "height": 16})
    transition_size: float = 0.25
    output_folder: Optional[str] = None
    seed: Optional[int] = None


class IsometricTileRequest(SavedOutputOptions):
    description: str
    size: int = 32
    tile_shape: Literal["thick tile", "thin tile", "block"] = "thick tile"
    output_folder: Optional[str] = None
    seed: Optional[int] = None


class TilesProRequest(SavedOutputOptions):
    description: str
    tile_size: int = 32
    tile_type: Literal["hex", "hex_pointy", "isometric", "octagon", "square_topdown"] = "square_topdown"
    tile_view: Literal["top-down", "high top-down", "low top-down", "side"] = "top-down"
    output_folder: Optional[str] = None
    seed: Optional[int] = None


class ExportPackRequest(SavedOutputOptions):
    asset_ids: list[str] = Field(default_factory=list)
    engine: Literal["godot", "unity", "phaser", "generic", "aseprite"] = "generic"
    output_folder: Optional[str] = None


class BatchRecipeRequest(SavedOutputOptions):
    recipe: Literal["item_set", "ui_pack", "tileset_pack", "character_pack"] = "item_set"
    theme: str
    count: int = Field(default=6, ge=1, le=32)
    size: int = Field(default=64, ge=8, le=512)
    directions: int = Field(default=4, ge=1, le=8)
    animations: list[str] = Field(default_factory=lambda: ["idle", "walk"])
    output_folder: Optional[str] = None
    seed: Optional[int] = None
    palette_limit: int = Field(default=24, ge=2, le=256)


class ValidateAssetRequest(BaseModel):
    asset_path: str
    expected_width: Optional[int] = None
    expected_height: Optional[int] = None
    palette_limit: Optional[int] = None


class AssetRecord(BaseModel):
    asset_id: str
    project_id: Optional[str] = None
    character_profile_id: Optional[str] = None
    asset_type: str
    prompt: str = ""
    seed: Optional[int] = None
    width: int = 0
    height: int = 0
    production_path: Optional[str] = None
    preview_path: Optional[str] = None
    metadata_path: Optional[str] = None
    validation: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


