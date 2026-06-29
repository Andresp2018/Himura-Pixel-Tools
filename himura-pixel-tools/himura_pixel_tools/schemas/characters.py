"""Character / style / pose / reference-lock / spritesheet schemas.

These match the ``data_objects`` block of the spec's
``character_consistency_system``.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class PaletteEntry(BaseModel):
    r: int
    g: int
    b: int
    a: int = 255
    hex: Optional[str] = None


class StyleProfile(BaseModel):
    style_profile_id: str
    name: str
    prompt_prefix: str = ""
    negative_prompt: str = ""
    palette_id: Optional[str] = None
    pixel_rules: list[str] = Field(
        default_factory=lambda: [
            "limited palette", "hard edges", "transparent background",
            "single-pixel outline",
        ]
    )
    lora_adapters: list[dict] = Field(default_factory=list)   # [{model_id, weight}]
    ip_adapter_weight: float = 0.6
    controlnet_weights: dict[str, float] = Field(default_factory=dict)
    internal_generation_size: dict = Field(default_factory=lambda: {"width": 768, "height": 768})
    default_sampler: str = "dpm++"


class ReferenceLock(BaseModel):
    character_id: str
    reference_image_paths: list[str] = Field(default_factory=list)
    ip_adapter_embedding_cache_path: Optional[str] = None
    palette: list[list[int]] = Field(default_factory=list)   # [[r,g,b], ...]
    silhouette_hash: Optional[str] = None
    prompt_lock: Optional[str] = None
    seed_pack: list[int] = Field(default_factory=list)


class CharacterProfile(BaseModel):
    character_id: str
    name: str
    description: str
    base_seed: int
    style_profile_id: str
    palette_id: Optional[str] = None
    canonical_size: dict = Field(default_factory=lambda: {"width": 64, "height": 64})
    directions_supported: list[int] = Field(default_factory=lambda: [4, 8])
    reference_assets: list[str] = Field(default_factory=list)
    reference_embeddings: dict[str, Any] = Field(default_factory=dict)
    visual_locked_traits: list[str] = Field(
        default_factory=lambda: [
            "hair", "face", "body_shape", "outfit", "weapon",
            "palette", "outline_style", "silhouette",
        ]
    )
    negative_traits: list[str] = Field(
        default_factory=lambda: [
            "do not change armor color",
            "do not add extra weapon",
            "do not change species",
        ]
    )
    animation_templates: list[str] = Field(
        default_factory=lambda: [
            "idle", "walk", "run", "attack", "hurt", "death", "cast",
        ]
    )
    approved_assets: list[str] = Field(default_factory=list)
    reference_lock: Optional[ReferenceLock] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PoseTemplate(BaseModel):
    pose_id: str
    animation: str
    direction: str           # N|NE|E|SE|S|SW|W|NW
    frame_index: int
    skeleton_keypoints: Optional[Any] = None    # normalized map or control PNG path
    root_pivot: dict = Field(default_factory=lambda: {"x": 0, "y": 0})
    foot_contact: bool = True
    frame_duration_ms: int = 120


class SpriteSheetSpec(BaseModel):
    sheet_id: str
    character_id: str
    animation: str
    directions: int
    frame_width: int
    frame_height: int
    frames_per_direction: int
    layout: str = "rows_by_direction"           # rows_by_direction|columns_by_direction|aseprite_tags
    pivot: dict = Field(default_factory=lambda: {"x": 0, "y": 0})
    frame_durations_ms: list[int] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)


class ConsistencyReport(BaseModel):
    character_id: str
    asset_id: str
    palette_delta: float = 0.0
    palette_ok: bool = True
    bbox_drift_px: int = 0
    bbox_ok: bool = True
    pivot_drift_px: int = 0
    pivot_ok: bool = True
    alpha_ok: bool = True
    embedding_similarity: Optional[float] = None
    overall_ok: bool = True
    notes: list[str] = Field(default_factory=list)
