"""Pixel-pipeline validation & metadata schemas (spec.true_pixel_pipeline)."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class PixelValidation(BaseModel):
    """Result of true_pixel_pipeline.production_validation_pseudocode."""

    ok: bool
    expected_width: int
    expected_height: int
    actual_width: int
    actual_height: int
    exact_size_ok: bool
    alpha_ok: bool
    color_count_ok: bool
    color_count: int
    palette_limit: Optional[int] = None
    pivot_ok: bool = True
    export_kind: str = "production"
    scaling_method_for_preview: str = "nearest"
    errors: list[str] = Field(default_factory=list)


class AssetManifest(BaseModel):
    """Sidecar JSON written next to every production asset."""

    asset_id: str
    asset_type: str
    prompt: str
    seed: Optional[int] = None
    width: int
    height: int
    model_profile_id: Optional[str] = None
    license_metadata: Optional[str] = None
    palette: list[list[int]] = Field(default_factory=list)
    pivot: dict[str, int] = Field(default_factory=lambda: {"x": 0, "y": 0})
    generated_at: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)
