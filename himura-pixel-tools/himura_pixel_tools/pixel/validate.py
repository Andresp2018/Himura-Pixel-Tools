"""True-pixel validation (spec.true_pixel_pipeline.production_validation)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..schemas.pixel import PixelValidation


def count_colors(image: Image.Image) -> int:
    """Count distinct opaque RGB colors (ignoring alpha=0 pixels)."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    opaque = arr[arr[..., 3] > 0]
    if opaque.shape[0] == 0:
        return 0
    rgb = opaque[..., :3]
    # pack RGB into a single int32 for fast unique
    packed = (rgb[:, 0].astype(np.int64) << 16) | (rgb[:, 1].astype(np.int64) << 8) | rgb[:, 2].astype(np.int64)
    return int(np.unique(packed).size)


def alpha_values_ok(image: Image.Image, allow_translucent: bool = False) -> bool:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[..., 3]
    if allow_translucent:
        return True
    # only 0 and 255 allowed
    unique = np.unique(alpha)
    return bool(np.all((unique == 0) | (unique == 255)))


def validate_asset(
    image: Image.Image,
    expected_width: int | None = None,
    expected_height: int | None = None,
    palette_limit: int | None = None,
    allow_translucent: bool = False,
) -> PixelValidation:
    """Run the production-validation pseudocode assertions."""
    rgba = image.convert("RGBA")
    w, h = rgba.size
    exp_w = int(expected_width) if expected_width else w
    exp_h = int(expected_height) if expected_height else h

    exact_size_ok = (w == exp_w) and (h == exp_h)
    alpha_ok = alpha_values_ok(rgba, allow_translucent=allow_translucent)
    n_colors = count_colors(rgba)
    color_ok = (palette_limit is None) or (n_colors <= palette_limit)

    errors: list[str] = []
    if not exact_size_ok:
        errors.append(f"size {w}x{h} != expected {exp_w}x{exp_h}")
    if not alpha_ok:
        errors.append("alpha contains values other than {0, 255}")
    if not color_ok:
        errors.append(f"{n_colors} colors > palette_limit {palette_limit}")

    return PixelValidation(
        ok=len(errors) == 0,
        expected_width=exp_w,
        expected_height=exp_h,
        actual_width=w,
        actual_height=h,
        exact_size_ok=exact_size_ok,
        alpha_ok=alpha_ok,
        color_count_ok=color_ok,
        color_count=n_colors,
        palette_limit=palette_limit,
        pivot_ok=True,
        export_kind="production",
        scaling_method_for_preview="nearest",
        errors=errors,
    )


def validate_file(path: str, expected_width: int | None = None, expected_height: int | None = None,
                  palette_limit: int | None = None) -> PixelValidation:
    img = Image.open(path)
    return validate_asset(img, expected_width, expected_height, palette_limit)
