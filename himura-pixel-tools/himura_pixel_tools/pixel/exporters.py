"""Asset exporters (spec.core_product_features.export_formats).

Writes:
  - production PNG (exact size)
  - 4x / 8x nearest-neighbor preview PNGs
  - sprite-sheet PNG
  - GIF / WebP previews
  - Aseprite / Unity / Godot / Phaser metadata JSON
  - ZIP export pack
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image

from ..schemas.pixel import AssetManifest
from . import spritesheet as ss


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_production_png(image: Image.Image, path: str | Path) -> str:
    """Write the exact-size production PNG (no anti-aliasing, RGBA)."""
    path = str(path)
    Image.open.__name__  # noqa: keep import warm
    rgba = image.convert("RGBA")
    rgba.save(path, format="PNG", optimize=False)
    return path


def save_preview_png(image: Image.Image, path: str | Path, scale: int = 4) -> str:
    """Nearest-neighbor upscale preview. Production scaling is nearest only."""
    path = str(path)
    rgba = image.convert("RGBA")
    w, h = rgba.size
    rgba.resize((w * scale, h * scale), Image.NEAREST).save(path, format="PNG")
    return path


def save_gif(frames: list[Image.Image], path: str | Path, duration_ms: int = 120) -> str:
    path = str(path)
    rgba_frames = [f.convert("RGBA") for f in frames]
    if not rgba_frames:
        return path
    # GIF needs palette + a checker backdrop for transparency
    composed = []
    bg = Image.new("RGBA", rgba_frames[0].size, (200, 200, 200, 255))
    for f in rgba_frames:
        c = bg.copy()
        c.alpha_composite(f)
        composed.append(c.convert("P", palette=Image.ADAPTIVE, colors=255))
    composed[0].save(path, save_all=True, append_images=composed[1:],
                     duration=duration_ms, loop=0, disposal=2)
    return path


def save_webp(frames: list[Image.Image], path: str | Path, duration_ms: int = 120) -> str:
    path = str(path)
    rgba_frames = [f.convert("RGBA") for f in frames]
    if not rgba_frames:
        return path
    if len(rgba_frames) == 1:
        rgba_frames[0].save(path, format="WEBP", lossless=True)
    else:
        rgba_frames[0].save(path, save_all=True, append_images=rgba_frames[1:],
                            format="WEBP", lossless=True, duration=duration_ms, loop=0)
    return path


def save_metadata_json(manifest: dict, path: str | Path) -> str:
    path = str(path)
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def build_manifest(
    asset_id: str, asset_type: str, prompt: str, image: Image.Image,
    seed: int | None = None, model_profile_id: str | None = None,
    license_metadata: str | None = None, palette: list | None = None,
    extra: dict | None = None,
) -> AssetManifest:
    from .cleanup import bbox_and_pivot
    bp = bbox_and_pivot(image)
    return AssetManifest(
        asset_id=asset_id,
        asset_type=asset_type,
        prompt=prompt,
        seed=seed,
        width=image.width,
        height=image.height,
        model_profile_id=model_profile_id,
        license_metadata=license_metadata,
        palette=palette or [],
        pivot=bp["pivot"],
        generated_at=_now(),
        extra=extra or {},
    )


def export_zip(files: Iterable[tuple[str, str | bytes]], zip_path: str | Path) -> str:
    """files = [(archive_name, file_path_or_bytes), ...]. Returns zip path."""
    zip_path = str(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files:
            if isinstance(content, (bytes, bytearray)):
                zf.writestr(name, content)
            else:
                zf.write(content, name)
    return zip_path


def export_engine_pack(
    frames: list[Image.Image],
    frame_width: int,
    frame_height: int,
    engine: str,
    output_folder: str,
    name: str = "asset",
    directions: int = 1,
    frames_per_direction: int | None = None,
    duration_ms: int = 120,
) -> dict:
    """Write a sprite-sheet PNG + engine metadata JSON into output_folder.

    Returns the dict of written paths.
    """
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    fpd = frames_per_direction or max(1, len(frames) // max(1, directions))
    sheet = ss.build_sprite_sheet(frames, frame_width, frame_height,
                                  directions, fpd, layout="rows_by_direction")
    sheet_path = str(out / f"{name}_sheet.png")
    sheet.save(sheet_path, format="PNG")

    meta_path = str(out / f"{name}_{engine}.json")
    rows = directions
    cols = fpd
    if engine == "aseprite":
        meta = ss.aseprite_metadata(name, directions, frame_width, frame_height, fpd, duration_ms)
    elif engine == "unity":
        meta = ss.unity_metadata(name, sheet.width, sheet.height, frame_width, frame_height, rows, cols)
    elif engine == "godot":
        meta = ss.godot_metadata(name, [
            {"x": (fi % cols) * frame_width, "y": di * frame_height,
             "w": frame_width, "h": frame_height}
            for di in range(rows) for fi in range(cols)
        ])
    elif engine == "phaser":
        meta = ss.phaser_atlas(name, frame_width, frame_height, rows, cols)
    else:  # generic
        meta = {"name": name, "frame_width": frame_width, "frame_height": frame_height,
                "rows": rows, "cols": cols, "duration_ms": duration_ms}
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"sprite_sheet_png": sheet_path, "metadata_json": meta_path}
