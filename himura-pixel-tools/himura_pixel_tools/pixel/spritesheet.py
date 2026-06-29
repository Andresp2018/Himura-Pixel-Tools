"""Sprite-sheet composition + engine atlas metadata (spec.animation_architecture).

Layouts:
  - rows_by_direction:    rows = directions, cols = frames_per_direction
  - columns_by_direction: cols = directions, rows = frames_per_direction
  - aseprite_tags:        single row, frames tagged by direction/anim
"""

from __future__ import annotations

from typing import Iterable

from PIL import Image

DIRECTIONS_4 = ["S", "W", "E", "N"]        # down, left, right, up (RPG convention)
DIRECTIONS_8 = ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]


def build_sprite_sheet(
    frames: list[Image.Image],
    frame_width: int,
    frame_height: int,
    directions: int = 4,
    frames_per_direction: int = 1,
    layout: str = "rows_by_direction",
) -> Image.Image:
    """Composite aligned frames into a single PNG sprite sheet.

    All frames must already be exact-size and pivot-aligned.
    """
    n = len(frames)
    if n == 0:
        return Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))

    if layout == "columns_by_direction":
        rows = max(1, frames_per_direction)
        cols = max(1, n // rows)
    elif layout == "aseprite_tags":
        rows = 1
        cols = n
    else:  # rows_by_direction
        cols = max(1, frames_per_direction)
        rows = max(1, n // cols)

    sheet = Image.new("RGBA", (cols * frame_width, rows * frame_height), (0, 0, 0, 0))
    for i, frame in enumerate(frames):
        r, c = i // cols, i % cols
        if r >= rows:
            break
        f = frame.convert("RGBA").resize((frame_width, frame_height), Image.NEAREST)
        sheet.alpha_composite(f, (c * frame_width, r * frame_height))
    return sheet


def split_sprite_sheet(sheet: Image.Image, frame_width: int, frame_height: int) -> list[Image.Image]:
    """Inverse of build_sprite_sheet — cut a grid sheet back into frames."""
    cols = sheet.width // frame_width
    rows = sheet.height // frame_height
    out: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            out.append(sheet.crop((c * frame_width, r * frame_height,
                                   (c + 1) * frame_width, (r + 1) * frame_height)))
    return out


# ── Engine metadata exporters ─────────────────────────────────────────────────


def aseprite_metadata(
    animation: str, directions: int, frame_width: int, frame_height: int,
    frames_per_direction: int, frame_duration_ms: int = 120,
) -> dict:
    dirs = DIRECTIONS_4 if directions == 4 else DIRECTIONS_8
    tags = []
    i = 0
    for d in dirs:
        tags.append({
            "name": f"{animation}_{d}",
            "from": i,
            "to": i + frames_per_direction - 1,
            "direction": "forward",
            "color": "#40d088",
        })
        i += frames_per_direction
    return {
        "app": "http://www.aseprite.org/",
        "version": "1.x",
        "frames": [
            {"frame": {"x": (fi % frames_per_direction) * frame_width,
                       "y": (di) * frame_height,
                       "w": frame_width, "h": frame_height},
             "duration": frame_duration_ms / 1000.0}
            for di in range(directions) for fi in range(frames_per_direction)
        ],
        "meta": {
            "size": {"w": frames_per_direction * frame_width, "h": directions * frame_height},
            "frameTags": tags,
        },
    }


def godot_metadata(name: str, frames: Iterable[dict]) -> dict:
    """Godot 3 SpriteFrames-style atlas descriptor."""
    return {
        "resource_type": "SpriteFrames",
        "name": name,
        "frames": list(frames),
    }


def unity_metadata(name: str, sheet_w: int, sheet_h: int, frame_w: int, frame_h: int,
                   rows: int, cols: int) -> dict:
    return {
        "name": name,
        "textureWidth": sheet_w,
        "textureHeight": sheet_h,
        "frames": [
            {"name": f"{name}_{r}_{c}",
             "x": c * frame_w, "y": r * frame_h,
             "width": frame_w, "height": frame_h}
            for r in range(rows) for c in range(cols)
        ],
    }


def phaser_atlas(name: str, frame_w: int, frame_h: int, rows: int, cols: int,
                 durations_ms: list[int] | None = None) -> dict:
    durations_ms = durations_ms or []
    frames = {}
    i = 0
    for r in range(rows):
        for c in range(cols):
            dur = durations_ms[i] if i < len(durations_ms) else 120
            frames[f"{name}_{i}"] = {
                "frame": {"x": c * frame_w, "y": r * frame_h, "w": frame_w, "h": frame_h},
                "duration": dur,
            }
            i += 1
    return {
        "textures": [{"imageName": f"{name}.png", "width": cols * frame_w, "height": rows * frame_h}],
        "frames": frames,
    }
