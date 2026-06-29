"""Tileset post-processing: seamless healing, isometric diamond masks, and
16-tile Wang (corner-blob) autotile composition.

These give each tileset *type* genuinely different output (the previous code
produced the same kind of tile regardless of type) and add real autotiling for
the Wang set — the corner-based system pixellab uses for top-down terrain.

Pure Pillow + NumPy, so the geometry is unit-testable without a GPU.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def make_seamless(img: Image.Image, band_frac: float = 0.25) -> Image.Image:
    """Heal opposite edges so the tile repeats without a visible seam.

    The top band is cross-blended with the bottom band (and left with right),
    so row 0 ≈ row H-1 and column 0 ≈ column W-1 — i.e. the texture tiles.
    """
    a = np.array(img.convert("RGB")).astype(np.float32)
    h, w, _ = a.shape
    out = a.copy()
    bh = max(1, int(h * band_frac))
    for i in range(bh):
        wgt = 0.5 * (1.0 - i / bh)            # 0.5 at the very edge → 0 inward
        out[i] = a[i] * (1 - wgt) + a[h - 1 - i] * wgt
        out[h - 1 - i] = a[h - 1 - i] * (1 - wgt) + a[i] * wgt
    a2 = out.copy()
    bw = max(1, int(w * band_frac))
    for j in range(bw):
        wgt = 0.5 * (1.0 - j / bw)
        out[:, j] = a2[:, j] * (1 - wgt) + a2[:, w - 1 - j] * wgt
        out[:, w - 1 - j] = a2[:, w - 1 - j] * (1 - wgt) + a2[:, j] * wgt
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def diamond_mask(w: int, h: int, feather: int = 1) -> Image.Image:
    """Rhombus (isometric) alpha mask: opaque inside the diamond, clear outside."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    d = np.abs(xx - cx) / (cx + 1e-6) + np.abs(yy - cy) / (cy + 1e-6)
    alpha = np.clip((1.0 - d) * (h / max(1, feather)), 0.0, 1.0)
    return Image.fromarray((alpha * 255).astype(np.uint8), "L")


def apply_diamond(img: Image.Image) -> Image.Image:
    """Cut an isometric diamond out of a square tile (transparent corners)."""
    rgba = img.convert("RGBA")
    rgba.putalpha(diamond_mask(rgba.width, rgba.height))
    return rgba


def _corner_weights(w: int, h: int) -> dict[str, np.ndarray]:
    """Bilinear corner influence maps (sum to 1 at every pixel)."""
    x = np.linspace(0.0, 1.0, w)[None, :]
    y = np.linspace(0.0, 1.0, h)[:, None]
    return {
        "tl": (1 - x) * (1 - y), "tr": x * (1 - y),
        "bl": (1 - x) * y, "br": x * y,
    }


# Wang corner bits: TL=1, TR=2, BL=4, BR=8  → 16 tiles
WANG_CORNERS = ["tl", "tr", "bl", "br"]


def wang_16(lower: Image.Image, upper: Image.Image, tile_w: int, tile_h: int,
            ) -> tuple[list[Image.Image], Image.Image]:
    """Build a 16-tile corner-blob Wang set from two textures.

    Each tile encodes which of its 4 corners are the *upper* material; the rest
    are *lower*, blended with smooth bilinear corner masks. Returns the 16 tiles
    plus a 4×4 sheet — drop-in for corner-based autotiling.
    """
    lo = np.array(lower.convert("RGB").resize((tile_w, tile_h), Image.LANCZOS)).astype(np.float32)
    up = np.array(upper.convert("RGB").resize((tile_w, tile_h), Image.LANCZOS)).astype(np.float32)
    cw = _corner_weights(tile_w, tile_h)

    tiles: list[Image.Image] = []
    for bits in range(16):
        mask = np.zeros((tile_h, tile_w), dtype=np.float32)
        for i, corner in enumerate(WANG_CORNERS):
            if bits & (1 << i):
                mask += cw[corner]
        mask = np.clip(mask, 0.0, 1.0)[..., None]
        tile = lo * (1 - mask) + up * mask
        tiles.append(Image.fromarray(np.clip(tile, 0, 255).astype(np.uint8), "RGB"))

    sheet = Image.new("RGB", (4 * tile_w, 4 * tile_h), (0, 0, 0))
    for i, t in enumerate(tiles):
        r, c = i // 4, i % 4
        sheet.paste(t, (c * tile_w, r * tile_h))
    return tiles, sheet


# Per-type tileset pipeline profiles: how each tileset_type is generated.
#   view        : added to the prompt
#   transparent : tile has a transparent background (platforms/iso)
#   seamless    : heal edges for repeating ground textures
#   shape       : "diamond" cuts an isometric rhombus
#   wang        : build a 16-tile autotile set from two textures
TILESET_PROFILES = {
    "top_down":     {"view": "strict top-down orthographic overhead view, camera looking straight down, square seamless ground texture, no visible side faces",
                     "negative": "isometric, diamond tile, side view, perspective, horizon, visible walls, side faces",
                     "transparent": False, "seamless": True, "shape": None, "wang": False},
    "sidescroller": {"view": "strict side-view 2D platformer terrain block, horizontal top edge, visible front side face, no overhead view",
                     "negative": "top-down, overhead map, isometric, diamond tile, camera looking straight down",
                     "transparent": True, "seamless": False, "shape": None, "wang": False},
    "isometric":    {"view": "true isometric 2:1 diamond floor tile, axonometric camera, visible top face and short side faces, transparent corners",
                     "negative": "top-down flat texture, square seamless texture, side-scroller platform, front elevation, horizon",
                     "transparent": True, "seamless": False, "shape": "diamond", "wang": False},
    "wang":         {"view": "strict top-down orthographic seamless terrain texture for 16-tile corner autotiling, camera looking straight down",
                     "negative": "isometric, diamond tile, side view, perspective, horizon, visible walls, side faces",
                     "transparent": False, "seamless": True, "shape": None, "wang": True},
}


def tileset_profile(tileset_type: str) -> dict:
    return TILESET_PROFILES.get(tileset_type, TILESET_PROFILES["top_down"])


def seam_score(tile: Image.Image) -> float:
    """Mean per-channel mismatch (0..255) between a tile's opposite edges.

    For a seamless ground tile the left↔right and top↔bottom edge pixels should
    be nearly identical; a high score means a visible grid line in-engine.
    Audit §3 "Seam Validation". Lower is better; ~0 is a perfect repeat.
    """
    a = np.asarray(tile.convert("RGB")).astype(np.float32)
    h, w, _ = a.shape
    if h < 2 or w < 2:
        return 0.0
    vert = np.abs(a[:, 0, :] - a[:, -1, :]).mean()     # left vs right column
    horiz = np.abs(a[0, :, :] - a[-1, :, :]).mean()    # top vs bottom row
    return float((vert + horiz) / 2.0)


def is_seamless(tile: Image.Image, tolerance: float = 12.0) -> bool:
    """True if opposite edges match within ``tolerance`` (0..255 per channel)."""
    return seam_score(tile) <= tolerance
