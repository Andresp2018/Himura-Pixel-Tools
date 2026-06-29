"""Ordered (Bayer-matrix) dithering for palette mapping.

Research basis: the ComfyUI ``PixelArtAddDitherPattern`` node (bayer-4 / bayer-8)
and Pixel Art Lab's *adaptive* dithering, which applies texture only to smooth
areas while leaving sprite edges/outlines crisp.

This module maps an image onto a fixed palette while adding an ordered-dither
pattern, so flat regions and gentle gradients get retro banding-free shading
instead of a hard posterised step. ``edge_safe=True`` zeroes the dither near
strong luminance/alpha edges so one-pixel outlines and highlights survive.

Pure Pillow + NumPy (reuses the OKLab nearest-colour search in ``pixelate``).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from . import pixelate

# ── Bayer threshold matrices (normalised to the open interval (-0.5, 0.5)) ─────

_BAYER2 = np.array([[0, 2], [3, 1]], dtype=np.float32)
_BAYER4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float32)
_BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32)

_MATRICES = {
    "bayer2": _BAYER2,
    "bayer4": _BAYER4,
    "bayer8": _BAYER8,
}


def available_matrices() -> list[str]:
    return list(_MATRICES.keys())


def _normalised_matrix(name: str) -> np.ndarray:
    m = _MATRICES.get((name or "bayer4").lower(), _BAYER4)
    n = m.size
    # map integer threshold indices to (-0.5, 0.5)
    return (m + 0.5) / n - 0.5


def _tile(matrix: np.ndarray, h: int, w: int) -> np.ndarray:
    mh, mw = matrix.shape
    reps_y = (h + mh - 1) // mh
    reps_x = (w + mw - 1) // mw
    return np.tile(matrix, (reps_y, reps_x))[:h, :w]


def _palette_step(palette: np.ndarray) -> float:
    """Approximate the typical gap between palette colours (sets dither gain)."""
    if len(palette) < 2:
        return 0.0
    pal = palette.astype(np.float32)
    # mean nearest-neighbour distance in RGB space
    d = np.sqrt(((pal[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    nn = nn[np.isfinite(nn)]
    return float(nn.mean()) if nn.size else 32.0


def _edge_mask(rgb: np.ndarray, alpha: np.ndarray, thresh: float = 36.0) -> np.ndarray:
    """True where a strong luma/alpha edge sits — dithering is suppressed here."""
    luma = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    gx = np.zeros_like(luma); gx[:, 1:-1] = np.abs(luma[:, 2:] - luma[:, :-2])
    gy = np.zeros_like(luma); gy[1:-1, :] = np.abs(luma[2:, :] - luma[:-2, :])
    a = alpha.astype(np.float32)
    ax = np.zeros_like(a); ax[:, 1:-1] = np.abs(a[:, 2:] - a[:, :-2])
    ay = np.zeros_like(a); ay[1:-1, :] = np.abs(a[2:, :] - a[:-2, :])
    edge = gx + gy + ax + ay
    return edge > thresh


def apply_palette_dithered(image: Image.Image, palette: list[list[int]],
                           matrix: str = "bayer4", strength: float = 0.6,
                           edge_safe: bool = True) -> Image.Image:
    """Map every opaque pixel to ``palette`` with an ordered dither pattern.

    ``strength`` (0..1) scales the dither amplitude relative to the palette's
    own colour spacing. With ``edge_safe`` the dither is removed at strong
    luma/alpha edges so outlines and highlights stay clean.
    """
    if not palette:
        return image.convert("RGBA")
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3]
    opaque = alpha >= 16
    if not opaque.any():
        return rgba

    pal = np.array(palette, dtype=np.uint8)
    if len(pal) <= 1 or strength <= 0:
        return pixelate.apply_palette(rgba, palette)

    h, w = alpha.shape
    offsets = _tile(_normalised_matrix(matrix), h, w)        # (-0.5, 0.5)
    gain = _palette_step(pal) * float(max(0.0, min(strength, 1.0)))
    perturb = offsets * gain                                  # per-pixel scalar

    if edge_safe:
        perturb = np.where(_edge_mask(arr[..., :3], alpha), 0.0, perturb)

    perturbed = np.clip(rgb + perturb[..., None], 0.0, 255.0).astype(np.uint8)

    # nearest palette colour in OKLab (perceptual) for the perturbed pixels
    pal_oklab = pixelate._rgb_to_oklab(pal.reshape(1, -1, 3)).reshape(-1, 3)
    px = perturbed[opaque]
    px_oklab = pixelate._rgb_to_oklab(px.reshape(1, -1, 3)).reshape(-1, 3)
    out = np.empty_like(px)
    chunk = 200_000
    for start in range(0, px_oklab.shape[0], chunk):
        stop = min(start + chunk, px_oklab.shape[0])
        d = px_oklab[start:stop, None, :] - pal_oklab[None, :, :]
        idx = np.argmin(np.sum(d * d, axis=2), axis=1)
        out[start:stop] = pal[idx]

    arr[..., :3][opaque] = out
    return Image.fromarray(arr, mode="RGBA")
