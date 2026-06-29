"""True-pixel snapper — a Python reimplementation of the algorithm behind
Hugo-Dz/spritefusion-pixel-snapper.

AI image models can't think in a pixel grid: the "pixels" they draw drift in
size and phase, so a generated sprite is really a fuzzy high-res image. This
module recovers the real grid and snaps the image to genuine, uniform pixels:

  1. compute_profiles   — Sobel gradient projected onto the X and Y axes.
  2. estimate_step_size — peak detection on each profile → median peak spacing
                          = the native cell size (auto-detected pixel size).
  3. walk               — an elastic walker lays grid cuts, snapping each cut to
                          the nearest strong edge so the grid follows the art.
  4. resample           — each cell collapses to its most frequent (mode) color.
  5. quantize           — optional k-means color reduction.

Pure NumPy + Pillow, so it runs anywhere with no torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class SnapResult:
    image: Image.Image          # the snapped low-res true-pixel image
    pixel_size_x: float         # detected cell width (source px per output px)
    pixel_size_y: float
    out_width: int
    out_height: int
    colors: int


# ── gradient profiles ─────────────────────────────────────────────────────────


def _gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def compute_profiles(rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel-like [-1,0,1] gradient magnitude projected onto each axis."""
    g = _gray(rgb)
    g = np.where(alpha >= 16, g, 0.0)
    gx = np.zeros_like(g)
    gx[:, 1:-1] = np.abs(g[:, 2:] - g[:, :-2])
    gy = np.zeros_like(g)
    gy[1:-1, :] = np.abs(g[2:, :] - g[:-2, :])
    # add an alpha edge term so silhouettes count even on flat color
    a = alpha.astype(np.float32)
    ax = np.zeros_like(a); ax[:, 1:-1] = np.abs(a[:, 2:] - a[:, :-2])
    ay = np.zeros_like(a); ay[1:-1, :] = np.abs(a[2:, :] - a[:-2, :])
    profile_x = (gx + 2.0 * ax).sum(axis=0)   # length W
    profile_y = (gy + 2.0 * ay).sum(axis=1)   # length H
    return profile_x, profile_y


def estimate_step_size(profile: np.ndarray, peak_threshold_multiplier: float = 0.2,
                       peak_distance_filter: int = 4) -> Optional[float]:
    """Median spacing between strong local maxima of the gradient profile."""
    n = len(profile)
    if n < 3:
        return None
    max_val = float(profile.max())
    if max_val <= 1e-6:
        return None
    threshold = max_val * peak_threshold_multiplier
    peaks = []
    for i in range(1, n - 1):
        v = profile[i]
        if v > threshold and v >= profile[i - 1] and v >= profile[i + 1]:
            if peaks and (i - peaks[-1]) < peak_distance_filter:
                # keep the stronger of the two close peaks
                if v > profile[peaks[-1]]:
                    peaks[-1] = i
                continue
            peaks.append(i)
    if len(peaks) < 2:
        return None
    diffs = np.diff(peaks)
    diffs = diffs[diffs >= 1]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def resolve_step_sizes(sx: Optional[float], sy: Optional[float], w: int, h: int,
                       fallback_target_segments: int = 64, max_step_ratio: float = 1.8,
                       ) -> tuple[float, float]:
    """Fill missing estimates and keep the two axes coherent."""
    if sx is None and sy is None:
        s = max(1.0, min(w, h) / fallback_target_segments)
        return s, s
    if sx is None:
        sx = sy
    if sy is None:
        sy = sx
    lo, hi = min(sx, sy), max(sx, sy)
    if lo > 0 and hi / lo > max_step_ratio:
        sx = sy = lo
    return max(1.0, float(sx)), max(1.0, float(sy))


def walk(profile: np.ndarray, step: float, search_ratio: float = 0.35,
         min_window: float = 2.0, strength_threshold: float = 0.5) -> np.ndarray:
    """Elastic walker: place grid cuts ~step apart, snapping to nearby edges."""
    n = len(profile)
    if step <= 0:
        return np.array([0, n], dtype=np.int64)
    mean_val = float(profile.mean()) if n else 0.0
    cuts = [0]
    pos = 0.0
    window = max(step * search_ratio, min_window)
    guard = 0
    while pos < n - 1 and guard < 100000:
        guard += 1
        target = pos + step
        lo = int(max(cuts[-1] + 1, np.floor(target - window)))
        hi = int(min(n - 1, np.ceil(target + window)))
        if lo >= hi:
            nxt = int(round(target))
        else:
            seg = profile[lo:hi]
            local = int(np.argmax(seg)) + lo
            nxt = local if profile[local] > mean_val * strength_threshold else int(round(target))
        nxt = max(cuts[-1] + 1, min(n - 1, nxt))
        cuts.append(nxt)
        pos = float(nxt)
        if nxt >= n - 1:
            break
    # close the grid; if the final cell is a thin sliver (< half a step), merge
    # it into the previous cell instead of emitting an extra 1px row/column.
    if cuts[-1] != n:
        if (n - cuts[-1]) < step * 0.5 and len(cuts) >= 2:
            cuts[-1] = n
        else:
            cuts.append(n)
    return np.array(sorted(set(cuts)), dtype=np.int64)


def _resample_mode(arr: np.ndarray, x_cuts: np.ndarray, y_cuts: np.ndarray) -> np.ndarray:
    """Collapse each grid cell to its most frequent opaque RGBA value (mode)."""
    th = len(y_cuts) - 1
    tw = len(x_cuts) - 1
    out = np.zeros((th, tw, 4), dtype=np.uint8)
    for j in range(th):
        y0, y1 = y_cuts[j], y_cuts[j + 1]
        if y1 <= y0:
            continue
        for i in range(tw):
            x0, x1 = x_cuts[i], x_cuts[i + 1]
            if x1 <= x0:
                continue
            cell = arr[y0:y1, x0:x1].reshape(-1, 4)
            opaque = cell[cell[:, 3] >= 16]
            if opaque.shape[0] == 0:
                continue  # leave transparent
            # mode over packed RGBA
            packed = (opaque[:, 0].astype(np.uint64) << 24 | opaque[:, 1].astype(np.uint64) << 16
                      | opaque[:, 2].astype(np.uint64) << 8 | opaque[:, 3].astype(np.uint64))
            vals, counts = np.unique(packed, return_counts=True)
            out[j, i] = opaque[np.where(packed == vals[int(np.argmax(counts))])[0][0]]
    return out


def snap_uniform(image: Image.Image, cell_w: int, cell_h: int,
                 k_colors: int = 0, protect_extremes: bool = False) -> Image.Image:
    """Mode-snap an image on a fixed uniform grid (used by the main pipeline,
    where the target cell size is known: each output pixel = one cell).

    ``protect_extremes`` keeps the darkest/brightest colours (outline + glint)
    when the palette is reduced — see ``pixelate.extract_palette``.
    """
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    h, w = arr.shape[:2]
    cell_w = max(1, int(cell_w))
    cell_h = max(1, int(cell_h))
    x_cuts = np.append(np.arange(0, w, cell_w), w).astype(np.int64)
    y_cuts = np.append(np.arange(0, h, cell_h), h).astype(np.int64)
    out = Image.fromarray(_resample_mode(arr, x_cuts, y_cuts), "RGBA")
    if k_colors and k_colors > 1:
        from . import pixelate
        pal = pixelate.extract_palette(out, max_colors=k_colors,
                                       protect_extremes=protect_extremes)
        out = pixelate.apply_palette(out, pal)
    return out


def snap_to_true_pixels(image: Image.Image, pixel_size: Optional[float] = None,
                        k_colors: int = 0, max_output: int = 512) -> SnapResult:
    """Snap an image to true pixels. ``pixel_size`` overrides auto-detection;
    ``k_colors`` > 1 also k-means-quantizes the palette."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    h, w = arr.shape[:2]
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    if pixel_size and pixel_size > 0:
        sx = sy = float(pixel_size)
    else:
        px, py = compute_profiles(rgb, alpha)
        sx, sy = resolve_step_sizes(estimate_step_size(px), estimate_step_size(py), w, h)

    # guard against absurd output dimensions
    if w / sx > max_output:
        sx = w / max_output
    if h / sy > max_output:
        sy = h / max_output

    px_profile, py_profile = compute_profiles(rgb, alpha)
    x_cuts = walk(px_profile, sx)
    y_cuts = walk(py_profile, sy)
    out_arr = _resample_mode(arr, x_cuts, y_cuts)
    out = Image.fromarray(out_arr, "RGBA")

    n_colors = len(np.unique(out_arr[out_arr[..., 3] >= 16][:, :3].reshape(-1, 3), axis=0)) \
        if (out_arr[..., 3] >= 16).any() else 0
    if k_colors and k_colors > 1:
        from . import pixelate
        pal = pixelate.extract_palette(out, max_colors=k_colors)
        out = pixelate.apply_palette(out, pal)
        n_colors = len(pal)

    return SnapResult(image=out, pixel_size_x=sx, pixel_size_y=sy,
                      out_width=out.width, out_height=out.height, colors=n_colors)
