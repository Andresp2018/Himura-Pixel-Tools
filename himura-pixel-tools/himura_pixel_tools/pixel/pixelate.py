"""High-quality pixelation: OKLab k-means + edge-profile grid snap.

This is the core of the true-pixel pipeline's "downsample_to_true_pixels" and
"palette_quantization" steps. It produces crisp, anti-alias-free pixel art at
the *exact* requested canvas — never an interpolated upscale.

Ported and generalized from the example's ``python/pixelate.py`` grid-snap
algorithm (OKLab k-means + luma/alpha edge profile cut selection). Pure
Pillow + NumPy — no torch dependency, so it runs anywhere.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# ── color-space helpers (OKLab) ───────────────────────────────────────────────


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_oklab(rgb_linear: np.ndarray) -> np.ndarray:
    r = rgb_linear[..., 0]
    g = rgb_linear[..., 1]
    b = rgb_linear[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, a, bb], axis=-1)


def _rgb_to_oklab(rgb_u8: np.ndarray) -> np.ndarray:
    return _linear_to_oklab(_srgb_to_linear(rgb_u8.astype(np.float32) / 255.0))


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


# ── k-means palette in OKLab ──────────────────────────────────────────────────


def _kmeans_oklab(rgb_u8, alpha_u8, k, seed=42, max_iter=12, sample_cap=20000):
    h, w = alpha_u8.shape
    opaque_mask = alpha_u8 >= 16
    n_opaque = int(opaque_mask.sum())
    empty_labels = np.full((h, w), -1, dtype=np.int32)
    empty_palette = np.zeros((k, 3), dtype=np.uint8)
    if n_opaque == 0:
        return empty_labels, empty_palette

    opaque_rgb = rgb_u8[opaque_mask]
    oklab_full = _rgb_to_oklab(opaque_rgb)
    rng = np.random.default_rng(seed)

    oklab_fit = oklab_full
    if n_opaque > sample_cap:
        sample_idx = rng.choice(n_opaque, size=sample_cap, replace=False)
        oklab_fit = oklab_full[sample_idx]

    k_eff = min(k, oklab_fit.shape[0])
    centroids = np.empty((k_eff, 3), dtype=np.float32)
    first = rng.integers(0, oklab_fit.shape[0])
    centroids[0] = oklab_fit[first]
    closest_sq = np.sum((oklab_fit - centroids[0]) ** 2, axis=1)
    for i in range(1, k_eff):
        total = closest_sq.sum()
        if total <= 1e-12:
            centroids[i] = centroids[0]
            continue
        probs = closest_sq / total
        pick = rng.choice(oklab_fit.shape[0], p=probs)
        centroids[i] = oklab_fit[pick]
        closest_sq = np.minimum(closest_sq, np.sum((oklab_fit - centroids[i]) ** 2, axis=1))

    for _ in range(max_iter):
        diffs = oklab_fit[:, None, :] - centroids[None, :, :]
        assignments = np.argmin(np.sum(diffs * diffs, axis=2), axis=1)
        new_centroids = np.zeros_like(centroids)
        moved = 0.0
        for c in range(k_eff):
            members = oklab_fit[assignments == c]
            new_centroids[c] = members.mean(axis=0) if members.size else centroids[c]
            moved += float(np.sum((new_centroids[c] - centroids[c]) ** 2))
        centroids = new_centroids
        if moved < 1e-6:
            break

    # Label every opaque pixel against final centroids (chunked for memory).
    full_labels = np.empty(n_opaque, dtype=np.int32)
    chunk = 200_000
    for start in range(0, n_opaque, chunk):
        stop = min(start + chunk, n_opaque)
        d = oklab_full[start:stop, None, :] - centroids[None, :, :]
        full_labels[start:stop] = np.argmin(np.sum(d * d, axis=2), axis=1)

    labels = empty_labels.copy()
    labels[opaque_mask] = full_labels

    # Palette in linear-averaged sRGB so blended regions look natural.
    linear_all = _srgb_to_linear(opaque_rgb.astype(np.float32) / 255.0)
    palette_linear = np.zeros((k_eff, 3), dtype=np.float32)
    for c in range(k_eff):
        members = linear_all[full_labels == c]
        if members.size:
            palette_linear[c] = members.mean(axis=0)
    palette_srgb = (_linear_to_srgb(palette_linear) * 255.0 + 0.5).astype(np.uint8)
    if k_eff < k:
        palette_srgb = np.concatenate([palette_srgb, np.tile(palette_srgb[0:1], (k - k_eff, 1))], axis=0)
    return labels, palette_srgb


# ── edge profile → grid cuts ──────────────────────────────────────────────────


def _edge_profiles(rgb_u8, alpha_u8):
    luma = (0.299 * rgb_u8[..., 0] + 0.587 * rgb_u8[..., 1] + 0.114 * rgb_u8[..., 2]).astype(np.float32)
    alpha = alpha_u8.astype(np.float32)
    luma = np.where(alpha >= 16, luma, 0.0)

    def _grad_x(img):
        g = np.zeros_like(img)
        g[:, 1:-1] = img[:, 2:] - img[:, :-2]
        return np.abs(g)

    def _grad_y(img):
        g = np.zeros_like(img)
        g[1:-1, :] = img[2:, :] - img[:-2, :]
        return np.abs(g)

    ALPHA_W = 2.0
    gx = _grad_x(luma) + ALPHA_W * _grad_x(alpha)
    gy = _grad_y(luma) + ALPHA_W * _grad_y(alpha)
    return gx.sum(axis=0), gy.sum(axis=1)


def _find_cuts(profile, target_count, samples=64):
    source = len(profile)
    if target_count <= 0 or source <= 0:
        return np.array([0, source], dtype=np.int32)
    step = source / target_count
    if step <= 1.0:
        return np.linspace(0, source, target_count + 1).round().astype(np.int32)

    def _score(phi):
        xs = (phi + np.arange(1, target_count) * step).astype(np.int32)
        xs = np.clip(xs, 1, source - 2)
        return profile[xs - 1].sum() + profile[xs].sum() + profile[xs + 1].sum()

    phis = np.linspace(0.0, step, samples, endpoint=False)
    scores = np.array([_score(p) for p in phis])
    best_phi = phis[int(np.argmax(scores))]
    best_score = scores.max()
    if best_score < _score(step / 2.0) * 1.05:
        return np.linspace(0, source, target_count + 1).round().astype(np.int32)

    cuts = np.zeros(target_count + 1, dtype=np.int32)
    cuts[0] = 0
    cuts[-1] = source
    cuts[1:-1] = np.clip((best_phi + np.arange(1, target_count) * step).round().astype(np.int32), 1, source - 1)
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + 1
    cuts[-1] = source
    return cuts


def _resample(labels, rgb_u8, alpha_u8, x_cuts, y_cuts, palette, k):
    target_h = len(y_cuts) - 1
    target_w = len(x_cuts) - 1

    luma = (0.299 * rgb_u8[..., 0] + 0.587 * rgb_u8[..., 1] + 0.114 * rgb_u8[..., 2]).astype(np.float32)
    gx = np.zeros_like(luma); gx[:, 1:-1] = np.abs(luma[:, 2:] - luma[:, :-2])
    gy = np.zeros_like(luma); gy[1:-1, :] = np.abs(luma[2:, :] - luma[:-2, :])
    edge = gx + gy
    max_edge = float(edge.max()) if edge.size else 1.0
    if max_edge < 1e-6:
        max_edge = 1.0
    weight = 1.0 + 2.0 * (edge / max_edge)
    weight[alpha_u8 < 16] = 0.0

    out_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    out_alpha = np.zeros((target_h, target_w), dtype=np.uint8)

    for j in range(target_h):
        y0, y1 = y_cuts[j], y_cuts[j + 1]
        if y1 <= y0:
            continue
        for i in range(target_w):
            x0, x1 = x_cuts[i], x_cuts[i + 1]
            if x1 <= x0:
                continue
            cell_alpha = alpha_u8[y0:y1, x0:x1]
            if (cell_alpha < 16).mean() > 0.5:
                continue
            cell_labels = labels[y0:y1, x0:x1].ravel()
            cell_weights = weight[y0:y1, x0:x1].ravel()
            valid = cell_labels >= 0
            if not valid.any():
                continue
            totals = np.bincount(cell_labels[valid], weights=cell_weights[valid], minlength=k)
            if totals.sum() <= 0:
                continue
            out_rgb[j, i] = palette[int(np.argmax(totals))]
            out_alpha[j, i] = 255
    return out_rgb, out_alpha


# ── public API ────────────────────────────────────────────────────────────────


def pixelate(image: Image.Image, target_width: int, target_height: int, colors: int = 16, seed: int = 42) -> Image.Image:
    """Downsample an image to exact true-pixel dimensions using grid-snap.

    Uses OKLab k-means palette + edge-aware cut selection so the output is
    crisp pixel art — never bilinear anti-aliasing. Transparency preserved.
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_width and target_height must be positive")

    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    rgb_u8 = arr[..., :3]
    alpha_u8 = arr[..., 3]

    if colors <= 1:
        return rgba.resize((target_width, target_height), Image.NEAREST)

    labels, palette = _kmeans_oklab(rgb_u8, alpha_u8, k=int(colors), seed=seed)
    profile_x, profile_y = _edge_profiles(rgb_u8, alpha_u8)
    x_cuts = _find_cuts(profile_x, target_width)
    y_cuts = _find_cuts(profile_y, target_height)
    out_rgb, out_alpha = _resample(labels, rgb_u8, alpha_u8, x_cuts, y_cuts, palette, k=int(colors))

    out = np.concatenate([out_rgb, out_alpha[..., None]], axis=-1)
    return Image.fromarray(out, mode="RGBA")


def apply_palette(image: Image.Image, palette: list[list[int]]) -> Image.Image:
    """Remap every opaque pixel to its nearest color in ``palette`` (OKLab).

    Used to lock a character's directions/animation frames to the canonical
    reference palette so identity colors never drift between sprites.
    """
    if not palette:
        return image.convert("RGBA")
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    opaque = alpha >= 16
    if not opaque.any():
        return rgba

    pal = np.array(palette, dtype=np.uint8)
    pal_oklab = _rgb_to_oklab(pal.reshape(1, -1, 3)).reshape(-1, 3)
    px = rgb[opaque]
    px_oklab = _rgb_to_oklab(px.reshape(1, -1, 3)).reshape(-1, 3)
    # nearest palette entry per pixel (chunked to bound memory)
    out = np.empty_like(px)
    chunk = 200_000
    for start in range(0, px_oklab.shape[0], chunk):
        stop = min(start + chunk, px_oklab.shape[0])
        d = px_oklab[start:stop, None, :] - pal_oklab[None, :, :]
        idx = np.argmin(np.sum(d * d, axis=2), axis=1)
        out[start:stop] = pal[idx]
    rgb[opaque] = out
    arr[..., :3] = rgb
    return Image.fromarray(arr, mode="RGBA")


def extract_palette(image: Image.Image, max_colors: int = 16,
                    protect_extremes: bool = False) -> list[list[int]]:
    """Return the dominant opaque palette as [[r,g,b], ...].

    Only palette entries that are actually used by at least one pixel are
    returned, so a single-color image yields a one-entry palette.

    ``protect_extremes`` (research: Pixel Art Lab's projected-rare strategy)
    reserves palette slots for the darkest and brightest opaque colours so
    outline strokes and highlight glints survive quantization instead of being
    averaged away. The total stays within ``max_colors``.
    """
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[..., 3]
    rgb = arr[..., :3]
    opaque = rgb[alpha >= 16]
    if opaque.size == 0:
        return []

    # rare-colour protection: capture the exact darkest/brightest opaque colours
    extremes: list[list[int]] = []
    if protect_extremes and opaque.shape[0] >= 2:
        luma = 0.299 * opaque[:, 0] + 0.587 * opaque[:, 1] + 0.114 * opaque[:, 2]
        for idx in (int(np.argmin(luma)), int(np.argmax(luma))):
            c = opaque[idx].tolist()
            if c not in extremes:
                extremes.append(c)

    k = max(1, min(max_colors - len(extremes), 32))
    if k < 1:
        k = 1
    # _kmeans_oklab expects (H,W,3) rgb + (H,W) alpha — wrap the flat opaque
    # pixels into a single-row image so the palette is computed correctly.
    h = opaque.shape[0]
    rgb2d = opaque.reshape(1, h, 3)
    alpha2d = np.full((1, h), 255, dtype=np.uint8)
    labels, palette = _kmeans_oklab(rgb2d, alpha2d, k=k)
    used = np.unique(labels[labels >= 0])
    result = [palette[i].tolist() for i in used if 0 <= i < len(palette)]

    if extremes:
        # append each extreme only if it isn't already well-represented (OKLab)
        existing = np.array(result, dtype=np.uint8) if result else np.zeros((0, 3), np.uint8)
        ex_oklab = _rgb_to_oklab(existing.reshape(1, -1, 3)).reshape(-1, 3) if len(existing) else None
        for c in extremes:
            if len(result) >= max_colors:
                break
            if ex_oklab is None or len(ex_oklab) == 0:
                result.append(c)
                ex_oklab = _rgb_to_oklab(np.array([c], np.uint8).reshape(1, -1, 3)).reshape(-1, 3)
                continue
            c_ok = _rgb_to_oklab(np.array([c], np.uint8).reshape(1, -1, 3)).reshape(-1, 3)
            d = np.sum((ex_oklab - c_ok) ** 2, axis=1).min()
            if d > 0.0009:   # ~perceptibly different from every existing entry
                result.append(c)
                ex_oklab = np.vstack([ex_oklab, c_ok])
    return result
