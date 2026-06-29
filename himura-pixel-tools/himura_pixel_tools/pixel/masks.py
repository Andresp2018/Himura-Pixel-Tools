"""Background/mask creation for the true-pixel pipeline.

Implements step 2 (``background_and_mask``): transparent mask via flood-fill
from corners (chroma-key style), plus optional local segmentation when a
removal model is available. Falls back gracefully to corner flood-fill only,
which needs no extra model and works fully offline.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _flood_fill_mask(alpha: np.ndarray, image_rgb: np.ndarray, tol: float = 40.0) -> np.ndarray:
    """Flood fill the background inward from every border pixel.

    Seeding from all four edges (not just the corners) catches backgrounds that
    are merely *near*-flat — the common case for diffusion output, where the
    "white"/"grey" backdrop drifts a few values across the frame. Each seed
    compares against a shared reference color sampled from the border median so
    a gradient backdrop is still treated as one region.
    """
    h, w = alpha.shape
    bg = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)

    # reference background color = median of the 1px border ring
    border = np.concatenate([
        image_rgb[0, :, :], image_rgb[-1, :, :],
        image_rgb[:, 0, :], image_rgb[:, -1, :],
    ], axis=0).astype(np.float32)
    ref = np.median(border, axis=0)

    stack: list[tuple[int, int]] = []
    for x in range(w):
        stack.append((0, x)); stack.append((h - 1, x))
    for y in range(h):
        stack.append((y, 0)); stack.append((y, w - 1))

    tol_sq = tol * tol
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w or visited[y, x]:
            continue
        visited[y, x] = True
        px = image_rgb[y, x].astype(np.float32)
        if float(((px - ref) ** 2).sum()) > tol_sq:
            continue
        bg[y, x] = True
        stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
    return bg


def remove_background_flat(image: Image.Image, tolerance: float = 40.0) -> Image.Image:
    """Transparent background via corner flood-fill (no model needed)."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[..., :3]
    alpha = arr[..., 3].copy()
    bg = _flood_fill_mask(alpha, rgb, tol=tolerance)
    alpha[bg] = 0
    arr[..., 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def chroma_key(image: Image.Image, key: tuple[int, int, int] = (255, 0, 255),
               tol: float = 120.0, despill: bool = True) -> Image.Image:
    """Remove a known chroma-key backdrop color (default magenta) exactly.

    Any pixel within ``tol`` (RGB Euclidean) of the key color becomes fully
    transparent. A light despill pulls the key color out of the kept fringe so
    sprites don't get a magenta halo. This is the reliable path when the image
    was generated on a deliberate solid backdrop.
    """
    rgba = image.convert("RGBA")
    arr = np.array(rgba).astype(np.int32)
    rgb = arr[..., :3]
    k = np.array(key, dtype=np.int32)
    dist = np.sqrt(((rgb - k) ** 2).sum(axis=2))
    bg = dist < tol
    arr[..., 3] = np.where(bg, 0, arr[..., 3])

    if despill:
        # magenta spill = high R & B, low G on kept edge pixels; clamp R,B toward G.
        keep = (~bg) & (dist < tol * 2.0)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        spill = keep & (r > g) & (b > g)
        cap = (g + np.maximum(r, b)) // 2
        arr[..., 0] = np.where(spill, np.minimum(r, cap), arr[..., 0])
        arr[..., 2] = np.where(spill, np.minimum(b, cap), arr[..., 2])

    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")



def remove_ground_artifacts(image: Image.Image, bottom_frac: float = 0.38) -> Image.Image:
    """Drop detached bottom blobs that read as ground, shadows, or pedestals.

    Diffusion models often obey "isolated character" but still add a floor pad
    under the feet. We keep the main subject and small nearby fragments, then
    remove wide/flat detached components in the lower part of the canvas.
    """
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[..., 3]
    mask = alpha >= 16
    if not mask.any():
        return rgba
    try:
        from scipy import ndimage as ndi
        labels, count = ndi.label(mask)
        if count <= 1:
            return rgba
        slices = ndi.find_objects(labels)
    except Exception:
        return rgba

    h, w = mask.shape
    sizes = np.bincount(labels.ravel())
    keep = np.zeros(count + 1, dtype=bool)
    main = int(np.argmax(sizes[1:]) + 1)
    keep[main] = True
    main_slice = slices[main - 1]
    if main_slice is None:
        return rgba
    my0, my1 = main_slice[0].start, main_slice[0].stop
    mx0, mx1 = main_slice[1].start, main_slice[1].stop
    bottom_start = int(h * (1.0 - bottom_frac))

    for label in range(1, count + 1):
        if label == main:
            continue
        sl = slices[label - 1]
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        bw, bh = x1 - x0, y1 - y0
        area = int(sizes[label])
        overlaps_subject_x = not (x1 < mx0 - w * 0.04 or x0 > mx1 + w * 0.04)
        near_subject_y = y0 <= my1 + h * 0.04
        wide_flat_bottom = y0 >= bottom_start and bw >= w * 0.30 and bh <= h * 0.18
        huge_floor = y0 >= bottom_start and bw >= (mx1 - mx0) * 1.25 and bh <= h * 0.24
        tiny_near_subject = area <= max(12, sizes[main] * 0.035) and overlaps_subject_x and near_subject_y
        if wide_flat_bottom or huge_floor:
            continue
        if tiny_near_subject:
            keep[label] = True
            continue
        if area >= sizes[main] * 0.08 and y1 < h * 0.96:
            keep[label] = True

    new_alpha = np.where(keep[labels], alpha, 0).astype(np.uint8)
    arr[..., 3] = new_alpha
    return Image.fromarray(arr, mode="RGBA")


def remove_background(image: Image.Image, use_model: bool = False) -> Image.Image:
    """Best-effort background removal.

    Tries BiRefNet segmentation when ``use_model`` is True and the runtime is
    available; otherwise uses corner flood-fill. Never raises — always returns
    an RGBA image.
    """
    if use_model:
        try:
            from .segmentation import remove_bg_model  # lazy import
            return remove_bg_model(image)
        except Exception:
            pass
    return remove_background_flat(image)


def make_mask_from_image(image: Image.Image) -> Image.Image:
    """Derive an L-mode mask (255=keep, 0=drop) from an RGBA image's alpha."""
    rgba = image.convert("RGBA")
    return rgba.split()[-1].point(lambda v: 255 if v > 16 else 0)
