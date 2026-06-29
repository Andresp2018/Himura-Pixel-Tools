"""Pixel cleanup, alpha snap, orphan removal (true_pixel_pipeline steps 5-7)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def snap_alpha(image: Image.Image, allow_translucent: bool = False) -> Image.Image:
    """Snap alpha to 0 or 255 (step 6).

    Production sprites must only contain fully-opaque or fully-transparent
    pixels, unless the profile explicitly allows translucent UI assets.
    """
    if allow_translucent:
        return image.convert("RGBA")
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[..., 3]
    arr[..., 3] = np.where(alpha >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def remove_orphan_pixels(image: Image.Image, min_neighbors: int = 1) -> Image.Image:
    """Remove isolated opaque pixels with fewer than N opaque 4-neighbors (step 7)."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[..., 3]
    opaque = (alpha > 0).astype(np.uint8)

    # count of opaque 4-neighbors per pixel
    up = np.zeros_like(opaque)
    up[1:, :] += opaque[:-1, :]
    up[:-1, :] += opaque[1:, :]
    up[:, 1:] += opaque[:, :-1]
    up[:, :-1] += opaque[:, 1:]
    orphan = (opaque == 1) & (up < (min_neighbors + 1))
    alpha[orphan] = 0
    arr[..., 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def enforce_outline(image: Image.Image, color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Optional: paint a single-pixel dark outline around the sprite silhouette."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[..., 3]
    opaque = alpha > 0
    # a pixel is on the boundary if any 4-neighbor is transparent
    boundary = np.zeros_like(opaque)
    boundary[1:, :] |= ~opaque[:-1, :]
    boundary[:-1, :] |= ~opaque[1:, :]
    boundary[:, 1:] |= ~opaque[:, :-1]
    boundary[:, :-1] |= ~opaque[:, 1:]
    edge = opaque & boundary
    # only paint outline where currently transparent-edge neighbor exists and px opaque
    for ci, cv in enumerate(color[:3]):
        arr[..., ci] = np.where(edge, cv, arr[..., ci])
    return Image.fromarray(arr, mode="RGBA")


def compose_on_canvas(image: Image.Image, canvas_w: int, canvas_h: int, pad: int = 2,
                      margin_frac: float = 0.06, align: str = "center") -> Image.Image:
    """Crop to the subject, scale it to fit the canvas, then center it (step 3).

    The previous implementation pasted the cropped subject at its native size,
    so any subject larger than the (downscaled) canvas overflowed and was
    clipped — the classic "head/feet cut off" bug. We now scale the crop to
    fit inside the canvas (preserving aspect ratio) with a small margin, so the
    full sprite is always visible and consistently sized. This runs on the
    hi-res (8x) canvas before pixelation, so a LANCZOS resize here is fine —
    the grid-snap downsample re-hardens the edges afterwards.

    align="bottom" feet-aligns standing characters; "center" centers both axes.
    """
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[..., 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    x0, x1 = max(0, xs.min() - pad), min(rgba.width, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(rgba.height, ys.max() + 1 + pad)
    crop = rgba.crop((x0, y0, x1, y1))

    margin_frac = min(max(margin_frac, 0.0), 0.45)
    avail_w = max(1, int(round(canvas_w * (1.0 - 2.0 * margin_frac))))
    avail_h = max(1, int(round(canvas_h * (1.0 - 2.0 * margin_frac))))
    scale = min(avail_w / crop.width, avail_h / crop.height)
    new_w = max(1, int(round(crop.width * scale)))
    new_h = max(1, int(round(crop.height * scale)))
    if (new_w, new_h) != crop.size:
        crop = crop.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    off_x = (canvas_w - new_w) // 2
    if align == "bottom":
        off_y = max(0, canvas_h - new_h - int(round(canvas_h * margin_frac)))
    else:
        off_y = (canvas_h - new_h) // 2
    canvas.alpha_composite(crop, (off_x, max(0, off_y)))
    return canvas


def compose_group_on_canvas(images: list[Image.Image], canvas_w: int, canvas_h: int,
                            pad: int = 2, margin_frac: float = 0.06,
                            align: str = "center") -> list[Image.Image]:
    """Compose a set of animation frames with ONE shared scale and pivot.

    Each frame is cropped to the *union* bounding box of all frames and placed
    at the same position, so the character stays planted on the canvas and a
    limb that swings out in one frame is never rescaled or clipped. This fixes
    the "frames jitter / parts get cut" problem that per-frame centering caused.
    """
    rgbas = [im.convert("RGBA") for im in images]
    boxes = []
    for rgba in rgbas:
        alpha = np.array(rgba)[..., 3]
        ys, xs = np.where(alpha > 0)
        if len(xs) and len(ys):
            boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    if not boxes:
        return [Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0)) for _ in rgbas]

    ux0 = max(0, min(b[0] for b in boxes) - pad)
    uy0 = max(0, min(b[1] for b in boxes) - pad)
    ux1 = min(rgbas[0].width, max(b[2] for b in boxes) + pad)
    uy1 = min(rgbas[0].height, max(b[3] for b in boxes) + pad)
    uw, uh = max(1, ux1 - ux0), max(1, uy1 - uy0)

    margin_frac = min(max(margin_frac, 0.0), 0.45)
    avail_w = max(1, int(round(canvas_w * (1.0 - 2.0 * margin_frac))))
    avail_h = max(1, int(round(canvas_h * (1.0 - 2.0 * margin_frac))))
    scale = min(avail_w / uw, avail_h / uh)
    new_w = max(1, int(round(uw * scale)))
    new_h = max(1, int(round(uh * scale)))
    off_x = (canvas_w - new_w) // 2
    if align == "bottom":
        off_y = max(0, canvas_h - new_h - int(round(canvas_h * margin_frac)))
    else:
        off_y = (canvas_h - new_h) // 2

    out: list[Image.Image] = []
    for rgba in rgbas:
        crop = rgba.crop((ux0, uy0, ux1, uy1)).resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        canvas.alpha_composite(crop, (off_x, off_y))
        out.append(canvas)
    return out


def bbox_and_pivot(image: Image.Image) -> dict:
    """Return non-transparent bbox + centered pivot for metadata."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[..., 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return {"bbox": None, "pivot": {"x": rgba.width // 2, "y": rgba.height // 2}}
    return {
        "bbox": {"x": int(xs.min()), "y": int(ys.min()),
                 "w": int(xs.max() - xs.min() + 1), "h": int(ys.max() - ys.min() + 1)},
        "pivot": {"x": rgba.width // 2, "y": rgba.height // 2},
    }
