"""ControlNet conditioning-image helpers.

Generates the spatial control images (canny edges, pose skeletons) that
``Pipelines.generate_with_controlnet`` consumes. Keeps the heavy optional
dependencies (openpose controlnet models) lazy.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def canny_edges(image: Image.Image, low: int = 80, high: int = 160) -> Image.Image:
    """Canny edge map for silhouette/item ControlNet."""
    try:
        import cv2  # type: ignore
        arr = np.array(image.convert("RGB"))
        edges = cv2.Canny(arr, low, high)
        return Image.fromarray(edges).convert("RGB")
    except Exception:
        # fallback: simple gradient magnitude
        g = image.convert("L")
        arr = np.array(g).astype(np.float32)
        gx = np.zeros_like(arr); gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
        gy = np.zeros_like(arr); gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
        mag = np.clip(np.abs(gx) + np.abs(gy), 0, 255).astype(np.uint8)
        return Image.fromarray(mag).convert("RGB")


def lineart(image: Image.Image) -> Image.Image:
    """Approximate lineart via grayscale + adaptive threshold."""
    try:
        import cv2  # type: ignore
        arr = np.array(image.convert("L"))
        return Image.fromarray(cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)).convert("RGB")
    except Exception:
        return image.convert("L").convert("RGB")


# Predefined pose skeletons for the default templates.  Each is a function that
# draws stick-figure keypoints onto a blank canvas — enough to drive the
# pose-controlnet for character turnaround/animation draft frames.

_JOINTS_64 = {
    "idle": [
        # (x, y) head, chest, hip, l.shoulder, r.shoulder, l.hand, r.hand, l.foot, r.foot
        (32, 12), (32, 26), (32, 40), (24, 26), (40, 26), (18, 40), (46, 40), (26, 56), (38, 56),
    ],
    "walk": [
        (32, 12), (32, 26), (32, 40), (24, 26), (40, 26), (16, 36), (48, 36), (22, 56), (42, 54),
    ],
    "attack": [
        (32, 12), (32, 26), (32, 40), (24, 26), (40, 26), (12, 24), (48, 38), (26, 56), (38, 56),
    ],
}


def skeleton_pose(width: int, height: int, animation: str = "idle", frame: int = 0) -> Image.Image:
    """Return a blank pose control image with a stick skeleton drawn on it."""
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        joints = _JOINTS_64.get(animation, _JOINTS_64["idle"])
        # scale joints to the requested canvas
        sx, sy = width / 64.0, height / 64.0
        pts = [(int(x * sx), int(y * sy)) for (x, y) in joints]
        bones = [(0, 1), (1, 2), (1, 3), (1, 4), (3, 5), (4, 6), (2, 7), (2, 8)]
        for a, b in bones:
            draw.line([pts[a], pts[b]], fill=(255, 255, 255), width=max(1, width // 32))
        for p in pts:
            r = max(2, width // 24)
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(255, 255, 255))
    except Exception:
        pass
    return canvas
