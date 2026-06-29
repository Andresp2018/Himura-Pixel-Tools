"""Procedural OpenPose skeletons for pose-conditioned generation.

ControlNet-OpenPose expects a rendered skeleton (COCO-18 colored stick figure)
as its control image. We don't have a pose *detector* for arbitrary directions,
so we synthesize the skeletons directly: a canonical standing figure, transformed
per facing (front/back/side/diagonal) and per animation frame (idle/walk/run/…).

This is what lets characters actually rotate and move while IP-Adapter keeps
their identity — the local stand-in for pixellab's trained rotation/skeleton
animation models.

Pure Pillow + NumPy. Output is an RGB image on black (the format ControlNet
OpenPose was trained on).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

# COCO-18 keypoint order used by OpenPose body pose.
KP = ["nose", "neck", "r_sho", "r_elb", "r_wri", "l_sho", "l_elb", "l_wri",
      "r_hip", "r_knee", "r_ank", "l_hip", "l_knee", "l_ank",
      "r_eye", "l_eye", "r_ear", "l_ear"]
KP_IDX = {n: i for i, n in enumerate(KP)}

# OpenPose limb connections (pairs of keypoint indices).
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
         (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
         (0, 15), (15, 17)]

# Standard OpenPose 18-color palette (per limb / per joint).
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
          (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
          (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
          (255, 0, 255), (255, 0, 170), (255, 0, 85)]


_LOCAL_POSE_ROOT = Path(__file__).resolve().parents[2] / "Openposes&Dense" / "512 by 512 crops"
_DIR_FOLDER = {
    "front": "B", "south": "B", "s": "B",
    "front-left": "BL", "south-west": "BL", "sw": "BL",
    "front-right": "BR", "south-east": "BR", "se": "BR",
    "left": "L", "west": "L", "w": "L",
    "right": "R", "east": "R", "e": "R",
    "back": "T", "north": "T", "n": "T",
    "back-left": "TL", "north-west": "TL", "nw": "TL",
    "back-right": "TR", "north-east": "TR", "ne": "TR",
}
_ANIM_FOLDERS = {
    "run": ["running bones update", "girl running bones"],
    "walk": ["walking bones update", "woman walking bones"],
    "idle": ["walking bones update", "woman walking bones"],
}


def _natural_key(path: Path) -> tuple:
    nums = [int(n) for n in re.findall(r"\d+", path.stem)]
    return tuple(nums) if nums else (0, path.name)


def _local_pose_image(animation: Optional[str], direction: str, frame: int, n_frames: int,
                      width: int, height: int) -> Optional[Image.Image]:
    if not _LOCAL_POSE_ROOT.exists():
        return None
    dkey = _DIR_FOLDER.get(direction.lower())
    if not dkey:
        return None
    folders = _ANIM_FOLDERS.get((animation or "idle").lower(), [])
    if not folders and animation:
        folders = _ANIM_FOLDERS.get("walk", [])
    for folder in folders:
        seq_dir = _LOCAL_POSE_ROOT / folder / dkey
        if not seq_dir.exists():
            continue
        files = sorted(seq_dir.glob("*.png"), key=_natural_key)
        if not files:
            continue
        if n_frames > 1:
            idx = round((frame % n_frames) * (len(files) - 1) / max(1, n_frames - 1))
        else:
            idx = len(files) // 2
        try:
            return Image.open(files[max(0, min(idx, len(files) - 1))]).convert("RGB").resize((width, height), Image.NEAREST)
        except Exception:
            continue
    return None


# ── canonical front-facing standing pose (normalized 0..1, y down) ────────────

_FRONT = {
    "nose": (0.50, 0.13), "neck": (0.50, 0.23),
    "r_sho": (0.42, 0.24), "r_elb": (0.38, 0.37), "r_wri": (0.37, 0.49),
    "l_sho": (0.58, 0.24), "l_elb": (0.62, 0.37), "l_wri": (0.63, 0.49),
    "r_hip": (0.45, 0.53), "r_knee": (0.44, 0.71), "r_ank": (0.44, 0.92),
    "l_hip": (0.55, 0.53), "l_knee": (0.56, 0.71), "l_ank": (0.56, 0.92),
    "r_eye": (0.47, 0.11), "l_eye": (0.53, 0.11),
    "r_ear": (0.44, 0.12), "l_ear": (0.56, 0.12),
}


def _mirror(pose: dict) -> dict:
    """Mirror left/right across the vertical axis (front ↔ same, swaps sides)."""
    out = {}
    swap = {"r_sho": "l_sho", "l_sho": "r_sho", "r_elb": "l_elb", "l_elb": "r_elb",
            "r_wri": "l_wri", "l_wri": "r_wri", "r_hip": "l_hip", "l_hip": "r_hip",
            "r_knee": "l_knee", "l_knee": "r_knee", "r_ank": "l_ank", "l_ank": "r_ank",
            "r_eye": "l_eye", "l_eye": "r_eye", "r_ear": "l_ear", "l_ear": "r_ear"}
    for k, v in pose.items():
        if v is None:
            out[swap.get(k, k)] = None
            continue
        out[swap.get(k, k)] = (1.0 - v[0], v[1])
    return out


def _profile_left(pose: dict) -> dict:
    """Squeeze toward the centerline and push the face forward (facing left)."""
    out = {}
    for k, v in pose.items():
        if v is None:
            out[k] = None
            continue
        x, y = v
        x = 0.5 + (x - 0.5) * 0.35          # collapse width (limbs overlap)
        out[k] = (x, y)
    # push the head/face to the left (the way it looks)
    for k, dx in (("nose", -0.10), ("r_eye", -0.10), ("l_eye", -0.10),
                  ("r_ear", 0.02), ("l_ear", 0.02)):
        if out.get(k):
            out[k] = (out[k][0] + dx, out[k][1])
    # arms/legs slightly forward
    for k in ("r_wri", "l_wri", "r_elb", "l_elb"):
        if out.get(k):
            out[k] = (out[k][0] - 0.05, out[k][1])
    return out


def direction_pose(direction: str) -> dict:
    """Return a keypoint dict for a facing direction."""
    d = direction.lower()
    if d in ("front", "s", "south"):
        return dict(_FRONT)
    if d in ("back", "n", "north"):
        p = dict(_FRONT)
        # hide the face for a back view
        for f in ("nose", "r_eye", "l_eye", "r_ear", "l_ear"):
            p[f] = None
        return p
    if d in ("left", "w", "west"):
        return _profile_left(_FRONT)
    if d in ("right", "e", "east"):
        return _mirror(_profile_left(_FRONT))
    if d in ("front-left", "sw"):
        return _blend(_FRONT, _profile_left(_FRONT), 0.5)
    if d in ("front-right", "se"):
        return _blend(_FRONT, _mirror(_profile_left(_FRONT)), 0.5)
    if d in ("back-left", "nw"):
        return _blend(direction_pose("back"), _profile_left(_FRONT), 0.5)
    if d in ("back-right", "ne"):
        return _blend(direction_pose("back"), _mirror(_profile_left(_FRONT)), 0.5)
    return dict(_FRONT)


def _blend(a: dict, b: dict, t: float) -> dict:
    out = {}
    for k in a:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            out[k] = va if vb is None else vb
        else:
            out[k] = (va[0] * (1 - t) + vb[0] * t, va[1] * (1 - t) + vb[1] * t)
    return out


# ── animation: limb swings over a cycle ───────────────────────────────────────


def _rotate(point: tuple, pivot: tuple, deg: float) -> tuple:
    r = math.radians(deg)
    dx, dy = point[0] - pivot[0], point[1] - pivot[1]
    return (pivot[0] + dx * math.cos(r) - dy * math.sin(r),
            pivot[1] + dx * math.sin(r) + dy * math.cos(r))


def animation_pose(animation: str, direction: str, frame: int, n_frames: int) -> dict:
    """A facing pose with limbs swung for the given animation frame."""
    pose = direction_pose(direction)
    if n_frames <= 1:
        return pose
    # +pi/2 so frame 0 starts at a leg-contact pose (max swing) rather than the
    # neutral passing pose — otherwise frames 0 and N/2 would be identical.
    phase = 2.0 * math.pi * frame / n_frames + math.pi / 2.0
    amp = {"walk": 22.0, "run": 38.0, "idle": 4.0, "attack": 26.0,
           "hurt": 10.0, "cast": 14.0, "death": 8.0}.get(animation, 16.0)
    swing = math.sin(phase) * amp

    def swing_limb(joint, end, foot, sign):
        if pose.get(joint) and pose.get(end):
            pose[end] = _rotate(pose[end], pose[joint], sign * swing)
            if foot and pose.get(foot):
                pose[foot] = _rotate(pose[foot], pose[joint], sign * swing * 1.2)

    # legs alternate; arms swing opposite to the same-side leg
    swing_limb("r_hip", "r_knee", "r_ank", +1)
    swing_limb("l_hip", "l_knee", "l_ank", -1)
    swing_limb("r_sho", "r_elb", "r_wri", -1)
    swing_limb("l_sho", "l_elb", "l_wri", +1)

    # vertical bob for idle/run
    bob = (math.sin(phase * 2) * 0.01) if animation in ("idle", "run") else 0.0
    if bob:
        for k in list(pose):
            if pose.get(k):
                pose[k] = (pose[k][0], pose[k][1] + bob)
    return pose


# ── rendering ─────────────────────────────────────────────────────────────────


def render_pose(pose: dict, width: int = 768, height: int = 768,
                margin: float = 0.08) -> Image.Image:
    """Render a keypoint dict as an OpenPose skeleton (RGB on black)."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    usable_h = height * (1 - 2 * margin)
    off_y = height * margin

    def px(name):
        v = pose.get(name)
        if v is None:
            return None
        return (v[0] * width, off_y + v[1] * usable_h)

    limb_w = max(2, int(round(height * 0.018)))
    joint_r = max(2, int(round(height * 0.012)))

    for i, (a, b) in enumerate(LIMBS):
        pa, pb = px(KP[a]), px(KP[b])
        if pa is None or pb is None:
            continue
        draw.line([pa, pb], fill=COLORS[i % len(COLORS)], width=limb_w)
    for i, name in enumerate(KP):
        p = px(name)
        if p is None:
            continue
        draw.ellipse([p[0] - joint_r, p[1] - joint_r, p[0] + joint_r, p[1] + joint_r],
                     fill=COLORS[i % len(COLORS)])
    return img


def direction_skeleton(direction: str, width: int = 768, height: int = 768) -> Image.Image:
    local = _local_pose_image("idle", direction, 0, 1, width, height)
    if local is not None:
        return local
    return render_pose(direction_pose(direction), width, height)


def animation_skeleton(animation: str, direction: str, frame: int, n_frames: int,
                       width: int = 768, height: int = 768) -> Image.Image:
    local = _local_pose_image(animation, direction, frame, n_frames, width, height)
    if local is not None:
        return local
    return render_pose(animation_pose(animation, direction, frame, n_frames), width, height)
