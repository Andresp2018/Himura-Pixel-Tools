"""Model-based background removal (BiRefNet) for reliable transparency.

The model is downloaded once (HuggingFace) and cached in-process, so every
generation can cut the subject out cleanly regardless of what backdrop the
diffusion model painted — this is the reliable replacement for chroma-key /
flood-fill tricks. If the model can't be loaded (offline, missing deps), the
caller (``masks.remove_background``) falls back to flood-fill.
"""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image

_MODEL = None
_LOCK = threading.Lock()
_FAILED = False
_MODEL_ID = "ZhengPeng7/BiRefNet_lite"


def _get_model():
    """Load + cache the BiRefNet segmentation model (thread-safe, lazy)."""
    global _MODEL, _FAILED
    if _MODEL is not None:
        return _MODEL
    if _FAILED:
        return None
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        if _FAILED:
            return None
        try:
            import torch
            from transformers import AutoModelForImageSegmentation
            model = AutoModelForImageSegmentation.from_pretrained(
                _MODEL_ID, trust_remote_code=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            try:
                if device == "cuda":
                    model.half()
            except Exception:
                pass
            _MODEL = model
            return _MODEL
        except Exception:
            _FAILED = True
            return None


def available() -> bool:
    return _get_model() is not None


def remove_bg_model(image: Image.Image, threshold: float = 0.5) -> Image.Image:
    """Cut the subject out using BiRefNet; raises if the model is unavailable."""
    import torch
    import torch.nn.functional as F  # noqa: N812
    from torchvision import transforms

    model = _get_model()
    if model is None:
        raise RuntimeError("segmentation model unavailable")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = next(model.parameters()).dtype
    processor = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    rgb = image.convert("RGB")
    with torch.no_grad():
        tensor = processor(rgb).unsqueeze(0).to(device=device, dtype=dtype)
        logits = model(tensor)[-1] if isinstance(model(tensor), (list, tuple)) else model(tensor)
        # BiRefNet returns a list of feature maps; the last is the final mask.
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        mask = torch.sigmoid(logits).float()
        mask = F.interpolate(mask, size=rgb.size[::-1], mode="bilinear", align_corners=False)
        mask = mask.squeeze().cpu().numpy()
    alpha = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha = np.where(alpha >= int(threshold * 255), alpha, 0)
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    return rgba
