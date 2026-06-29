"""Optional external image provider: Google Gemini.

This is the *only* non-local generation path in Himura Pixel Tools, and it is
strictly opt-in: it is used **only** when the user selects the ``gemini``
provider AND has pasted a Gemini API key in Settings. When active it replaces
the on-GPU SDXL/FLUX source generation; everything downstream (segmentation,
true-pixel snap, palette/dither, export) still runs locally, so the output is
the same exact-size pixel art.

Implemented with the Python standard library only (``urllib``) — no extra
dependency, no SDK, no telemetry. The call is synchronous and runs inside the
job worker thread.

Network note: the user's prompt is sent to Google's Generative Language API.
That only happens when the key is set and the provider is ``gemini``; the
default install never makes an outbound generation call.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Optional

from PIL import Image
import io

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiError(RuntimeError):
    """Raised when the Gemini provider cannot return an image."""


def _extract_inline_image(payload: dict) -> Optional[bytes]:
    """Pull the first inline image out of a generateContent response."""
    for cand in payload.get("candidates", []) or []:
        content = cand.get("content") or {}
        for part in content.get("parts", []) or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                try:
                    return base64.b64decode(inline["data"])
                except Exception:
                    continue
    return None


def _compose_prompt(prompt: str, negative: str, width: int, height: int) -> str:
    """Gemini has no separate negative field, so fold the constraints into text."""
    parts = [prompt.strip()]
    parts.append(
        f"Render as crisp pixel art suitable for a {width}x{height} game sprite: "
        "blocky pixels, hard edges, limited palette, no anti-aliasing, "
        "centered single subject on a plain flat background."
    )
    if negative:
        parts.append("Avoid: " + negative)
    return "\n".join(p for p in parts if p)


def generate_image(prompt: str, *, api_key: str, model: str = "gemini-2.5-flash-image",
                   negative: str = "", width: int = 1024, height: int = 1024,
                   timeout: float = 120.0) -> Image.Image:
    """Generate a single source image with Gemini and return it as a PIL image.

    Raises ``GeminiError`` on any auth/network/parse failure so the caller can
    fall back to local generation.
    """
    key = (api_key or "").strip()
    if not key:
        raise GeminiError("no Gemini API key configured")

    url = f"{_API_ROOT}/{model}:generateContent?key={key}"
    body = {
        "contents": [{"parts": [{"text": _compose_prompt(prompt, negative, width, height)}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        raise GeminiError(f"Gemini HTTP {e.code}: {detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise GeminiError(f"Gemini network error: {e.reason}") from e
    except Exception as e:  # pragma: no cover - defensive
        raise GeminiError(f"Gemini request failed: {e}") from e

    img_bytes = _extract_inline_image(payload)
    if not img_bytes:
        # surface any text the model returned (often an explanation/refusal)
        note = ""
        try:
            for cand in payload.get("candidates", []):
                for part in (cand.get("content") or {}).get("parts", []):
                    if part.get("text"):
                        note = part["text"][:300]
                        break
        except Exception:
            pass
        raise GeminiError("Gemini returned no image" + (f": {note}" if note else ""))

    try:
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise GeminiError(f"could not decode Gemini image: {e}") from e
