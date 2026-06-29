"""Curated retro hardware palettes (research: NES / Game Boy / C64 / PICO-8).

Locking a sprite to one of these palettes reproduces the exact colour
constraints of classic console hardware — the same idea the Aseprite
Pixel-Plugin exposes (NES, Game Boy, C64, PICO-8). Each preset is a plain
``[[r, g, b], ...]`` list so it drops straight into
``pixelate.apply_palette`` / ``dither.apply_palette_dithered``.

Pure data + a tiny accessor, so it has no torch/Pillow dependency.
"""

from __future__ import annotations

# ── classic hardware / fantasy-console palettes ───────────────────────────────

GAMEBOY_DMG = [
    [15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15],
]

GAMEBOY_POCKET = [
    [8, 24, 32], [52, 104, 86], [136, 192, 112], [224, 248, 208],
]

PICO8 = [
    [0, 0, 0], [29, 43, 83], [126, 37, 83], [0, 135, 81],
    [171, 82, 54], [95, 87, 79], [194, 195, 199], [255, 241, 232],
    [255, 0, 77], [255, 163, 0], [255, 236, 39], [0, 228, 54],
    [41, 173, 255], [131, 118, 156], [255, 119, 168], [255, 204, 170],
]

# A representative 16-colour NES working palette (a usable subset of the PPU's
# 54 displayable colours — enough for a single sprite's locked palette).
NES16 = [
    [0, 0, 0], [124, 124, 124], [188, 188, 188], [252, 252, 252],
    [0, 0, 252], [0, 120, 248], [60, 188, 252], [104, 68, 252],
    [216, 0, 204], [228, 0, 88], [248, 56, 0], [228, 92, 16],
    [172, 124, 0], [0, 168, 68], [0, 232, 216], [248, 216, 120],
]

# Commodore 64 — the full 16-colour fixed palette (Pepto calibration).
C64 = [
    [0, 0, 0], [255, 255, 255], [136, 0, 0], [170, 255, 238],
    [204, 68, 204], [0, 204, 85], [0, 0, 170], [238, 238, 119],
    [221, 136, 85], [102, 68, 0], [255, 119, 119], [51, 51, 51],
    [119, 119, 119], [170, 255, 102], [0, 136, 255], [187, 187, 187],
]

# CGA mode 4 (palette 1, high-intensity) — the iconic 4-colour DOS look.
CGA = [
    [0, 0, 0], [85, 255, 255], [255, 85, 255], [255, 255, 255],
]

# Sweetie-16 (a popular modern 16-colour all-purpose pixel-art palette).
SWEETIE16 = [
    [26, 28, 44], [93, 39, 93], [177, 62, 83], [239, 125, 87],
    [255, 205, 117], [167, 240, 112], [56, 183, 100], [37, 113, 121],
    [41, 54, 111], [59, 93, 201], [65, 166, 246], [115, 239, 247],
    [244, 244, 244], [148, 176, 194], [86, 108, 134], [51, 60, 87],
]

PALETTE_PRESETS: dict[str, list[list[int]]] = {
    "gameboy": GAMEBOY_DMG,
    "gameboy_pocket": GAMEBOY_POCKET,
    "pico8": PICO8,
    "nes": NES16,
    "c64": C64,
    "cga": CGA,
    "sweetie16": SWEETIE16,
}

# Friendly labels for the UI dropdown.
PALETTE_LABELS = {
    "gameboy": "Game Boy (DMG, 4)",
    "gameboy_pocket": "Game Boy Pocket (4)",
    "pico8": "PICO-8 (16)",
    "nes": "NES (16)",
    "c64": "Commodore 64 (16)",
    "cga": "CGA (4)",
    "sweetie16": "Sweetie-16 (16)",
}


def get_preset(name: str | None) -> list[list[int]] | None:
    """Return the palette for a preset name, or ``None`` for unknown/empty."""
    if not name:
        return None
    return PALETTE_PRESETS.get(str(name).strip().lower())


def preset_names() -> list[str]:
    return list(PALETTE_PRESETS.keys())
