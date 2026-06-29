"""Unit tests for the true-pixel pipeline (no torch required).

Covers spec acceptance tests AT-002 (exact-size export) and AT-006
(sprite-sheet dimensions) at the pixel-processing level.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from himura_pixel_tools.pixel.pixelate import pixelate, extract_palette
from himura_pixel_tools.pixel.validate import validate_asset, count_colors, alpha_values_ok
from himura_pixel_tools.pixel.cleanup import snap_alpha, remove_orphan_pixels, compose_on_canvas
from himura_pixel_tools.pixel.spritesheet import build_sprite_sheet, aseprite_metadata
from himura_pixel_tools.pixel.pipeline import PixelPipelineOptions, run_true_pixel_pipeline
from himura_pixel_tools import config


def _rand_image(w=256, h=256):
    return Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8"), "RGB")


def test_exact_size_pixelate():
    img = _rand_image(256, 256)
    out = pixelate(img, 32, 32, colors=16, seed=1)
    assert out.size == (32, 32)
    v = validate_asset(out, 32, 32, palette_limit=16)
    assert v.exact_size_ok
    assert v.color_count_ok


def test_alpha_snap_only_0_255():
    img = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
    out = snap_alpha(img)
    alphas = set(out.convert("RGBA").getchannel("A").get_flattened_data())
    assert alphas.issubset({0, 255})


def test_orphan_removal():
    arr = np.zeros((5, 5, 4), dtype="uint8")
    arr[2, 2] = [255, 0, 0, 255]  # single isolated opaque pixel
    img = Image.fromarray(arr, "RGBA")
    out = remove_orphan_pixels(img, min_neighbors=1)
    assert out.convert("RGBA").getchannel("A").getpixel((2, 2)) == 0


def test_compose_centers_on_canvas():
    big = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    big.alpha_composite(Image.new("RGBA", (20, 20), (255, 255, 255, 255)), (40, 40))
    out = compose_on_canvas(big, 32, 32, pad=2)
    assert out.size == (32, 32)
    bbox = out.getbbox()
    assert bbox is not None


def test_sprite_sheet_dimensions_at006():
    # AT-006: 4-direction walk cycle, 6 frames each, 64x64 → 384x256 rows_by_direction
    frames = [Image.new("RGBA", (64, 64), (200, 0, 0, 255))] * (4 * 6)
    sheet = build_sprite_sheet(frames, 64, 64, directions=4, frames_per_direction=6,
                               layout="rows_by_direction")
    assert sheet.size == (384, 256)
    meta = aseprite_metadata("walk", 4, 64, 64, 6)
    assert len(meta["meta"]["frameTags"]) == 4


def test_pipeline_writes_exact_production_png():
    img = _rand_image(256, 256)
    out_dir = str(config.CACHE_ROOT / "test_pipeline")
    res = run_true_pixel_pipeline(
        img, out_dir, "test prompt", asset_type="item",
        options=PixelPipelineOptions(target_width=32, target_height=32,
                                     palette_limit=16, generate_preview_scale=[4, 8],
                                     seed=1),
        seed=1,
    )
    assert Path(res.production_path).exists()
    produced = Image.open(res.production_path)
    assert produced.size == (32, 32)              # exact production size
    assert res.validation.exact_size_ok
    assert res.validation.alpha_ok
    # previews are larger and nearest-neighbor scaled
    prev = Image.open(res.preview_paths[0])
    assert prev.size == (128, 128)
    assert Path(res.metadata_path).exists()


def test_palette_extraction():
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    pal = extract_palette(img, max_colors=4)
    assert pal and pal[0] == [10, 20, 30]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))



def test_remove_ground_artifacts_drops_detached_floor():
    from PIL import Image, ImageDraw
    from himura_pixel_tools.pixel import masks

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([28, 10, 36, 44], fill=(220, 40, 40, 255))
    draw.rectangle([8, 52, 56, 57], fill=(60, 50, 40, 255))

    cleaned = masks.remove_ground_artifacts(img)
    alpha = cleaned.split()[-1]
    assert alpha.getbbox() == (28, 10, 37, 45)
