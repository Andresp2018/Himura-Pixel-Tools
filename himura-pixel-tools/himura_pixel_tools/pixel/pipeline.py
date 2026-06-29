"""True-pixel pipeline orchestrator (spec.true_pixel_pipeline).

Runs the 9-step pipeline on a raw model output and writes a production-ready
exact-size asset + previews + sidecar metadata. This is the single entry point
both the API and the MCP bridge call for any generation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from ..schemas.pixel import AssetManifest, PixelValidation
from . import cleanup, exporters, masks, pixelate, snap, validate


@dataclass
class PixelPipelineOptions:
    target_width: int = 64
    target_height: int = 64
    transparent: bool = True
    palette_limit: Optional[int] = 24
    pad: int = 2
    allow_translucent: bool = False
    generate_preview_scale: list[int] = field(default_factory=lambda: [4, 8])
    cleanup_orphans: bool = True
    use_segmentation: bool = True       # model-based bg removal (BiRefNet) when transparent
    snap_mode: bool = True              # true-pixel mode-snap downsample (spritefusion-style)
    seed: int = 42
    # game-ready / consistency knobs
    lock_palette: Optional[list] = None   # remap output to this exact palette
    add_outline: bool = False             # 1px dark silhouette outline
    outline_color: tuple = (0, 0, 0)
    fit_margin: float = 0.06              # subject margin inside the canvas
    align: str = "center"                 # "center" | "bottom" (feet-aligned)
    chroma_key: Optional[tuple] = None    # key out this bg color instead of flood-fill
    skip_background: bool = False         # source already has transparency
    skip_compose: bool = False            # source is already canvas-sized & aligned
    remove_floor_artifacts: bool = False  # strip detached lower ground/shadow blobs
    # research-driven palette knobs
    palette_preset: Optional[str] = None  # lock to a retro preset (gameboy/nes/...)
    dither: str = "none"                  # none | bayer2 | bayer4 | bayer8
    dither_strength: float = 0.6          # 0..1 dither amplitude
    protect_extremes: bool = False        # keep outline/glint colours on reduce


@dataclass
class PixelPipelineResult:
    production_path: str
    preview_paths: list[str]
    metadata_path: str
    manifest: AssetManifest
    validation: PixelValidation
    image: Image.Image  # the exact-size production image (in memory)


def run_true_pixel_pipeline(
    source: Image.Image,
    output_folder: str,
    prompt: str,
    asset_id: Optional[str] = None,
    asset_type: str = "item",
    options: Optional[PixelPipelineOptions] = None,
    seed: Optional[int] = None,
    model_profile_id: Optional[str] = None,
    license_metadata: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> PixelPipelineResult:
    """Execute all 9 steps of the true-pixel pipeline."""
    opts = options or PixelPipelineOptions()
    asset_id = asset_id or uuid.uuid4().hex[:12]
    seed = seed if seed is not None else opts.seed
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    def step(n: int, label: str):
        if on_progress:
            on_progress(n, label)

    # Step 1: generate_source (already done — source provided)
    step(1, "generate_source")

    # Step 2: background_and_mask
    step(2, "background_and_mask")
    if opts.skip_background:
        img = source.convert("RGBA")
    elif opts.transparent and opts.chroma_key:
        img = masks.chroma_key(source, key=tuple(opts.chroma_key))
    elif opts.transparent:
        img = masks.remove_background(source, use_model=opts.use_segmentation)
    else:
        img = source.convert("RGBA")
    if opts.remove_floor_artifacts and opts.transparent:
        img = masks.remove_ground_artifacts(img)

    # Step 3: composition_crop + scale-to-fit + center to exact canvas
    step(3, "composition_crop")
    if not opts.skip_compose:
        img = cleanup.compose_on_canvas(img, opts.target_width * 8, opts.target_height * 8,
                                        pad=opts.pad, margin_frac=opts.fit_margin, align=opts.align)

    # Step 4: downsample_to_true_pixels (grid-snap, no anti-aliasing)
    step(4, "downsample_to_true_pixels")
    colors = opts.palette_limit or 0
    if opts.skip_compose:
        # source is already an exact target-sized canvas; just snap colors.
        cw = max(1, img.width // opts.target_width)
        ch = max(1, img.height // opts.target_height)
        img = snap.snap_uniform(img, cw, ch, k_colors=colors,
                                protect_extremes=opts.protect_extremes)
    elif opts.snap_mode:
        # mode-based true-pixel snap on the known 8x grid (spritefusion-style)
        img = snap.snap_uniform(img, 8, 8, k_colors=colors,
                                protect_extremes=opts.protect_extremes)
    else:
        img = pixelate.pixelate(img, opts.target_width, opts.target_height, colors=colors, seed=seed)
    # guarantee exact target dimensions
    if img.size != (opts.target_width, opts.target_height):
        img = img.resize((opts.target_width, opts.target_height), Image.NEAREST)

    # Step 5: palette_quantization — lock to a fixed palette when requested
    # (retro preset > character identity lock), optionally with ordered dither,
    # otherwise record the body palette.
    step(5, "palette_quantization")
    from . import palettes as _palettes
    target_palette = _palettes.get_preset(opts.palette_preset) or opts.lock_palette
    use_dither = (opts.dither or "none").lower() != "none"
    if target_palette:
        if use_dither:
            from . import dither as _dither
            img = _dither.apply_palette_dithered(
                img, target_palette, matrix=opts.dither,
                strength=opts.dither_strength, edge_safe=True)
        else:
            img = pixelate.apply_palette(img, target_palette)
    elif use_dither and colors:
        # no fixed palette: derive one (protecting extremes) then dither onto it
        from . import dither as _dither
        derived = pixelate.extract_palette(img, max_colors=colors,
                                           protect_extremes=opts.protect_extremes)
        img = _dither.apply_palette_dithered(
            img, derived, matrix=opts.dither,
            strength=opts.dither_strength, edge_safe=True)
    palette = pixelate.extract_palette(img, max_colors=colors or 16)
    # when a fixed palette is in force, validate against its true size
    effective_palette_limit = len(target_palette) if target_palette else opts.palette_limit

    # Step 6: alpha_snap
    step(6, "alpha_snap")
    img = cleanup.snap_alpha(img, allow_translucent=opts.allow_translucent)
    if opts.remove_floor_artifacts and opts.transparent:
        img = masks.remove_ground_artifacts(img)

    # Step 7: pixel_cleanup
    step(7, "pixel_cleanup")
    if opts.cleanup_orphans:
        img = cleanup.remove_orphan_pixels(img, min_neighbors=1)
    if opts.add_outline:
        img = cleanup.enforce_outline(img, color=tuple(opts.outline_color))

    # Step 8: validation
    step(8, "validation")
    v = validate.validate_asset(
        img, opts.target_width, opts.target_height,
        palette_limit=effective_palette_limit, allow_translucent=opts.allow_translucent,
    )

    # Step 9: export
    step(9, "export")
    prod_name = f"{asset_id}.png"
    prod_path = str(out / prod_name)
    exporters.save_production_png(img, prod_path)

    preview_paths: list[str] = []
    for scale in opts.generate_preview_scale:
        p = str(out / f"{asset_id}_preview_{scale}x.png")
        exporters.save_preview_png(img, p, scale=scale)
        preview_paths.append(p)

    manifest = exporters.build_manifest(
        asset_id=asset_id, asset_type=asset_type, prompt=prompt, image=img,
        seed=seed, model_profile_id=model_profile_id,
        license_metadata=license_metadata, palette=palette,
    )
    meta_path = str(out / f"{asset_id}.json")
    exporters.save_metadata_json(manifest.model_dump(), meta_path)

    return PixelPipelineResult(
        production_path=prod_path,
        preview_paths=preview_paths,
        metadata_path=meta_path,
        manifest=manifest,
        validation=v,
        image=img,
    )
