"""High-level orchestration functions invoked by the API & MCP bridge.

Each function maps a spec endpoint / MCP tool to the runtime + pixel pipeline,
returns a plain dict that becomes the job result. Progress is reported through
the Job object passed in.
"""

from __future__ import annotations

import base64
import io
import json
import random
import uuid
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .. import config
from ..characters import CharacterSystem
from ..pixel import exporters, validate
from ..pixel.pipeline import PixelPipelineOptions, run_true_pixel_pipeline
from ..runtime import external
from ..runtime.pipelines import GenParams, Pipelines
from ..schemas.jobs import (AnimateRequest, BatchRecipeRequest, CreateCharacterRequest,
                            GenerateRequest, InpaintRequest, IsometricTileRequest,
                            SidescrollerTilesetRequest, TilesetRequest,
                            TopdownTilesetRequest, TurnaroundRequest, UIAssetRequest)
from .security import resolve_output_folder


def _new_asset_id() -> str:
    return uuid.uuid4().hex[:12]


def _progress(job, frac: float, msg: str = ""):
    job.progress = float(frac)
    if msg:
        job.log_lines.append(msg)


def _record_asset(asset_id: str, asset_type: str, prompt: str, seed, width: int,
                  height: int, result, character_profile_id: Optional[str] = None) -> None:
    """Persist a finished sprite to the assets table so it can be browsed and
    bundled by the engine export. Best-effort â€” never fails a job."""
    try:
        from .. import db
        import json as _json
        with db.get_session() as s:
            existing = s.exec(db.select(db.AssetRow).where(
                db.AssetRow.asset_id == asset_id)).first()
            if existing:
                return
            s.add(db.AssetRow(
                asset_id=asset_id, asset_type=asset_type, prompt=prompt or "",
                seed=int(seed) if seed is not None else None,
                width=int(width), height=int(height),
                character_profile_id=character_profile_id,
                production_path=getattr(result, "production_path", None),
                preview_path=(result.preview_paths[0] if getattr(result, "preview_paths", None) else None),
                metadata_path=getattr(result, "metadata_path", None),
                validation_json=_json.dumps(getattr(result, "validation", None).model_dump()
                                            if getattr(result, "validation", None) else {}),
            ))
            s.commit()
    except Exception:
        pass


# Ã¢â€â‚¬Ã¢â€â‚¬ shared runtime accessors Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def _record_file_asset(asset_id: str, asset_type: str, prompt: str, seed, width: int,
                       height: int, production_path: str, metadata_path: Optional[str] = None,
                       preview_path: Optional[str] = None) -> None:
    try:
        from .. import db
        with db.get_session() as s:
            existing = s.exec(db.select(db.AssetRow).where(
                db.AssetRow.asset_id == asset_id)).first()
            if existing:
                return
            s.add(db.AssetRow(
                asset_id=asset_id, asset_type=asset_type, prompt=prompt or "",
                seed=int(seed) if seed is not None else None,
                width=int(width), height=int(height), production_path=production_path,
                preview_path=preview_path, metadata_path=metadata_path,
                validation_json="{}",
            ))
            s.commit()
    except Exception:
        pass

_loader = None
_pipelines = None
_chars = None


def get_loader():
    global _loader
    if _loader is None:
        from ..runtime.model_loader import ModelLoader
        _loader = ModelLoader()
    return _loader


def get_pipelines() -> Pipelines:
    global _pipelines
    if _pipelines is None:
        _pipelines = Pipelines(get_loader())
    return _pipelines


def get_chars() -> CharacterSystem:
    global _chars
    if _chars is None:
        _chars = CharacterSystem(get_pipelines())
    return _chars


# â”€â”€ generation provider + adapter helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _is_flux2_name(value: str) -> bool:
    blob = (value or "").lower()
    return any(token in blob for token in ("flux.2", "flux2", "flux-2", "klein"))


def _lora_flux_generation(idx) -> Optional[str]:
    """Return flux1/flux2 for FLUX-family LoRAs, or None for non-FLUX/unknown.

    FLUX.1 LoRAs cannot be loaded into FLUX.2/Klein transformers: the layer
    shapes differ and diffusers raises a state_dict size mismatch. Treat a
    generic FLUX LoRA as FLUX.1 unless it explicitly says FLUX.2/Klein.
    """
    compat = idx.base_compatibility or []
    if "flux_optional_future" not in compat:
        return None
    blob = " ".join(str(v or "") for v in (
        idx.model_id, idx.display_name, idx.source_url, idx.local_path,
        idx.trigger, idx.notes, " ".join(compat),
    )).lower()
    return "flux2" if _is_flux2_name(blob) else "flux1"


def _handle_flux_generation(handle) -> Optional[str]:
    if not handle or (handle.base_kind or "") != "flux_optional_future":
        return None
    return "flux2" if _is_flux2_name(handle.model_id) else "flux1"


def _lora_base_compatible(loader, lora_id: str) -> bool:
    """True if the LoRA matches the resident base family and FLUX generation."""
    handle = loader.handle
    if not handle:
        return True
    idx = loader._find_index(lora_id)
    if idx is None or not idx.base_compatibility:
        return True
    base_tag = handle.base_kind or "sdxl"
    if base_tag not in idx.base_compatibility:
        return False
    active_flux = _handle_flux_generation(handle)
    lora_flux = _lora_flux_generation(idx)
    if active_flux and lora_flux:
        return active_flux == lora_flux
    return True


def apply_selected_lora(loader, lora_id: Optional[str] = None,
                        weight: Optional[float] = None) -> Optional[str]:
    """Attach the effective style LoRA to the resident pipeline.

    Precedence: the per-job ``lora_id`` (selected in a tab) > the global
    ``pixel_lora_id`` in Settings. One LoRA at a time â€” any previously attached
    LoRA is detached first, so switching the selection per job works. No-op when
    unset, incompatible with the base, or not installed.
    """
    cfg = config.RuntimeConfig.load()
    want = (lora_id or cfg.pixel_lora_id or "").strip()
    handle = loader.handle
    if not want or not handle or not handle.pipeline:
        return None
    if want in handle.active_loras:
        return want
    if not _lora_base_compatible(loader, want):
        return None
    # detach any previously-attached LoRA so the selection is exclusive
    for prev in list(handle.active_loras.keys()):
        try:
            loader.detach_lora(prev)
        except Exception:
            pass
    try:
        w = float(weight) if weight is not None else float(cfg.pixel_lora_weight or 1.0)
        loader.attach_lora(want, weight=w)
        return want
    except Exception:
        return None


# Back-compat alias (uses the global pixel LoRA only).
def apply_pixel_lora(loader) -> Optional[str]:
    return apply_selected_lora(loader, None)


def prepare_local_base(loader, base_id: str, lora_id: Optional[str] = None,
                       lora_weight: Optional[float] = None) -> Optional[str]:
    """Load the local base model + selected/global LoRA, unless Gemini is the
    active provider (then no local generation model is needed)."""
    if config.RuntimeConfig.load().gemini_enabled():
        return None
    loader.load_base_model(base_id)
    apply_selected_lora(loader, lora_id, lora_weight)
    return base_id


def generate_source_image(job, pipes, gen: GenParams, base_id: str,
                          lora_id: Optional[str] = None,
                          lora_weight: Optional[float] = None) -> tuple[Image.Image, int, str]:
    """Produce the raw source image either from Gemini (opt-in, key-gated) or the
    local SDXL/FLUX pipeline. Gemini failures fall back to local with a warning.
    Returns (image, seed, model_profile_id)."""
    cfg = config.RuntimeConfig.load()
    if cfg.gemini_enabled():
        try:
            img = external.generate_image(
                gen.prompt, api_key=cfg.gemini_api_key, model=cfg.gemini_model,
                negative=gen.negative_prompt, width=gen.width, height=gen.height)
            seed = gen.seed if gen.seed not in (None, -1) else random.randint(0, 2**32 - 1)
            return img, int(seed), f"gemini:{cfg.gemini_model}"
        except external.GeminiError as e:
            if job is not None:
                job.warnings.append(f"Gemini unavailable, used local model: {e}")
            # ensure a local base + LoRA are ready before falling back
            get_loader().load_base_model(base_id)
            apply_selected_lora(get_loader(), lora_id, lora_weight)
    img, seed = pipes.text_to_image(gen, base_model_id=base_id)
    return img, seed, base_id


# Ã¢â€â‚¬Ã¢â€â‚¬ generate asset Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def generate_asset(job, req: GenerateRequest) -> dict:
    """POST /api/jobs/generate Ã¢â‚¬â€ text/item/prop/character/ui/background."""
    out = resolve_output_folder(req.output_root)
    w = int(req.target_size.width) if str(req.target_size.width).isdigit() else 64
    h = int(req.target_size.height) if str(req.target_size.height).isdigit() else 64

    _progress(job, 0.05, "loading generation provider")
    loader = get_loader()
    base_id = config.RuntimeConfig.load().last_base_model_id
    prepare_local_base(loader, base_id, req.lora_id, req.lora_weight)

    _progress(job, 0.15, "generating source image")
    gen = GenParams(
        prompt=config.build_prompt(req.prompt, req.asset_type, solid_bg=req.transparent),
        negative_prompt=config.build_negative(req.negative_prompt, req.asset_type),
        width=768, height=768,
        num_inference_steps=req.num_inference_steps or (8 if req.quality_preset == "fast" else 30),
        guidance_scale=req.guidance_scale or 7.5,
        seed=req.seed,
        quality_preset=req.quality_preset or "quality",
    )
    pipes = get_pipelines()
    base_img, used_seed, model_id = generate_source_image(
        job, pipes, gen, base_id, req.lora_id, req.lora_weight)

    _progress(job, 0.6, "running true-pixel pipeline")
    # Each category gets its own framing/alignment + model-based background
    # removal so it renders as a clean, isolated, game-ready asset.
    prof = config.asset_profile(req.asset_type)
    opts = PixelPipelineOptions(
        target_width=w, target_height=h, transparent=req.transparent,
        palette_limit=req.palette_limit, seed=used_seed,
        generate_preview_scale=req.generate_preview_scale,
        align=prof["align"], fit_margin=prof["margin"],
        use_segmentation=req.transparent and prof["isolated"],
        remove_floor_artifacts=req.transparent and req.asset_type in {"character", "character_turnaround", "character_animation", "prop", "item", "map_object"},
        palette_preset=req.palette_preset, dither=req.dither or "none",
        dither_strength=req.dither_strength if req.dither_strength is not None else 0.6,
        protect_extremes=bool(req.protect_extremes),
    )
    asset_id = _new_asset_id()
    result = run_true_pixel_pipeline(
        base_img, str(out), req.prompt, asset_id=asset_id,
        asset_type=req.asset_type, options=opts, seed=used_seed,
        model_profile_id=model_id,
        on_progress=lambda n, m: _progress(job, 0.6 + 0.35 * n / 9.0, m),
    )
    _record_asset(asset_id, req.asset_type, req.prompt, used_seed, w, h, result)
    _progress(job, 1.0, "done")
    if not result.validation.ok:
        job.warnings.append(f"validation: {'; '.join(result.validation.errors)}")
    return {
        "production_png": result.production_path,
        "preview_png": result.preview_paths[0] if result.preview_paths else None,
        "preview_png_8x": result.preview_paths[1] if len(result.preview_paths) > 1 else None,
        "metadata_json": result.metadata_path,
        "seed": used_seed,
        "model_profile_id": model_id,
        "outputs": {
            "production_png": result.production_path,
            "preview_png": result.preview_paths[0] if result.preview_paths else None,
            "preview_png_8x": result.preview_paths[1] if len(result.preview_paths) > 1 else None,
            "metadata_json": result.metadata_path,
            "files": [p for p in [result.production_path, *(result.preview_paths), result.metadata_path] if p],
            "named_files": {
                "production": result.production_path,
                "preview_4x": result.preview_paths[0] if result.preview_paths else None,
                "preview_8x": result.preview_paths[1] if len(result.preview_paths) > 1 else None,
                "metadata": result.metadata_path,
            },
        },
    }


# Ã¢â€â‚¬Ã¢â€â‚¬ create character Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def create_character(job, req) -> dict:
    chars = get_chars()
    _progress(job, 0.05, "creating profile")
    profile = chars.create_profile(
        name=req.name, description=req.description or req.prompt or req.name,
        width=req.width, height=req.height, directions=req.directions,
        style_profile_id=req.style_profile_id, base_seed=req.seed)
    out = resolve_output_folder(req.output_folder, project_name="characters")
    loader = get_loader()
    loader.load_base_model(config.RuntimeConfig.load().last_base_model_id)
    apply_selected_lora(loader, req.lora_id, req.lora_weight)
    _progress(job, 0.2, "generating canonical reference")
    ref = chars.generate_canonical_reference(profile, str(out), prompt=req.prompt)
    _progress(job, 1.0, "done")
    return {
        "character_profile_id": profile.character_id,
        "production_png": ref["production_png"],
        "preview_png": ref["preview_png"],
        "preview_png_8x": ref.get("preview_png_8x"),
        "metadata_json": ref["metadata_json"],
        "outputs": {
            "production_png": ref["production_png"],
            "preview_png": ref["preview_png"],
            "preview_png_8x": ref.get("preview_png_8x"),
            "metadata_json": ref["metadata_json"],
            "files": [p for p in [ref["production_png"], ref["preview_png"], ref.get("preview_png_8x"), ref["metadata_json"]] if p],
            "named_files": {
                "canonical": ref["production_png"],
                "preview_4x": ref["preview_png"],
                "preview_8x": ref.get("preview_png_8x"),
                "metadata": ref["metadata_json"],
            },
        },
    }


# Ã¢â€â‚¬Ã¢â€â‚¬ turnaround Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def generate_turnaround(job, req) -> dict:
    chars = get_chars()
    profile = chars.get_profile(req.character_profile_id)
    if not profile:
        raise ValueError(f"unknown character_profile_id: {req.character_profile_id}")
    out = resolve_output_folder(req.output_folder, project_name="characters")
    loader = get_loader()
    loader.load_base_model(config.RuntimeConfig.load().last_base_model_id)
    apply_selected_lora(loader, req.lora_id, req.lora_weight)
    _progress(job, 0.1, "generating turnaround")
    res = chars.generate_turnaround(profile, req.directions, str(out), req.width, req.height)
    _progress(job, 1.0, "done")
    first = next(iter(res["directions"].values()), None)
    direction_files = [p for p in res["directions"].values() if p]
    return {"directions": res["directions"], "count": res["count"],
            "production_png": first,
            "files": direction_files,
            "outputs": {"production_png": first,
                        "files": direction_files,
                        "turnaround_pngs": res["directions"],
                        "named_files": res["directions"]}}


# Ã¢â€â‚¬Ã¢â€â‚¬ animate Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def animate_character(job, req) -> dict:
    chars = get_chars()
    profile = chars.get_profile(req.character_profile_id)
    if not profile:
        raise ValueError(f"unknown character_profile_id: {req.character_profile_id}")
    out = resolve_output_folder(req.output_folder, project_name="characters")
    loader = get_loader()
    loader.load_base_model(config.RuntimeConfig.load().last_base_model_id)
    apply_selected_lora(loader, req.lora_id, req.lora_weight)
    _progress(job, 0.1, "generating animation")
    res = chars.generate_animation(profile, req.animation, req.directions,
                                   req.frames_per_direction, str(out))
    _progress(job, 1.0, "done")
    return {
        "sprite_sheet_png": res["sprite_sheet_png"],
        "metadata_json": res["metadata_json"],
        "gif_preview": res["gif_preview"],
        "outputs": {
            "sprite_sheet_png": res["sprite_sheet_png"],
            "metadata_json": res["metadata_json"],
            "gif_preview": res.get("gif_preview"),
            "webp_preview": res.get("webp_preview"),
            "files": [p for p in [res["sprite_sheet_png"], res["metadata_json"], res.get("gif_preview"), res.get("webp_preview")] if p],
            "named_files": {
                "sprite_sheet": res["sprite_sheet_png"],
                "metadata": res["metadata_json"],
                "gif_preview": res.get("gif_preview"),
                "webp_preview": res.get("webp_preview"),
            },
        },
    }


# Ã¢â€â‚¬Ã¢â€â‚¬ portrait Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def generate_portrait(job, req) -> dict:
    chars = get_chars()
    profile = chars.get_profile(req.character_profile_id)
    if not profile:
        raise ValueError(f"unknown character_profile_id: {req.character_profile_id}")
    out = resolve_output_folder(req.output_folder, project_name="characters")
    loader = get_loader()
    loader.load_base_model(config.RuntimeConfig.load().last_base_model_id)
    apply_selected_lora(loader, req.lora_id, req.lora_weight)
    _progress(job, 0.1, "generating portrait")
    res = chars.generate_portrait(
        profile, str(out), width=req.width, height=req.height,
        palette_limit=req.palette_limit, expression=req.expression,
        transparent=req.transparent, seed=req.seed)
    _progress(job, 1.0, "done")
    files = [p for p in [res["production_png"], res.get("preview_png"),
                         res.get("preview_png_8x"), res["metadata_json"]] if p]
    return {
        "production_png": res["production_png"],
        "preview_png": res.get("preview_png"),
        "preview_png_8x": res.get("preview_png_8x"),
        "metadata_json": res["metadata_json"],
        "seed": res.get("seed"),
        "outputs": {
            "production_png": res["production_png"],
            "preview_png": res.get("preview_png"),
            "preview_png_8x": res.get("preview_png_8x"),
            "metadata_json": res["metadata_json"],
            "files": files,
            "named_files": {
                "portrait": res["production_png"],
                "preview_4x": res.get("preview_png"),
                "preview_8x": res.get("preview_png_8x"),
                "metadata": res["metadata_json"],
            },
        },
    }


# Ã¢â€â‚¬Ã¢â€â‚¬ inpaint Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def inpaint_asset(job, req: InpaintRequest) -> dict:
    img = Image.open(req.asset_path).convert("RGBA")
    mask = Image.open(req.mask_path).convert("L")
    out = resolve_output_folder(req.output_folder, project_name="inpaint")
    _progress(job, 0.1, "loading model")
    loader = get_loader()
    loader.load_base_model(config.RuntimeConfig.load().last_base_model_id)
    _progress(job, 0.3, "inpainting")
    pipes = get_pipelines()
    gen = GenParams(prompt=req.edit_prompt, width=768, height=768, num_inference_steps=30)
    result_img, seed = pipes.inpaint(img, mask, gen, strength=req.strength)
    _progress(job, 0.7, "running true-pixel pipeline")
    opts = PixelPipelineOptions(target_width=img.width, target_height=img.height,
                                transparent=True, palette_limit=24, seed=seed,
                                generate_preview_scale=[])
    asset_id = _new_asset_id()
    pp = run_true_pixel_pipeline(result_img, str(out), req.edit_prompt,
                                 asset_id=asset_id, asset_type="inpaint_edit",
                                 options=opts, seed=seed)
    _progress(job, 1.0, "done")
    return {"production_png": pp.production_path, "seed": seed,
            "outputs": {"production_png": pp.production_path,
                        "metadata_json": pp.metadata_path,
                        "files": [pp.production_path, pp.metadata_path],
                        "named_files": {"production": pp.production_path,
                                        "metadata": pp.metadata_path}}}


# Ã¢â€â‚¬Ã¢â€â‚¬ tileset Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def _gen_tile_texture(pipes, description, view, seed, transparent, negative_view=""):
    """Generate one tile texture at a model-friendly resolution (single swatch,
    never a whole map)."""
    neg = (config.DEFAULT_NEGATIVE_PROMPT +
           ", tilemap, level map, whole map, scene, multiple tiles, grid, "
           "border, frame, character, object, text")
    if negative_view:
        neg += ", " + negative_view
    prompt = (f"{description}, {view}, single tile, evenly lit, no border, no grid"
              + config.DEFAULT_PIXEL_SUFFIX)
    gen = GenParams(prompt=prompt, negative_prompt=neg, width=512, height=512,
                    seed=seed, num_inference_steps=24)
    img, _ = pipes.text_to_image(gen)
    return img


def create_tileset(job, req: TilesetRequest) -> dict:
    """Generate a tileset whose pipeline matches its TYPE:
       top_down   â†’ seamless repeating ground tiles
       sidescroller â†’ side-view platform tiles, transparent background
       isometric  â†’ diamond iso tiles, transparent corners
       wang       â†’ a 16-tile corner-blob autotile set (lowerâ†”upper transition)
    """
    from ..pixel import tiles as tilemod

    out = resolve_output_folder(req.output_folder, project_name="tilesets")
    loader = get_loader()
    loader.load_base_model(config.RuntimeConfig.load().last_base_model_id)
    apply_selected_lora(loader, req.lora_id, req.lora_weight)
    pipes = get_pipelines()
    prof = tilemod.tileset_profile(req.tileset_type)
    tw, th = req.tile_width, req.tile_height

    def _snap_tile(src, seed, palette):
        opts = PixelPipelineOptions(
            target_width=tw, target_height=th,
            transparent=prof["transparent"], palette_limit=16, seed=seed,
            generate_preview_scale=[], fit_margin=0.0, lock_palette=palette,
            use_segmentation=False)
        aid = _new_asset_id()
        return run_true_pixel_pipeline(
            src, str(out), f"{req.description} {req.tileset_type} tile",
            asset_id=aid, asset_type="tileset_" + req.tileset_type, options=opts, seed=seed)

    tiles: list[Image.Image] = []
    first_palette: list | None = None

    if prof["wang"]:
        # Wang autotile: one "lower" base texture + one "upper" transition texture,
        # composited into the 16 corner combinations.
        _progress(job, 0.2, "generating base texture")
        lower = _gen_tile_texture(pipes, req.description, prof["view"], 0, False, prof.get("negative", ""))
        lower = tilemod.make_seamless(lower)
        _progress(job, 0.5, "generating transition texture")
        upper = _gen_tile_texture(pipes, req.description + ", lighter edge / transition variant",
                                  prof["view"], 1, False, prof.get("negative", ""))
        upper = tilemod.make_seamless(upper)
        wang_tiles, _sheet = tilemod.wang_16(lower, upper, tw, th)
        tiles = wang_tiles
        n = 16
    else:
        n = req.tile_count or 8
        for i in range(n):
            _progress(job, 0.1 + 0.7 * i / max(1, n), f"tile {i+1}/{n}")
            src = _gen_tile_texture(pipes, req.description, prof["view"], i, prof["transparent"], prof.get("negative", ""))
            if prof["seamless"]:
                src = tilemod.make_seamless(src)
            pp = _snap_tile(src, i, first_palette)
            if first_palette is None:
                first_palette = pp.manifest.palette or None
            img = pp.image
            if prof["shape"] == "diamond":
                img = tilemod.apply_diamond(img)
            tiles.append(img)

    # compose into a sheet (RGBA so transparent types keep their alpha)
    cols = 4 if prof["wang"] else min(4, n)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * tw, rows * th), (0, 0, 0, 0))
    for i, t in enumerate(tiles):
        r, c = i // cols, i % cols
        sheet.alpha_composite(t.convert("RGBA"), (c * tw, r * th))
    sheet_path = str(out / f"tileset_{req.tileset_type}_{_new_asset_id()}.png")
    sheet.save(sheet_path, format="PNG")
    meta_path = sheet_path.replace(".png", ".json")
    Path(meta_path).write_text(__import__("json").dumps({
        "tileset_type": req.tileset_type, "tile_width": tw, "tile_height": th,
        "tile_count": n, "rows": rows, "cols": cols, "seamless": prof["seamless"],
        "wang_autotile": prof["wang"], "transparent": prof["transparent"],
    }, indent=2), encoding="utf-8")
    sheet_asset_type = {
        "top_down": "tileset_top_down",
        "sidescroller": "tileset_sidescroller",
        "isometric": "isometric_tile",
        "wang": "tileset_top_down",
    }.get(req.tileset_type, "tileset_top_down")
    _record_file_asset(Path(sheet_path).stem, sheet_asset_type, req.description, None,
                       cols * tw, rows * th, sheet_path, meta_path)
    # Seam validation (audit Â§3): warn if a "seamless" tile won't repeat cleanly.
    if prof["seamless"] and tiles:
        worst = max(tilemod.seam_score(t) for t in tiles)
        if worst > 12.0:
            job.warnings.append(
                f"tileset seams may be visible (seam score {worst:.1f}/255) â€” "
                "regenerate or raise the seamless healing band")
    _progress(job, 1.0, "done")
    return {"sprite_sheet_png": sheet_path, "metadata_json": meta_path,
            "outputs": {"sprite_sheet_png": sheet_path,
                        "metadata_json": meta_path,
                        "files": [sheet_path, meta_path],
                        "named_files": {"sprite_sheet": sheet_path,
                                        "metadata": meta_path}}}


# Ã¢â€â‚¬Ã¢â€â‚¬ export pack Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬



# -- batch recipes -------------------------------------------------------------

_RECIPE_ITEM_NOUNS = [
    "sword", "shield", "potion", "key", "ring", "scroll", "gem", "helmet",
    "boots", "amulet", "axe", "bow", "staff", "coin", "book", "lantern",
]

_RECIPE_UI_ELEMENTS = [
    ("dialogue panel", 256, 128, ["panel"]),
    ("primary button", 128, 48, ["button"]),
    ("inventory slot", 64, 64, ["slot"]),
    ("health bar", 192, 32, ["bar"]),
    ("tab header", 128, 40, ["tab"]),
    ("small icon frame", 64, 64, ["frame"]),
    ("quest notification panel", 192, 72, ["panel", "badge"]),
    ("tooltip box", 160, 80, ["panel"]),
]


def _safe_recipe_name(value: str) -> str:
    keep = []
    for ch in (value or "recipe").lower():
        keep.append(ch if ch.isalnum() else "_")
    name = "".join(keep).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name[:48] or "recipe"


def _batch_add_result(result: dict[str, Any], label: str, files: list[str],
                      named: dict[str, str], summaries: list[dict[str, Any]]) -> None:
    outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
    result_files: list[str] = []
    for key in ("production_png", "preview_png", "preview_png_8x", "sprite_sheet_png",
                "metadata_json", "gif_preview", "webp_preview", "zip_path"):
        value = outputs.get(key) or result.get(key)
        if value:
            result_files.append(str(value))
    for value in outputs.get("files") or result.get("files") or []:
        if value:
            result_files.append(str(value))
    for key, value in (outputs.get("named_files") or {}).items():
        if value:
            named[f"{label}_{key}"] = str(value)
            result_files.append(str(value))
    for key, value in (outputs.get("turnaround_pngs") or result.get("directions") or {}).items():
        if value:
            named[f"{label}_{key}"] = str(value)
            result_files.append(str(value))
    seen = {p.lower() for p in files}
    for item_path in result_files:
        if item_path.lower() not in seen:
            seen.add(item_path.lower())
            files.append(item_path)
    summaries.append({
        "label": label,
        "asset_id": result.get("asset_id") or result.get("object_id") or result.get("ui_asset_id"),
        "character_profile_id": result.get("character_profile_id"),
        "files": result_files,
    })


def _recipe_palette(theme: str) -> dict[str, tuple[int, int, int, int]]:
    blob = (theme or "").lower()
    if "forest" in blob or "nature" in blob:
        return {
            "dark": (18, 38, 34, 255), "mid": (39, 91, 65, 255),
            "light": (92, 173, 112, 255), "accent": (84, 220, 190, 255),
            "shadow": (8, 13, 18, 255), "empty": (0, 0, 0, 0),
        }
    if "ice" in blob or "crystal" in blob:
        return {
            "dark": (25, 36, 68, 255), "mid": (48, 85, 132, 255),
            "light": (134, 215, 230, 255), "accent": (93, 242, 255, 255),
            "shadow": (8, 13, 26, 255), "empty": (0, 0, 0, 0),
        }
    return {
        "dark": (42, 34, 48, 255), "mid": (96, 68, 92, 255),
        "light": (205, 157, 110, 255), "accent": (90, 198, 184, 255),
        "shadow": (12, 12, 18, 255), "empty": (0, 0, 0, 0),
    }


def _rect(draw, xy, fill, outline=None):
    x0, y0, x1, y1 = [int(round(v)) for v in xy]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline)


def _draw_pixel_ui(element: str, width: int, height: int, theme: str) -> Image.Image:
    from PIL import ImageDraw
    p = _recipe_palette(theme)
    img = Image.new("RGBA", (int(width), int(height)), p["empty"])
    d = ImageDraw.Draw(img)
    w, h = img.size
    e = element.lower()

    def frame(x0, y0, x1, y1, inset=4):
        inset = max(0, min(int(inset), max(0, (x1 - x0) // 2), max(0, (y1 - y0) // 2)))
        _rect(d, [x0, y0, x1, y1], p["shadow"])
        _rect(d, [x0 + 2, y0 + 2, x1 - 2, y1 - 2], p["dark"], p["light"])
        _rect(d, [x0 + inset, y0 + inset, x1 - inset, y1 - inset], p["mid"], p["accent"])

    if "button" in e:
        frame(4, h // 5, w - 5, h - h // 5, 5)
        _rect(d, [10, h // 5 + 5, w - 11, h // 2], (*p["light"][:3], 255))
        _rect(d, [12, h // 2, w - 13, h - h // 5 - 5], p["mid"])
    elif "slot" in e or "frame" in e:
        s = min(w, h) - 8
        x0, y0 = (w - s) // 2, (h - s) // 2
        frame(x0, y0, x0 + s - 1, y0 + s - 1, max(4, s // 8))
        _rect(d, [x0 + s // 4, y0 + s // 4, x0 + s * 3 // 4, y0 + s * 3 // 4], (0, 0, 0, 35))
    elif "bar" in e:
        frame(3, h // 3, w - 4, h * 2 // 3, 4)
        fill_w = max(8, int((w - 18) * 0.72))
        _rect(d, [9, h // 3 + 6, 9 + fill_w, h * 2 // 3 - 6], p["accent"])
        _rect(d, [9, h // 3 + 6, 9 + fill_w, h // 2], p["light"])
    elif "tab" in e:
        _rect(d, [5, h // 4, w - 6, h - 3], p["shadow"])
        _rect(d, [7, 3, w - 8, h - 5], p["dark"], p["light"])
        _rect(d, [13, 8, w - 14, h - 11], p["mid"], p["accent"])
    else:
        frame(4, 4, w - 5, h - 5, 8)
        _rect(d, [12, 12, w - 13, 20], p["light"])
        _rect(d, [12, h - 22, w - 13, h - 14], p["shadow"])
    return img


def _create_recipe_ui_asset(job, out: Path, theme: str, element: str,
                            width: int, height: int, seed: Optional[int]) -> dict:
    from ..pixel import exporters as _exporters
    import json as _json
    aid = _new_asset_id()
    img = _draw_pixel_ui(element, int(width), int(height), theme)
    prod = str(out / f"{aid}.png")
    prev = str(out / f"{aid}_preview_8x.png")
    meta = str(out / f"{aid}.json")
    _exporters.save_production_png(img, prod)
    _exporters.save_preview_png(img, prev, scale=8)
    manifest = _exporters.build_manifest(
        aid, "ui_element", f"{theme} {element}", img, seed=seed,
        extra={"recipe_element": element, "deterministic_ui": True})
    Path(meta).write_text(_json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
    _record_file_asset(aid, "ui_element", f"{theme} {element}", seed,
                       int(width), int(height), prod, meta, prev)
    return {
        "ui_asset_id": aid,
        "production_png": prod,
        "preview_png_8x": prev,
        "metadata_json": meta,
        "outputs": {
            "production_png": prod,
            "preview_png_8x": prev,
            "metadata_json": meta,
            "files": [prod, prev, meta],
            "named_files": {"production": prod, "preview_8x": prev, "metadata": meta},
        },
    }


def create_batch_recipe(job, req: BatchRecipeRequest) -> dict:
    """Generate a cohesive multi-asset pack from one shared recipe request."""
    root = resolve_output_folder(req.output_folder, project_name="recipes")
    recipe_id = f"{req.recipe}_{_safe_recipe_name(req.theme)}_{_new_asset_id()}"
    out = root / recipe_id
    out.mkdir(parents=True, exist_ok=True)

    files: list[str] = []
    named: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    warnings_start = len(job.warnings)

    def add(result: dict[str, Any], label: str) -> dict[str, Any]:
        _batch_add_result(result, label, files, named, summaries)
        return result

    seed = int(req.seed) if req.seed is not None else None
    _progress(job, 0.03, f"starting {req.recipe} recipe")

    if req.recipe == "item_set":
        total = max(1, req.count)
        for i in range(total):
            noun = _RECIPE_ITEM_NOUNS[i % len(_RECIPE_ITEM_NOUNS)]
            item_seed = seed + i if seed is not None else None
            prompt = f"{req.theme} {noun}, single inventory item icon"
            _progress(job, 0.05 + 0.9 * i / total, f"item {i + 1}/{total}: {noun}")
            add(generate_asset(job, GenerateRequest(
                asset_type="item", prompt=prompt,
                target_size={"width": req.size, "height": req.size},
                transparent=True, palette_limit=req.palette_limit,
                output_root=str(out), seed=item_seed,
                lora_id=req.lora_id, lora_weight=req.lora_weight,
                exclude_from_saved_outputs=True,
            )), f"item_{i + 1:02d}_{noun}")

    elif req.recipe == "ui_pack":
        total = max(1, min(req.count, len(_RECIPE_UI_ELEMENTS)))
        for i, (element, width, height, parts) in enumerate(_RECIPE_UI_ELEMENTS[:total]):
            item_seed = seed + i if seed is not None else None
            _progress(job, 0.05 + 0.9 * i / total, f"ui {i + 1}/{total}: {element}")
            add(_create_recipe_ui_asset(
                job, out, req.theme, element, width, height, item_seed),
                f"ui_{i + 1:02d}_{_safe_recipe_name(element)}")

    elif req.recipe == "tileset_pack":
        _progress(job, 0.08, "top-down wang tileset")
        add(create_topdown_tileset(job, TopdownTilesetRequest(
            lower_description=f"{req.theme} ground base",
            upper_description=f"{req.theme} path or raised terrain",
            transition_description=f"{req.theme} edge transition",
            tile_size={"width": req.size, "height": req.size},
            output_folder=str(out), seed=seed,
            lora_id=req.lora_id, lora_weight=req.lora_weight,
            exclude_from_saved_outputs=True,
        )), "tiles_topdown_wang")
        _progress(job, 0.42, "sidescroller platform tileset")
        add(create_sidescroller_tileset(job, SidescrollerTilesetRequest(
            lower_description=f"{req.theme} platform block",
            transition_description=f"{req.theme} surface trim",
            tile_size={"width": req.size, "height": req.size},
            output_folder=str(out), seed=(seed + 1 if seed is not None else None),
            lora_id=req.lora_id, lora_weight=req.lora_weight,
            exclude_from_saved_outputs=True,
        )), "tiles_sidescroller")
        _progress(job, 0.72, "isometric tile")
        add(create_isometric_tile(job, IsometricTileRequest(
            description=f"{req.theme} isometric terrain tile", size=req.size,
            output_folder=str(out), seed=(seed + 2 if seed is not None else None),
            lora_id=req.lora_id, lora_weight=req.lora_weight,
            exclude_from_saved_outputs=True,
        )), "tiles_isometric")

    elif req.recipe == "character_pack":
        _progress(job, 0.05, "canonical character")
        created = add(create_character(job, CreateCharacterRequest(
            name=req.theme.title(), description=req.theme,
            width=req.size, height=req.size, directions=req.directions,
            output_folder=str(out), seed=seed,
            lora_id=req.lora_id, lora_weight=req.lora_weight,
            exclude_from_saved_outputs=True,
        )), "character_canonical")
        character_id = created.get("character_profile_id")
        if not character_id:
            raise ValueError("character recipe did not create a character_profile_id")
        _progress(job, 0.35, "character turnaround")
        add(generate_turnaround(job, TurnaroundRequest(
            character_profile_id=character_id, directions=req.directions,
            width=req.size, height=req.size, output_folder=str(out),
            lora_id=req.lora_id, lora_weight=req.lora_weight,
            exclude_from_saved_outputs=True,
        )), "character_turnaround")
        animations = [a.strip() for a in req.animations if a and a.strip()] or ["idle", "walk"]
        for i, anim in enumerate(animations):
            _progress(job, 0.55 + 0.4 * i / max(1, len(animations)), f"animation {anim}")
            add(animate_character(job, AnimateRequest(
                character_profile_id=character_id, animation=anim,
                directions=req.directions, output_folder=str(out),
                lora_id=req.lora_id, lora_weight=req.lora_weight,
                exclude_from_saved_outputs=True,
            )), f"character_anim_{_safe_recipe_name(anim)}")

    manifest_path = out / f"{recipe_id}_manifest.json"
    manifest = {
        "recipe_id": recipe_id,
        "recipe": req.recipe,
        "theme": req.theme,
        "count": req.count,
        "size": req.size,
        "directions": req.directions,
        "animations": req.animations,
        "lora_id": req.lora_id,
        "outputs": summaries,
        "warnings": job.warnings[warnings_start:],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files.append(str(manifest_path))
    named["manifest"] = str(manifest_path)
    _progress(job, 1.0, "recipe done")

    first_png = next((p for p in files if p.lower().endswith(".png")), None)
    first_sheet = next((p for p in files if "sheet" in Path(p).stem.lower() and p.lower().endswith(".png")), None)
    return {
        "asset_id": recipe_id,
        "production_png": first_png,
        "sprite_sheet_png": first_sheet,
        "metadata_json": str(manifest_path),
        "outputs": {
            "production_png": first_png,
            "sprite_sheet_png": first_sheet,
            "metadata_json": str(manifest_path),
            "files": files,
            "named_files": named,
            "extra": {"recipe_id": recipe_id},
        },
    }


# Local PixelLab-style object/UI/tile workflows.
PIXELLAB_OBJECT_DIRECTIONS = [
    ("south", "front view facing south toward camera"),
    ("south-east", "three-quarter front view facing south-east"),
    ("east", "right side view facing east"),
    ("north-east", "three-quarter back view facing north-east"),
    ("north", "back view facing north away from camera"),
    ("north-west", "three-quarter back view facing north-west"),
    ("west", "left side view facing west"),
    ("south-west", "three-quarter front view facing south-west"),
]


def _asset_record(asset_id: str):
    try:
        from .. import db
        with db.get_session() as s:
            return s.exec(db.select(db.AssetRow).where(db.AssetRow.asset_id == asset_id)).first()
    except Exception:
        return None



def _image_from_data_url(value: Optional[str]) -> Optional[Image.Image]:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=False)
        return Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception as exc:
        raise ValueError(f"invalid uploaded source image: {exc}") from exc

def pixellab_asset_response(asset_id: str) -> dict:
    row = _asset_record(asset_id)
    if not row:
        return {"error": f"asset not found: {asset_id}"}
    files = [p for p in [row.production_path, row.preview_path, row.metadata_path] if p]
    return {
        "object_id": row.asset_id,
        "status": "completed",
        "asset_type": row.asset_type,
        "description": row.prompt,
        "production_png": row.production_path,
        "preview_png": row.preview_path,
        "metadata_json": row.metadata_path,
        "outputs": {
            "production_png": row.production_path,
            "preview_png": row.preview_path,
            "metadata_json": row.metadata_path,
            "files": files,
            "named_files": {"production": row.production_path, "preview": row.preview_path, "metadata": row.metadata_path},
        },
    }


def _generate_exact(job, *, description: str, width: int, height: int, asset_type: str,
                    output_folder: Optional[str], transparent: bool, seed: Optional[int],
                    palette_limit: int = 24, view_hint: str = "",
                    lora_id: Optional[str] = None, lora_weight: Optional[float] = None) -> dict:
    out = resolve_output_folder(output_folder, project_name=asset_type + "s")
    loader = get_loader()
    base_id = config.RuntimeConfig.load().last_base_model_id
    prepare_local_base(loader, base_id, lora_id, lora_weight)
    prompt_text = f"{description}, {view_hint}" if view_hint else description
    _progress(job, 0.2, "generating source image")
    gen = GenParams(
        prompt=config.build_prompt(prompt_text, asset_type, solid_bg=transparent),
        negative_prompt=config.build_negative(None, asset_type),
        width=768, height=768, seed=seed, num_inference_steps=28,
    )
    base_img, used_seed, base_id = generate_source_image(
        job, get_pipelines(), gen, base_id, lora_id, lora_weight)
    prof = config.asset_profile(asset_type)
    _progress(job, 0.65, "running true-pixel pipeline")
    opts = PixelPipelineOptions(
        target_width=int(width), target_height=int(height), transparent=transparent,
        palette_limit=palette_limit, seed=used_seed, align=prof["align"],
        fit_margin=prof["margin"], use_segmentation=transparent and prof["isolated"],
        remove_floor_artifacts=transparent and asset_type in {"character", "character_turnaround", "character_animation", "prop", "item", "map_object"},
    )
    asset_id = _new_asset_id()
    result = run_true_pixel_pipeline(
        base_img, str(out), prompt_text, asset_id=asset_id, asset_type=asset_type,
        options=opts, seed=used_seed, model_profile_id=base_id)
    _record_asset(asset_id, asset_type, prompt_text, used_seed, width, height, result)
    return {
        "asset_id": asset_id,
        "production_png": result.production_path,
        "preview_png": result.preview_paths[0] if result.preview_paths else None,
        "preview_png_8x": result.preview_paths[1] if len(result.preview_paths) > 1 else None,
        "metadata_json": result.metadata_path,
        "seed": used_seed,
        "model_profile_id": base_id,
        "outputs": {
            "production_png": result.production_path,
            "preview_png": result.preview_paths[0] if result.preview_paths else None,
            "preview_png_8x": result.preview_paths[1] if len(result.preview_paths) > 1 else None,
            "metadata_json": result.metadata_path,
            "files": [p for p in [result.production_path, *result.preview_paths, result.metadata_path] if p],
            "named_files": {"production": result.production_path, "preview": result.preview_paths[0] if result.preview_paths else None, "metadata": result.metadata_path},
        },
    }


def create_1_direction_object(job, req) -> dict:
    _progress(job, 0.05, "creating one-direction object")
    res = _generate_exact(
        job, description=req.description, width=req.size, height=req.size,
        asset_type="prop", output_folder=req.output_folder, transparent=True,
        seed=req.seed, palette_limit=24, view_hint=f"{req.view} view, single isolated object",
        lora_id=req.lora_id, lora_weight=req.lora_weight)
    _progress(job, 1.0, "done")
    res["object_id"] = res["asset_id"]
    return res


def create_8_direction_object(job, req) -> dict:
    from ..pixel import spritesheet as ss
    out = resolve_output_folder(req.output_folder, project_name="objects")
    loader = get_loader()
    base_id = config.RuntimeConfig.load().last_base_model_id
    loader.load_base_model(base_id)
    apply_selected_lora(loader, req.lora_id, req.lora_weight)
    pipes = get_pipelines()
    size = int(req.size)
    prompt_base = f"{req.description}, {req.view}, single isolated object"
    negative = config.build_negative(None, "prop") + ", multiple objects, grid, sprite sheet"
    seed = int(req.seed) if req.seed is not None else None
    lock_palette = None
    ref_img = None
    direction_paths: dict[str, str] = {}
    frames = []
    for i, (name, hint) in enumerate(PIXELLAB_OBJECT_DIRECTIONS):
        _progress(job, 0.08 + 0.78 * i / 8.0, f"rotation {i+1}/8")
        gen = GenParams(
            prompt=config.build_prompt(f"{prompt_base}, {hint}", "prop"),
            negative_prompt=negative, width=768, height=768,
            seed=(seed + i * 17 if seed is not None else None), num_inference_steps=26)
        if ref_img is not None:
            try:
                src, used_seed = pipes.generate_with_ip_adapter(gen, [ref_img], weight=0.62)
            except Exception:
                src, used_seed = pipes.text_to_image(gen, base_model_id=base_id)
        else:
            src, used_seed = pipes.text_to_image(gen, base_model_id=base_id)
        opts = PixelPipelineOptions(
            target_width=size, target_height=size, transparent=True, palette_limit=24,
            seed=used_seed, align="center", fit_margin=0.10,
            lock_palette=lock_palette, use_segmentation=True,
            remove_floor_artifacts=True)
        aid = _new_asset_id()
        pp = run_true_pixel_pipeline(src, str(out), f"{prompt_base}, {hint}",
                                     asset_id=aid, asset_type="rotation_sheet",
                                     options=opts, seed=used_seed, model_profile_id=base_id)
        if ref_img is None:
            ref_img = pp.image
            lock_palette = pp.manifest.palette or None
        _record_asset(aid, "rotation_sheet", f"{prompt_base}, {hint}", used_seed, size, size, pp)
        direction_paths[name] = pp.production_path
        frames.append(pp.image)
    sheet = ss.build_sprite_sheet(frames, size, size, directions=8, frames_per_direction=1,
                                  layout="rows_by_direction")
    sheet_path = str(out / f"object_8dir_{_new_asset_id()}.png")
    sheet.save(sheet_path, format="PNG")
    meta_path = sheet_path.replace(".png", ".json")
    Path(meta_path).write_text(json.dumps({
        "description": req.description, "view": req.view, "directions": list(direction_paths),
        "direction_pngs": direction_paths, "sprite_sheet_png": sheet_path,
        "tile_width": size, "tile_height": size,
    }, indent=2), encoding="utf-8")
    _record_file_asset(Path(sheet_path).stem, "rotation_sheet", req.description, seed,
                       size, size * 8, sheet_path, meta_path)
    _progress(job, 1.0, "done")
    return {
        "object_id": Path(sheet_path).stem,
        "sprite_sheet_png": sheet_path,
        "metadata_json": meta_path,
        "directions": direction_paths,
        "outputs": {
            "sprite_sheet_png": sheet_path,
            "metadata_json": meta_path,
            "turnaround_pngs": direction_paths,
            "files": [sheet_path, meta_path, *direction_paths.values()],
            "named_files": {"sprite_sheet": sheet_path, "metadata": meta_path, **direction_paths},
        },
    }


def create_object_state(job, req) -> dict:
    row = _asset_record(req.object_id) if req.object_id else None
    uploaded = _image_from_data_url(getattr(req, "source_image", None))
    if uploaded is None:
        if not row or not row.production_path:
            raise ValueError(f"unknown object_id: {req.object_id}")
        img = Image.open(row.production_path).convert("RGBA")
        source_prompt = row.prompt or "object"
        source_width = row.width or img.width
        source_height = row.height or img.height
        source_id = req.object_id
    else:
        img = uploaded
        source_prompt = "uploaded object"
        source_width = img.width
        source_height = img.height
        source_id = req.object_id or "uploaded"

    out = resolve_output_folder(req.output_folder, project_name="objects")
    loader = get_loader()
    base_id = config.RuntimeConfig.load().last_base_model_id
    loader.load_base_model(base_id)
    apply_selected_lora(loader, req.lora_id, req.lora_weight)
    _progress(job, 0.25, "editing object state")
    prompt_text = f"{source_prompt}, {req.edit_description}"
    gen = GenParams(
        prompt=config.build_prompt(prompt_text, "prop"),
        negative_prompt=config.build_negative(None, "prop"), width=768, height=768,
        seed=req.seed, num_inference_steps=28)
    src, used_seed = get_pipelines().img2img(img, gen, strength=0.55, base_model_id=base_id)
    opts = PixelPipelineOptions(target_width=source_width, target_height=source_height,
                                transparent=True, palette_limit=24, seed=used_seed,
                                align="center", fit_margin=0.10, use_segmentation=True,
                                remove_floor_artifacts=True)
    aid = _new_asset_id()
    pp = run_true_pixel_pipeline(src, str(out), prompt_text,
                                 asset_id=aid, asset_type="prop", options=opts, seed=used_seed,
                                 model_profile_id=base_id)
    _record_asset(aid, "prop", prompt_text, used_seed, source_width, source_height, pp)
    _progress(job, 1.0, "done")
    return {
        "object_id": aid, "source_object_id": source_id, "group_id": source_id,
        "production_png": pp.production_path, "preview_png": pp.preview_paths[0] if pp.preview_paths else None,
        "metadata_json": pp.metadata_path, "seed": used_seed,
        "outputs": {"production_png": pp.production_path,
                    "preview_png": pp.preview_paths[0] if pp.preview_paths else None,
                    "metadata_json": pp.metadata_path,
                    "files": [p for p in [pp.production_path, *pp.preview_paths, pp.metadata_path] if p],
                    "named_files": {"production": pp.production_path,
                                    "preview": pp.preview_paths[0] if pp.preview_paths else None,
                                    "metadata": pp.metadata_path}},
    }


def create_map_object(job, req) -> dict:
    detail = f", {req.detail}" if req.detail else ""
    res = _generate_exact(
        job, description=req.description + detail, width=req.width, height=req.height,
        asset_type="map_object", output_folder=req.output_folder, transparent=True,
        seed=req.seed, palette_limit=24, view_hint=f"{req.view} map object",
        lora_id=req.lora_id, lora_weight=req.lora_weight)
    res["object_id"] = res["asset_id"]
    _progress(job, 1.0, "done")
    return res


def create_ui_asset(job, req) -> dict:
    parts = [req.description]
    if req.color_palette:
        parts.append(req.color_palette + " color palette")
    if req.elements:
        parts.append("UI elements: " + ", ".join(req.elements))
    res = _generate_exact(
        job, description=", ".join(parts), width=req.width, height=req.height,
        asset_type="ui_element", output_folder=req.output_folder, transparent=req.no_background,
        seed=req.seed, palette_limit=32, view_hint="pixel art game UI panel, clean reusable interface asset",
        lora_id=req.lora_id, lora_weight=req.lora_weight)
    res["ui_asset_id"] = res["asset_id"]
    _progress(job, 1.0, "done")
    return res


def create_topdown_tileset(job, req) -> dict:
    size = req.tile_size or {"width": 16, "height": 16}
    desc = f"{req.lower_description} to {req.upper_description}"
    if req.transition_description:
        desc += f", transition {req.transition_description}"
    treq = TilesetRequest(description=desc, tile_width=int(size.get("width", 16)),
                          tile_height=int(size.get("height", 16)), tileset_type="wang",
                          tile_count=16, output_folder=req.output_folder, seamless=True,
                          lora_id=req.lora_id, lora_weight=req.lora_weight)
    res = create_tileset(job, treq)
    res["tileset_id"] = Path(res["sprite_sheet_png"]).stem
    return res


def create_sidescroller_tileset(job, req) -> dict:
    size = req.tile_size or {"width": 16, "height": 16}
    desc = f"{req.lower_description} platform with {req.transition_description} surface"
    treq = TilesetRequest(description=desc, tile_width=int(size.get("width", 16)),
                          tile_height=int(size.get("height", 16)), tileset_type="sidescroller",
                          tile_count=8, output_folder=req.output_folder, seamless=False,
                          lora_id=req.lora_id, lora_weight=req.lora_weight)
    res = create_tileset(job, treq)
    res["tileset_id"] = Path(res["sprite_sheet_png"]).stem
    return res


def create_isometric_tile(job, req) -> dict:
    desc = f"{req.description}, {req.tile_shape}"
    treq = TilesetRequest(description=desc, tile_width=req.size, tile_height=req.size,
                          tileset_type="isometric", tile_count=1,
                          output_folder=req.output_folder, seamless=False,
                          lora_id=req.lora_id, lora_weight=req.lora_weight)
    res = create_tileset(job, treq)
    res["tile_id"] = Path(res["sprite_sheet_png"]).stem
    return res


def create_tiles_pro(job, req) -> dict:
    kind = "isometric" if req.tile_type == "isometric" else "top_down"
    if req.tile_type in ("hex", "hex_pointy", "octagon"):
        desc = f"{req.description}, {req.tile_type} shaped tiles, {req.tile_view}"
    else:
        desc = f"{req.description}, {req.tile_view}"
    treq = TilesetRequest(description=desc, tile_width=req.tile_size, tile_height=req.tile_size,
                          tileset_type=kind, tile_count=8,
                          output_folder=req.output_folder, seamless=True,
                          lora_id=req.lora_id, lora_weight=req.lora_weight)
    res = create_tileset(job, treq)
    res["tile_id"] = Path(res["sprite_sheet_png"]).stem
    return res

def export_pack(job, asset_ids: list[str], engine: str, output_folder: Optional[str]) -> dict:
    from .security import resolve_output_folder as _rof
    from ..pixel.exporters import export_zip
    out = _rof(output_folder, project_name="exports")
    files: list[tuple[str, str]] = []
    _progress(job, 0.1, "gathering assets")
    for aid in asset_ids:
        # assets are stored by their production path; find png+json siblings
        for p in out.parent.rglob(f"{aid}.png"):
            files.append((p.name, str(p)))
            meta = p.with_suffix(".json")
            if meta.exists():
                files.append((meta.name, str(meta)))
    zip_path = str(out / f"pack_{_new_asset_id()}.zip")
    export_zip(files, zip_path)
    _progress(job, 1.0, "done")
    return {"zip_path": zip_path,
            "outputs": {"zip_path": zip_path,
                        "files": [zip_path],
                        "named_files": {"zip": zip_path}}}





