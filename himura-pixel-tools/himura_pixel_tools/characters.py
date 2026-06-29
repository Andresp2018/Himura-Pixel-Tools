"""Character consistency system (spec.character_consistency_system).

Implements the generation_strategy:
  1. create CharacterProfile + StyleProfile
  2. generate a front-facing base character with prompt/palette/seed
  3. extract & store reference lock (palette, silhouette hash, seed pack)
  4. generate other directions with IP-Adapter reference + pose/edge ControlNet
  5. post-process to exact size + align pivot
  6. consistency scoring + regeneration loop
  7. animation keyframes from PoseTemplate, aligned, exported

Every character gets a persistent identity across prompts/directions/frames.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from . import config, db
from .pixel import cleanup, exporters, pixelate, validate
from .runtime.pipelines import GenParams, Pipelines
from .schemas.characters import (
    CharacterProfile, ConsistencyReport, ReferenceLock, StyleProfile,
)
from .schemas.jobs import JobOutputs

# â”€â”€ direction metadata â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DIRS_4 = ["front", "left", "right", "back"]
DIRS_8 = ["front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right"]
DIR_PROMPT = {
    "front": "front view, facing the camera, face visible, looking down toward viewer",
    "back": "back view, seen from behind, back of the head and back of the body, face hidden",
    "left": "left side profile view, facing left, side silhouette",
    "right": "right side profile view, facing right, side silhouette",
    "front-left": "three-quarter front-left view, facing front-left",
    "front-right": "three-quarter front-right view, facing front-right",
    "back-left": "three-quarter back-left view, facing away to the left",
    "back-right": "three-quarter back-right view, facing away to the right",
}
MIRROR_DIRECTIONS = {
    "right": "left",
    "front-right": "front-left",
    "back-right": "back-left",
}

DIR_NEGATIVE = {
    "back": "face, eyes, front view, looking at camera",
    "left": "front view, right-facing, looking at camera",
    "right": "front view, left-facing, looking at camera",
    "front-left": "back view, right-facing",
    "front-right": "back view, left-facing",
    "back-left": "front view, face visible, right-facing",
    "back-right": "front view, face visible, left-facing",
}

# default animation templates (spec.animation_architecture.default_animation_templates)
ANIM_TEMPLATES = {
    "idle":  {"frames": 4, "fps": 6,  "note": "subtle breathing/bob"},
    "walk":  {"frames": 6, "fps": 10, "note": "clear foot contact frames"},
    "run":   {"frames": 6, "fps": 12, "note": "larger limb movement, fixed canvas"},
    "attack":{"frames": 6, "fps": 12, "note": "weapon arc optional overlay"},
    "hurt":  {"frames": 3, "fps": 8,  "note": "knockback in metadata"},
    "death": {"frames": 8, "fps": 8,  "note": "often one direction"},
    "cast":  {"frames": 8, "fps": 10, "note": "effects layer separate"},
}


# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def silhouette_hash(image: Image.Image) -> str:
    arr = np.array(image.convert("RGBA"))
    alpha = (arr[..., 3] > 0).astype(np.uint8)
    # downsample silhouette for a stable hash
    h, w = alpha.shape
    block = 8
    sh, sw = max(1, h // block), max(1, w // block)
    small = np.zeros((block, block), dtype=np.uint8)
    for i in range(block):
        for j in range(block):
            y0, y1 = i * sh, (i + 1) * sh
            x0, x1 = j * sw, (j + 1) * sw
            small[i, j] = 1 if alpha[y0:y1, x0:x1].mean() > 0.5 else 0
    return hashlib.sha256(small.tobytes()).hexdigest()[:16]


def palette_delta(a: list[list[int]], b: list[list[int]]) -> float:
    """Average nearest-neighbor RGB distance between two palettes (rough CIEDE2000 proxy)."""
    if not a or not b:
        return 0.0
    pa = np.array(a, dtype=np.float32)
    pb = np.array(b, dtype=np.float32)
    total, n = 0.0, 0
    for c in pa:
        d = np.sqrt(((pb - c) ** 2).sum(axis=1)).min()
        total += float(d)
        n += 1
    return total / max(1, n)


def seed_pack(base_seed: int, n: int) -> list[int]:
    """Deterministic per-frame seeds derived from the base seed."""
    rng = np.random.default_rng(base_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]


# â”€â”€ CharacterSystem â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class CharacterSystem:
    """Owns CharacterProfile/ReferenceLock lifecycle + consistency scoring."""

    def __init__(self, pipelines: Pipelines):
        self.pipelines = pipelines

    # â”€â”€ CRUD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def create_profile(self, name: str, description: str, width: int = 64, height: int = 64,
                       directions: int = 4, style_profile_id: Optional[str] = None,
                       base_seed: Optional[int] = None) -> CharacterProfile:
        cid = _new_id("char")
        profile = CharacterProfile(
            character_id=cid,
            name=name,
            description=description,
            base_seed=int(base_seed) if base_seed is not None else abs(hash(name)) % (2**31),
            style_profile_id=style_profile_id or "style_default_rpg_64",
            canonical_size={"width": width, "height": height},
            directions_supported=[directions] if directions in (4, 8) else [4, 8],
            reference_assets=[],
            reference_lock=None,
            created_at=_now(),
            updated_at=_now(),
        )
        self._save_profile(profile)
        return profile

    def get_profile(self, character_id: str) -> Optional[CharacterProfile]:
        with db.get_session() as s:
            row = s.exec(db.select(db.CharacterProfileRow).where(
                db.CharacterProfileRow.character_profile_id == character_id)).first()
            if not row:
                return None
            return CharacterProfile(
                character_id=row.character_profile_id,
                name=row.name, description=row.description,
                base_seed=row.base_seed, style_profile_id=row.style_profile_id,
                palette_id=row.palette_id,
                canonical_size=json.loads(row.canonical_size_json),
                directions_supported=json.loads(row.directions_supported),
                reference_lock=ReferenceLock(**json.loads(row.reference_lock_json))
                                if row.reference_lock_json and row.reference_lock_json != "{}" else None,
                approved_assets=json.loads(row.approved_assets_json),
                created_at=row.created_at, updated_at=row.updated_at,
            )

    def list_profiles(self) -> list[CharacterProfile]:
        with db.get_session() as s:
            rows = s.exec(db.select(db.CharacterProfileRow)).all()
            return [self.get_profile(r.character_profile_id) for r in rows]

    def _save_profile(self, profile: CharacterProfile) -> None:
        with db.get_session() as s:
            existing = s.exec(db.select(db.CharacterProfileRow).where(
                db.CharacterProfileRow.character_profile_id == profile.character_id)).first()
            payload = dict(
                character_profile_id=profile.character_id,
                name=profile.name, description=profile.description,
                base_seed=profile.base_seed, style_profile_id=profile.style_profile_id,
                palette_id=profile.palette_id,
                canonical_size_json=json.dumps(profile.canonical_size),
                directions_supported=json.dumps(profile.directions_supported),
                reference_lock_json=profile.reference_lock.model_dump_json()
                                     if profile.reference_lock else "{}",
                approved_assets_json=json.dumps(profile.approved_assets),
                updated_at=_now(),
            )
            if existing:
                for k, v in payload.items():
                    setattr(existing, k, v)
                s.add(existing)
            else:
                s.add(db.CharacterProfileRow(**payload))
            s.commit()

    def approve_reference(self, character_id: str, asset_id: str) -> None:
        p = self.get_profile(character_id)
        if p and asset_id not in p.approved_assets:
            p.approved_assets.append(asset_id)
            self._save_profile(p)

    # â”€â”€ canonical reference generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _ref_init_image(self, profile: CharacterProfile) -> Optional[Image.Image]:
        """Load the canonical reference sprite to use as an img2img anchor."""
        lock = profile.reference_lock
        if lock and lock.reference_image_paths:
            for p in lock.reference_image_paths:
                if Path(p).exists():
                    try:
                        return Image.open(p).convert("RGBA")
                    except Exception:
                        continue
        return None

    def _save_direct_sprite(self, image: Image.Image, output_folder: str, prompt: str,
                            asset_type: str, seed: int) -> str:
        out = Path(output_folder)
        out.mkdir(parents=True, exist_ok=True)
        aid = _new_id("asset")
        img = image.convert("RGBA")
        prod = str(out / f"{aid}.png")
        prev = str(out / f"{aid}_preview_8x.png")
        meta = str(out / f"{aid}.json")
        exporters.save_production_png(img, prod)
        exporters.save_preview_png(img, prev, scale=8)
        manifest = exporters.build_manifest(aid, asset_type, prompt, img, seed=seed)
        Path(meta).write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
        return prod
    def _reference_conditioned(self, prompt: str, negative: str, seed: int,
                               init_img: Optional[Image.Image], strength: float,
                               steps: int = 28, prefer_ip: bool = False,
                               ip_weight: float = 0.7) -> tuple[Image.Image, int]:
        """Generate guided by the canonical reference for identity stability.

        Conditioning preference, best identity first:
          1. IP-Adapter (``prefer_ip``) — keeps the character's identity while the
             text prompt freely changes pose/facing. This is what makes a real
             turnaround/animation possible (it can rotate without losing the
             design). Auto-engages once the IP-Adapter model is installed.
          2. img2img off the reference sprite — reliable, no extra model, but
             can't rotate much (good for the front view).
          3. plain text-to-image — last resort.
        A *fixed* seed keeps colors/shape stable across directions and frames.
        """
        gen = GenParams(prompt=prompt, negative_prompt=negative, width=768, height=768,
                        seed=seed, num_inference_steps=steps)
        if prefer_ip and init_img is not None:
            try:
                return self.pipelines.generate_with_ip_adapter(
                    gen, [init_img], weight=ip_weight)
            except Exception:
                pass
        if init_img is not None:
            try:
                return self.pipelines.img2img(init_img, gen, strength=strength)
            except Exception:
                pass
        return self.pipelines.text_to_image(gen)

    def _openpose_id(self) -> Optional[str]:
        """Return a valid pose-capable ControlNet for the active base."""
        try:
            from . import config as _cfg
            base_id = (_cfg.RuntimeConfig.load().last_base_model_id or "").lower()
            want_flux = "flux" in base_id
            flux_match = sdxl_match = None
            for m in self.pipelines.loader.scan_model_store():
                if m.type != "controlnet":
                    continue
                report = self.pipelines.loader.validate_model_files(m.model_id)
                if not report.valid:
                    continue
                blob = (m.model_id + " " + (m.local_path or "")).lower()
                is_flux_cn = "flux" in blob or "flux_optional_future" in (m.base_compatibility or [])
                if is_flux_cn and ("union" in blob or "pose" in blob):
                    flux_match = flux_match or m.model_id
                elif not is_flux_cn and "openpose" in blob:
                    sdxl_match = sdxl_match or m.model_id
            return flux_match if want_flux else sdxl_match
        except Exception:
            return None

    def _pose_or_reference(self, prompt: str, negative: str, seed: int,
                           init_img: Optional[Image.Image], strength: float,
                           direction: Optional[str] = None, animation: Optional[str] = None,
                           frame: int = 0, n_frames: int = 1, steps: int = 26,
                           ip_weight: float = 0.75) -> tuple[Image.Image, int]:
        """Best available conditioned generation:
          1. ControlNet-OpenPose skeleton (facing/animation) + IP-Adapter identity
          2. IP-Adapter only  3. img2img  4. text-to-image
        The skeleton makes the character actually rotate / move; IP-Adapter keeps
        it on-model. Everything degrades gracefully if a model isn't installed."""
        cn_id = self._openpose_id() if init_img is not None else None
        if cn_id is not None:
            try:
                from .pixel import skeletons
                if animation:
                    pose_img = skeletons.animation_skeleton(animation, direction or "front",
                                                            frame, n_frames)
                else:
                    pose_img = skeletons.direction_skeleton(direction or "front")
                gen = GenParams(prompt=prompt, negative_prompt=negative, width=768, height=768,
                                seed=seed, num_inference_steps=steps)
                return self.pipelines.generate_with_pose(
                    gen, pose_img, cn_id, ref_images=[init_img], ip_weight=ip_weight)
            except Exception:
                pass
        return self._reference_conditioned(prompt, negative, seed, init_img, strength,
                                           steps=steps, prefer_ip=True, ip_weight=ip_weight)

    def generate_canonical_reference(self, profile: CharacterProfile, output_folder: str,
                                     prompt: Optional[str] = None) -> dict:
        """Generate the front-facing canonical reference + build ReferenceLock."""
        from .pixel.pipeline import PixelPipelineOptions, run_true_pixel_pipeline

        w = int(profile.canonical_size["width"])
        h = int(profile.canonical_size["height"])
        desc = prompt or profile.description
        full_prompt = config.build_prompt(desc, "character")
        negative = config.build_negative(None, "character")

        # generate at model-friendly size
        gen = GenParams(
            prompt=full_prompt, negative_prompt=negative, width=768, height=768,
            seed=profile.base_seed, num_inference_steps=30,
        )
        base_img, used_seed = self.pipelines.text_to_image(gen)

        # run true-pixel pipeline to exact size — feet-aligned, full sprite in frame
        opts = PixelPipelineOptions(target_width=w, target_height=h, transparent=True,
                                    palette_limit=24, seed=profile.base_seed,
                                    align="bottom", fit_margin=0.08,
                                    remove_floor_artifacts=True)
        result = run_true_pixel_pipeline(
            base_img, output_folder, full_prompt,
            asset_id=_new_id("asset"), asset_type="character", options=opts,
            seed=profile.base_seed,
        )

        # build the reference lock from the approved production image
        prod = result.image
        palette = pixelate.extract_palette(prod, max_colors=24)
        sh = silhouette_hash(prod)
        lock = ReferenceLock(
            character_id=profile.character_id,
            reference_image_paths=[result.production_path],
            palette=palette,
            silhouette_hash=sh,
            prompt_lock=profile.description,
            seed_pack=seed_pack(profile.base_seed, 64),
        )
        profile.reference_lock = lock
        profile.reference_assets = list(set(profile.reference_assets + [result.production_path]))
        self._save_profile(profile)

        return {
            "asset_id": result.manifest.asset_id,
            "production_png": result.production_path,
            "preview_png": result.preview_paths[0] if result.preview_paths else None,
            "preview_png_8x": result.preview_paths[1] if len(result.preview_paths) > 1 else None,
            "metadata_json": result.metadata_path,
            "seed": used_seed,
            "silhouette_hash": sh,
        }

    # â”€â”€ turnaround (4/8 directions) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate_turnaround(self, profile: CharacterProfile, directions: int,
                            output_folder: str, width: Optional[int] = None,
                            height: Optional[int] = None) -> dict:
        """Generate 4/8 directional sprites locked to the canonical identity.

        Every direction is rendered with img2img off the canonical reference
        (same fixed seed) and remapped to the reference palette, so the
        character keeps the same colors, size and silhouette while only the
        facing changes.
        """
        from .pixel.pipeline import PixelPipelineOptions, run_true_pixel_pipeline

        w = int(width or profile.canonical_size["width"])
        h = int(height or profile.canonical_size["height"])
        dirs = DIRS_4 if directions == 4 else DIRS_8
        outputs: dict[str, str] = {}
        output_images: dict[str, Image.Image] = {}
        ref_lock = profile.reference_lock
        lock_palette = ref_lock.palette if ref_lock else None
        init_img = self._ref_init_image(profile)
        seed = int(profile.base_seed)
        negative = config.build_negative(None, "character_turnaround")

        for d in dirs:
            direction_text = DIR_PROMPT.get(d, d)
            base_prompt = f"{direction_text}, {profile.description}"
            prompt = config.build_prompt(base_prompt, "character_turnaround")
            dir_negative = negative + (", " + DIR_NEGATIVE.get(d, "") if DIR_NEGATIVE.get(d) else "")
            dir_seed = seed if d == "front" else seed + (dirs.index(d) + 1) * 37
            mirror_source = MIRROR_DIRECTIONS.get(d)
            if mirror_source and mirror_source in output_images:
                mirrored = ImageOps.mirror(output_images[mirror_source])
                outputs[d] = self._save_direct_sprite(
                    mirrored, output_folder, prompt, "character_turnaround", dir_seed)
                output_images[d] = mirrored
                continue
            if d == "front" and init_img is not None:
                direct = init_img.convert("RGBA").resize((w, h), Image.NEAREST)
                outputs[d] = self._save_direct_sprite(
                    direct, output_folder, prompt, "character_turnaround", dir_seed)
                output_images[d] = direct
                continue
            # The front view should match the canonical sprite, so anchor it to
            # the reference with light img2img. Side/back views must actually
            # ROTATE the character — img2img off a front sprite refuses to turn
            # (that was the "all directions identical" bug), so we generate those
            # from text with an explicit facing prompt and a fixed seed, and lock
            # the palette so colors still match the reference.
            if d == "front":
                # front ≈ canonical pose: light img2img keeps it identical
                base_img, _ = self._reference_conditioned(
                    prompt, dir_negative, dir_seed, init_img, strength=0.4)
            else:
                # OpenPose skeleton facing this direction + IP-Adapter identity
                # rotates the character properly; falls back to IP/img2img/text.
                base_img, _ = self._pose_or_reference(
                    prompt, dir_negative, dir_seed, init_img, strength=0.85, direction=d, ip_weight=0.65)

            opts = PixelPipelineOptions(target_width=w, target_height=h, transparent=True,
                                        palette_limit=24, seed=dir_seed,
                                        align="bottom", fit_margin=0.08,
                                        lock_palette=lock_palette,
                                        remove_floor_artifacts=True)
            result = run_true_pixel_pipeline(
                base_img, output_folder, prompt,
                asset_id=_new_id("asset"), asset_type="character_turnaround",
                options=opts, seed=dir_seed,
            )
            outputs[d] = result.production_path
            output_images[d] = result.image

        return {"directions": outputs, "count": len(outputs)}

    # â”€â”€ animation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate_animation(self, profile: CharacterProfile, animation: str,
                           directions: int, frames_per_direction: Optional[int],
                           output_folder: str) -> dict:
        """Generate aligned sprite animation frames + sprite sheet."""
        from .pixel.pipeline import PixelPipelineOptions, run_true_pixel_pipeline
        from .pixel import spritesheet as ss
        from .pixel import exporters, masks, cleanup

        tmpl = ANIM_TEMPLATES.get(animation, ANIM_TEMPLATES["idle"])
        n_frames = int(frames_per_direction or tmpl["frames"])
        fps = tmpl["fps"]
        dur_ms = int(1000 / fps)
        dirs = DIRS_4 if directions == 4 else DIRS_8
        w = int(profile.canonical_size["width"])
        h = int(profile.canonical_size["height"])
        ref_lock = profile.reference_lock
        lock_palette = ref_lock.palette if ref_lock else None
        init_img = self._ref_init_image(profile)
        seed = int(profile.base_seed)
        negative = config.build_negative(None, "character_animation")

        # how much each frame moves relative to the locked pose. Larger actions
        # get more motion; idle barely moves so it reads as a calm loop.
        motion = {"idle": 0.30, "hurt": 0.45, "death": 0.55, "cast": 0.40}.get(animation, 0.5)

        all_frames: list[Image.Image] = []
        per_direction: dict[str, list[Image.Image]] = {}
        for di, d in enumerate(dirs):
            mirror_source = MIRROR_DIRECTIONS.get(d)
            if mirror_source and mirror_source in per_direction:
                dir_frames = [ImageOps.mirror(frame) for frame in per_direction[mirror_source]]
                all_frames.extend(dir_frames)
                per_direction[d] = dir_frames
                continue
            # 1) generate raw, background-removed hi-res frames for this direction
            raw_frames: list[Image.Image] = []
            for fr in range(n_frames):
                phase = f"{animation} pose {fr + 1} of {n_frames}, mid-motion"
                direction_text = DIR_PROMPT.get(d, d)
                base_prompt = f"{direction_text}, {phase}, {profile.description}"
                prompt = config.build_prompt(base_prompt, "character_animation")
                frame_negative = negative + (", " + DIR_NEGATIVE.get(d, "") if DIR_NEGATIVE.get(d) else "")
                # img2img off the reference keeps identity; a per-frame seed and a
                # higher strength move the limbs so frames actually differ.
                frame_seed = seed + di * 100 + fr * 13
                # skeleton keyframe for this animation/direction/frame drives the
                # pose; IP-Adapter keeps identity. Falls back to img2img motion.
                base_img, _ = self._pose_or_reference(
                    prompt, frame_negative, frame_seed, init_img, strength=motion,
                    direction=d, animation=animation, frame=fr, n_frames=n_frames,
                    steps=22, ip_weight=0.85)
                # Flat flood-fill preserves capes/weapons/limbs better for moving frames;
                # BiRefNet can over-mask small animated parts after pose changes.
                raw_frames.append(masks.remove_background(base_img, use_model=False))

            # 2) compose the whole direction with ONE shared scale + pivot so the
            #    character stays planted and no limb is rescaled or clipped.
            composed = cleanup.compose_group_on_canvas(
                raw_frames, w * 8, h * 8, margin_frac=0.08, align="bottom")

            # 3) pixelate each pre-composed frame to the exact tile size
            dir_frames: list[Image.Image] = []
            for fr, canvas in enumerate(composed):
                frame_seed = seed + di * 100 + fr * 13
                opts = PixelPipelineOptions(target_width=w, target_height=h, transparent=True,
                                            palette_limit=24, seed=frame_seed,
                                            lock_palette=lock_palette,
                                            skip_background=True, skip_compose=True,
                                            remove_floor_artifacts=False, cleanup_orphans=False,
                                            generate_preview_scale=[])
                result = run_true_pixel_pipeline(
                    canvas, output_folder, f"{animation} {d} frame {fr+1}",
                    asset_id=_new_id("asset"), asset_type="character_animation",
                    options=opts, seed=frame_seed)
                all_frames.append(result.image)
                dir_frames.append(result.image)
            per_direction[d] = dir_frames

        # build sprite sheet + metadata
        out = Path(output_folder)
        out.mkdir(parents=True, exist_ok=True)
        sheet = ss.build_sprite_sheet(all_frames, w, h, directions, n_frames,
                                      layout="rows_by_direction")
        sheet_path = str(out / f"{profile.character_id}_{animation}_sheet.png")
        sheet.save(sheet_path, format="PNG")

        durations = [dur_ms] * len(all_frames)
        # Previews loop a SINGLE direction (front) so a "4-direction idle" reads
        # as a clean 4-frame loop instead of a 16-frame walk through every facing.
        preview_frames = per_direction.get(dirs[0], all_frames)
        gif_path = str(out / f"{profile.character_id}_{animation}.gif")
        webp_path = str(out / f"{profile.character_id}_{animation}.webp")
        exporters.save_gif(preview_frames, gif_path, duration_ms=dur_ms)
        exporters.save_webp(preview_frames, webp_path, duration_ms=dur_ms)

        meta_path = str(out / f"{profile.character_id}_{animation}_metadata.json")
        meta = ss.aseprite_metadata(animation, directions, w, h, n_frames, dur_ms)
        meta["sprite_sheet_png"] = sheet_path
        meta["gif_preview"] = gif_path
        meta["webp_preview"] = webp_path
        meta["direction_names"] = dirs
        Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return {
            "sprite_sheet_png": sheet_path,
            "metadata_json": meta_path,
            "gif_preview": gif_path,
            "webp_preview": webp_path,
            "frame_count": len(all_frames),
            "directions": directions,
            "frames_per_direction": n_frames,
            "frame_durations_ms": durations,
        }

    # â”€â”€ portrait â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate_portrait(self, profile: CharacterProfile, output_folder: str,
                          width: int = 128, height: int = 128, palette_limit: int = 32,
                          expression: Optional[str] = None, transparent: bool = True,
                          seed: Optional[int] = None) -> dict:
        """Generate a face/bust portrait of the character (dialogue/UI art).

        Conditioned on the canonical reference so the portrait matches the
        sprite's colors and design, but framed as a close-up bust at higher
        resolution and color count than the in-world sprite.
        """
        from .pixel.pipeline import PixelPipelineOptions, run_true_pixel_pipeline

        expr = (expression or "neutral").strip()
        # A portrait is a NEW composition (close-up bust), not the body sprite
        # recolored — so we generate it from text with explicit portrait framing
        # and only borrow the character's colors via the locked palette. img2img
        # off the tiny full-body sprite was forcing a full-body result, which is
        # why the portrait "just changed the face".
        base_prompt = (
            f"detailed character portrait of {profile.description}, "
            f"{expr} expression, head and shoulders bust, face close-up, "
            f"facing viewer, looking at the camera, expressive detailed face, "
            f"RPG dialogue portrait, character splash art")
        prompt = config.build_prompt(base_prompt, "character_portrait", solid_bg=transparent)
        negative = (config.build_negative(None, "character_portrait") +
                    ", full body, tiny, distant, full figure, legs, feet, "
                    "walking, sprite sheet")
        use_seed = int(seed) if seed is not None else int(profile.base_seed)

        # try IP-Adapter for identity if installed; otherwise plain text-to-image.
        gen = GenParams(prompt=prompt, negative_prompt=negative, width=768, height=768,
                        seed=use_seed, num_inference_steps=34)
        ref = self._ref_init_image(profile)
        if ref is not None:
            try:
                base_img, used_seed = self.pipelines.generate_with_ip_adapter(
                    gen, [ref], weight=0.55)
            except Exception:
                base_img, used_seed = self.pipelines.text_to_image(gen)
        else:
            base_img, used_seed = self.pipelines.text_to_image(gen)

        # keep colors on-model, but allow the portrait its own richer shading.
        lock_palette = None
        opts = PixelPipelineOptions(target_width=int(width), target_height=int(height),
                                    transparent=transparent, palette_limit=int(palette_limit),
                                    seed=use_seed, align="center", fit_margin=0.04,
                                    lock_palette=lock_palette)
        result = run_true_pixel_pipeline(
            base_img, output_folder, prompt,
            asset_id=_new_id("asset"), asset_type="character_portrait",
            options=opts, seed=use_seed,
        )
        return {
            "character_profile_id": profile.character_id,
            "production_png": result.production_path,
            "preview_png": result.preview_paths[0] if result.preview_paths else None,
            "preview_png_8x": result.preview_paths[1] if len(result.preview_paths) > 1 else None,
            "metadata_json": result.metadata_path,
            "seed": used_seed,
        }

    # â”€â”€ consistency scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def score_consistency(self, profile: CharacterProfile, candidate: Image.Image,
                          asset_id: str) -> ConsistencyReport:
        """Compare a candidate sprite against the canonical ReferenceLock."""
        ref_lock = profile.reference_lock
        notes: list[str] = []
        if not ref_lock:
            return ConsistencyReport(character_id=profile.character_id, asset_id=asset_id,
                                     notes=["no reference lock â€” skipping scoring"])
        # palette delta
        cand_pal = pixelate.extract_palette(candidate, max_colors=24)
        delta = palette_delta(ref_lock.palette, cand_pal)
        palette_ok = delta <= 50.0  # rough threshold (configurable)
        if not palette_ok:
            notes.append(f"palette delta {delta:.1f} exceeds threshold")

        # silhouette drift
        cand_sh = silhouette_hash(candidate)
        bbox_ok = cand_sh == ref_lock.silhouette_hash
        # bbox drift (loose)
        bp = cleanup.bbox_and_pivot(candidate)
        bbox = bp.get("bbox")
        bbox_drift = 0
        if bbox:
            cw, ch = profile.canonical_size["width"], profile.canonical_size["height"]
            bbox_drift = abs(bbox["w"] - cw) + abs(bbox["h"] - ch)
        bbox_ok = bbox_drift <= 4

        alpha_ok = validate.alpha_values_ok(candidate)
        overall = palette_ok and bbox_ok and alpha_ok
        return ConsistencyReport(
            character_id=profile.character_id, asset_id=asset_id,
            palette_delta=delta, palette_ok=palette_ok,
            bbox_drift_px=bbox_drift, bbox_ok=bbox_ok,
            alpha_ok=alpha_ok, overall_ok=overall, notes=notes,
        )






