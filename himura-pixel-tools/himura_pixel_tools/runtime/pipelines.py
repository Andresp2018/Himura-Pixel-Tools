"""Runtime pipelines: text-to-image, img2img, inpaint, ControlNet, IP-Adapter.

Wraps the Diffusers pipeline calls behind a small, stable API the control layer
uses. All generation goes through the ModelLoader so VRAM policy is enforced.

This module never touches ComfyUI. Diffusers/PyTorch are used as libraries.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .. import config
from .model_loader import ModelLoader, RuntimeHandle, _import_diffusers, _import_torch


@dataclass
class GenParams:
    prompt: str
    negative_prompt: str = config.DEFAULT_NEGATIVE_PROMPT
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    quality_preset: str = "quality"   # quality | fast


def _make_generator(device: str, seed: Optional[int]) -> tuple[object, int]:
    torch = _import_torch()
    gen_device = "cuda" if device == "cuda" else "cpu"
    g = torch.Generator(device=gen_device)
    if seed is None or int(seed) == -1:
        seed = random.randint(0, 2**32 - 1)
    g.manual_seed(int(seed))
    return g, int(seed)


class Pipelines:
    """High-level generation API backed by ModelLoader."""

    def __init__(self, loader: ModelLoader):
        self.loader = loader

    # â”€â”€ internal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _device(self) -> str:
        torch = _import_torch()
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_handle(self, base_id: Optional[str] = None, precision: str = "fp16") -> RuntimeHandle:
        if self.loader.handle and self.loader.handle.pipeline is not None:
            return self.loader.handle
        return self.loader.load_base_model(
            base_id or self.loader._cfg.last_base_model_id, precision=precision)

    def _fast_params(self, params: GenParams) -> dict:
        """Apply LCM/Lightning-style fast preset: fewer steps, low guidance."""
        if params.quality_preset == "fast":
            return {
                "num_inference_steps": min(8, params.num_inference_steps or 8),
                "guidance_scale": 1.0,
            }
        return {}

    # â”€â”€ text-to-image â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def text_to_image(self, params: GenParams,
                      base_model_id: Optional[str] = None) -> tuple[Image.Image, int]:
        handle = self._ensure_handle(base_model_id)
        device = self._device()
        gen, seed = _make_generator(device, params.seed)
        fast = self._fast_params(params)
        steps = fast.get("num_inference_steps", params.num_inference_steps)
        guidance = fast.get("guidance_scale", params.guidance_scale)

        is_flux = handle.base_kind == "flux_optional_future"
        kwargs = dict(
            prompt=params.prompt,
            width=params.width, height=params.height,
            num_inference_steps=steps,
            guidance_scale=guidance if not is_flux else (params.guidance_scale or 3.5),
            generator=gen,
        )
        if not is_flux:
            kwargs["negative_prompt"] = params.negative_prompt

        with self.loader.lock:
            result = handle.pipeline(**kwargs)
        img = result.images[0]
        return img, seed

    # â”€â”€ img2img â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _img2img_pipe(self, handle: RuntimeHandle):
        """Build (once) an img2img pipeline that *shares* the loaded base model's
        weights, so reference-conditioned generation costs no extra VRAM.

        A plain ``StableDiffusionXLPipeline`` has no ``.img2img`` method (the old
        code called one that never existed), which is why every "consistent"
        path silently fell back to independent text-to-image. ``from_pipe``
        gives us the matching Img2Img pipeline class reusing the same modules.
        """
        diffusers = _import_diffusers()
        cache = self.loader._cache
        if cache.get("img2img_pipe") is not None and cache.get("img2img_src") == id(handle.pipeline):
            return cache["img2img_pipe"]
        pipe = diffusers.AutoPipelineForImage2Image.from_pipe(handle.pipeline)
        cache["img2img_pipe"] = pipe
        cache["img2img_src"] = id(handle.pipeline)
        return pipe

    def img2img(self, init_image: Image.Image, params: GenParams, strength: float = 0.6,
                base_model_id: Optional[str] = None) -> tuple[Image.Image, int]:
        handle = self._ensure_handle(base_model_id)
        device = self._device()
        init = init_image.convert("RGB").resize((params.width, params.height), Image.LANCZOS)
        gen, seed = _make_generator(device, params.seed)
        fast = self._fast_params(params)
        steps = fast.get("num_inference_steps", params.num_inference_steps)
        guidance = fast.get("guidance_scale", params.guidance_scale)
        with self.loader.lock:
            pipe = self._img2img_pipe(handle)
            result = pipe(
                prompt=params.prompt, image=init, strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance, negative_prompt=params.negative_prompt,
                generator=gen,
            )
        return result.images[0], seed

    # â”€â”€ inpaint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def inpaint(self, image: Image.Image, mask: Image.Image, params: GenParams,
                strength: float = 0.95, base_model_id: Optional[str] = None) -> tuple[Image.Image, int]:
        handle = self._ensure_handle(base_model_id)
        device = self._device()
        init = image.convert("RGBA").resize((params.width, params.height), Image.LANCZOS)
        m = mask.convert("L").resize((params.width, params.height), Image.NEAREST)
        gen, seed = _make_generator(device, params.seed)
        with self.loader.lock:
            inpaint_pipe = self.loader.build_pipeline("inpaint", {"base_model_id": base_model_id}).pipeline
            if inpaint_pipe is None:
                inpaint_pipe = handle.pipeline
            result = inpaint_pipe(
                prompt=params.prompt, image=init.convert("RGB"), mask_image=m,
                strength=strength, num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale, negative_prompt=params.negative_prompt,
                generator=gen,
            )
        out = result.images[0].convert("RGBA")
        # keep original pixels outside the mask
        base = init.copy()
        base.alpha_composite(out, (0, 0))
        # restore alpha from original where mask is black
        arr = np.array(base)
        marr = np.array(m)
        arr[..., 3] = np.where(marr > 128, 255, arr[..., 3])
        return Image.fromarray(arr, mode="RGBA"), seed

    # â”€â”€ ControlNet-conditioned generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate_with_controlnet(
        self, params: GenParams, control_image: Image.Image,
        controlnet_model_id: str, controlnet_scale: float = 0.8,
        base_model_id: Optional[str] = None,
    ) -> tuple[Image.Image, int]:
        diffusers = _import_diffusers()
        torch = _import_torch()
        handle = self._ensure_handle(base_model_id)
        cn = self.loader.attach_controlnet(controlnet_model_id, controlnet_scale)
        if cn is None:
            return self.text_to_image(params, base_model_id)
        device = self._device()
        gen, seed = _make_generator(device, params.seed)
        ci = control_image.convert("RGB").resize((params.width, params.height), Image.LANCZOS)

        # FLUX uses its own ControlNet pipeline + call signature (control_image,
        # no negative prompt). Build it from the resident pipe so VRAM is shared.
        if (handle.base_kind or "").startswith("flux"):
            FCN = getattr(diffusers, "FluxControlNetPipeline", None)
            if FCN is None:
                return self.text_to_image(params, base_model_id)
            try:
                pipe = FCN.from_pipe(handle.pipeline, controlnet=cn)
                self.loader._apply_vram_policy(pipe, self.loader._cfg)
                with self.loader.lock:
                    result = pipe(
                        prompt=params.prompt, control_image=ci,
                        controlnet_conditioning_scale=float(controlnet_scale),
                        num_inference_steps=params.num_inference_steps,
                        guidance_scale=params.guidance_scale or 3.5,
                        width=params.width, height=params.height, generator=gen)
                return result.images[0], seed
            finally:
                self.loader.clear_cuda_cache()

        try:
            pipe = diffusers.StableDiffusionXLControlNetPipeline(
                vae=handle.pipeline.vae, unet=handle.pipeline.unet,
                controlnet=cn, scheduler=handle.pipeline.scheduler,
                tokenizer=handle.pipeline.tokenizer,
                tokenizer_2=getattr(handle.pipeline, "tokenizer_2", None),
                text_encoder=handle.pipeline.text_encoder,
                text_encoder_2=getattr(handle.pipeline, "text_encoder_2", None),
            ).to(_dtype_for_handle(handle))
            with self.loader.lock:
                result = pipe(
                    prompt=params.prompt, image=ci,
                    num_inference_steps=params.num_inference_steps,
                    guidance_scale=params.guidance_scale,
                    negative_prompt=params.negative_prompt, generator=gen,
                )
            return result.images[0], seed
        finally:
            self.loader.clear_cuda_cache()

    # â”€â”€ IP-Adapter reference-conditioned generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate_with_ip_adapter(
        self, params: GenParams, reference_images: list[Image.Image],
        ip_adapter_model_id: str = "h94/IP-Adapter", weight: float = 0.6,
        base_model_id: Optional[str] = None,
    ) -> tuple[Image.Image, int]:
        handle = self._ensure_handle(base_model_id)
        self.loader.attach_ip_adapter(ip_adapter_model_id, weight=weight)
        device = self._device()
        gen, seed = _make_generator(device, params.seed)
        refs = [r.convert("RGB") for r in reference_images]
        is_flux = (handle.base_kind or "").startswith("flux")
        kwargs = dict(
            prompt=params.prompt, ip_adapter_image=refs,
            width=params.width, height=params.height,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale or (3.5 if is_flux else 7.5),
            generator=gen,
        )
        if not is_flux:
            kwargs["negative_prompt"] = params.negative_prompt
        try:
            with self.loader.lock:
                result = handle.pipeline(**kwargs)
            return result.images[0], seed
        finally:
            try:
                handle.pipeline.set_ip_adapter_scale(0.0)
            except Exception:
                pass


    # â”€â”€ Pose ControlNet (+ optional IP-Adapter identity) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate_with_pose(
        self, params: GenParams, pose_image: Image.Image, controlnet_model_id: str,
        ref_images: Optional[list[Image.Image]] = None, controlnet_scale: float = 0.7,
        ip_weight: float = 0.7, base_model_id: Optional[str] = None,
    ) -> tuple[Image.Image, int]:
        """Generate a character in a target pose/facing using ControlNet-OpenPose
        for structure and IP-Adapter for identity. This is the local stand-in for
        a trained rotation/skeleton-animation model: the skeleton dictates the
        facing/pose, the reference keeps the character on-model. Raises on any
        setup failure so callers can fall back to the reference-only path."""
        diffusers = _import_diffusers()
        handle = self._ensure_handle(base_model_id)
        cn = self.loader.attach_controlnet(controlnet_model_id, controlnet_scale)
        if cn is None:
            raise RuntimeError("openpose controlnet unavailable")

        if (handle.base_kind or "").startswith("flux"):
            return self._flux_pose(handle, cn, params, pose_image, ref_images,
                                   controlnet_scale, ip_weight)

        cache = self.loader._cache
        pipe = cache.get("cn_pipe")
        if pipe is None or cache.get("cn_pipe_src") != id(handle.pipeline):
            pipe = diffusers.StableDiffusionXLControlNetPipeline.from_pipe(
                handle.pipeline, controlnet=cn)
            self.loader._apply_vram_policy(pipe, self.loader._cfg)
            cache["cn_pipe"] = pipe
            cache["cn_pipe_src"] = id(handle.pipeline)

        device = self._device()
        gen, seed = _make_generator(device, params.seed)
        ctrl = pose_image.convert("RGB").resize((params.width, params.height), Image.LANCZOS)

        used_ip = False
        if ref_images:
            try:
                ip_idx = self.loader._find_index("h94/IP-Adapter")
                ip_sources = []
                if ip_idx is not None:
                    ip_sources.append(ip_idx.local_path)
                ip_sources.append("h94/IP-Adapter")
                for ip_source in ip_sources:
                    try:
                        pipe.load_ip_adapter(ip_source, subfolder="sdxl_models",
                                             weight_name="ip-adapter_sdxl.safetensors")
                        pipe.set_ip_adapter_scale(ip_weight)
                        used_ip = True
                        break
                    except Exception:
                        continue
            except Exception:
                used_ip = False

        kwargs = dict(
            prompt=params.prompt, image=ctrl,
            controlnet_conditioning_scale=float(controlnet_scale),
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            negative_prompt=params.negative_prompt, generator=gen,
            width=params.width, height=params.height,
        )
        if used_ip:
            kwargs["ip_adapter_image"] = [r.convert("RGB") for r in ref_images]
        try:
            with self.loader.lock:
                result = pipe(**kwargs)
        finally:
            if used_ip:
                try:
                    pipe.set_ip_adapter_scale(0.0)
                except Exception:
                    pass
        return result.images[0], seed


    # ── FLUX pose (ControlNet + optional XLabs IP-Adapter) ──────────────────
    def _flux_pose(self, handle, cn, params, pose_image, ref_images,
                   controlnet_scale, ip_weight):
        """FLUX equivalent of the SDXL pose path: FluxControlNet for structure,
        XLabs flux IP-Adapter for identity. Cached per resident pipe."""
        diffusers = _import_diffusers()
        FCN = getattr(diffusers, "FluxControlNetPipeline", None)
        if FCN is None:
            raise RuntimeError("FluxControlNetPipeline unavailable in this diffusers build")
        cache = self.loader._cache
        pipe = cache.get("flux_cn_pipe")
        if pipe is None or cache.get("flux_cn_pipe_src") != id(handle.pipeline):
            pipe = FCN.from_pipe(handle.pipeline, controlnet=cn)
            self.loader._apply_vram_policy(pipe, self.loader._cfg)
            cache["flux_cn_pipe"] = pipe
            cache["flux_cn_pipe_src"] = id(handle.pipeline)

        device = self._device()
        gen, seed = _make_generator(device, params.seed)
        ctrl = pose_image.convert("RGB").resize((params.width, params.height), Image.LANCZOS)

        used_ip = False
        if ref_images:
            try:
                ip_idx = self.loader._find_index("XLabs-AI/flux-ip-adapter")
                ip_source = ip_idx.local_path if ip_idx else "XLabs-AI/flux-ip-adapter"
                pipe.load_ip_adapter(
                    ip_source, weight_name="ip_adapter.safetensors",
                    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14")
                pipe.set_ip_adapter_scale(ip_weight)
                used_ip = True
            except Exception:
                used_ip = False

        kwargs = dict(
            prompt=params.prompt, control_image=ctrl,
            controlnet_conditioning_scale=float(controlnet_scale),
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale or 3.5,
            width=params.width, height=params.height, generator=gen,
        )
        if used_ip:
            kwargs["ip_adapter_image"] = [r.convert("RGB") for r in ref_images]
        try:
            with self.loader.lock:
                result = pipe(**kwargs)
        finally:
            if used_ip:
                try:
                    pipe.set_ip_adapter_scale(0.0)
                except Exception:
                    pass
        return result.images[0], seed


def _dtype_for_handle(handle: RuntimeHandle):
    torch = _import_torch()
    if handle.precision == "fp16":
        return torch.float16
    if handle.precision == "bf16":
        return torch.bfloat16
    return torch.float32


