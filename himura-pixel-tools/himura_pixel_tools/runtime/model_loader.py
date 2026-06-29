"""Local model loader â€” owns all model discovery, validation, loading, adapter
composition, and VRAM management inside Himura Pixel Tools.

Implements every function in ``spec.local_model_loader_spec.required_loader_functions``.
Uses Diffusers/PyTorch as a *library* (no ComfyUI). Designed for a single 12GB
VRAM GPU (RTX 3060): one base model resident at a time, adapters hot-swapped
per job and unloaded after, sequential batching.

The loader is thread-safe for the common (single-generate) case but generation
itself is serialized through a lock to honor ``max_parallel_jobs_on_rtx3060``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .. import config
from ..schemas.models import ModelIndex, ValidationReport, VRAMReport

# â”€â”€ Optional torch/diffusers imports (lazy, guarded) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_torch = None
_diffusers = None
_Hub = None
_torch_directml = None
_torch_directml_checked = False


def _import_torch():
    global _torch
    if _torch is None:
        import torch as _t  # noqa
        _torch = _t
    return _torch


def _import_diffusers():
    global _diffusers
    if _diffusers is None:
        import diffusers as _d  # noqa
        _diffusers = _d
    return _diffusers


def _hub():
    global _Hub
    if _Hub is None:
        from huggingface_hub import snapshot_download, hf_hub_download
        _Hub = type("Hub", (), {"snapshot_download": staticmethod(snapshot_download),
                                "hf_hub_download": staticmethod(hf_hub_download)})
    return _Hub

def _import_torch_directml_optional():
    global _torch_directml, _torch_directml_checked
    if not _torch_directml_checked:
        _torch_directml_checked = True
        try:
            import torch_directml as _tdml  # noqa
            _torch_directml = _tdml
        except Exception:
            _torch_directml = None
    return _torch_directml


def _preferred_backend(cfg=None) -> str:
    cfg = cfg or config.RuntimeConfig.load()
    extras = cfg.extras if isinstance(getattr(cfg, "extras", None), dict) else {}
    value = extras.get("gpu_backend") or os.environ.get("HIMURA_GPU_BACKEND") or "auto"
    return str(value).strip().lower()


def _directml_device(cfg=None):
    pref = _preferred_backend(cfg)
    if pref not in {"auto", "directml", "dml", "amd", "intel"}:
        return None
    tdm = _import_torch_directml_optional()
    if tdm is None:
        return None
    try:
        return tdm.device()
    except Exception:
        return None

# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


_BASE_CLASS_MAP = {
    "sdxl": "StableDiffusionXLPipeline",
    "sd15": "StableDiffusionPipeline",
    "flux": "FluxPipeline",
}


def _dtype_for(precision: str):
    torch = _import_torch()
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision in ("q8_gguf", "q6_gguf", "q5_gguf", "q4_gguf"):
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return torch.float32


def _detect_base_compat(model_id: str) -> list[str]:
    lc = model_id.lower()
    if "ideogram" in lc:
        return ["ideogram_gguf"]
    if "flux" in lc:
        return ["flux_optional_future"]
    if "xl" in lc or "sdxl" in lc:
        return ["sdxl"]
    if "stable-diffusion" in lc or "sd1" in lc or "1-5" in lc:
        return ["sd15"]
    return ["sdxl"]


def _is_flux2(name_blob: str) -> bool:
    """FLUX.2 (klein) uses a different transformer + Qwen3 text encoder than
    FLUX.1, so it must load through the Flux2* classes."""
    lc = (name_blob or "").lower()
    return any(t in lc for t in ("flux.2", "flux2", "flux-2", "klein"))


def _flux_pipeline_class(diffusers, is_flux2: bool):
    """Return the right txt2img pipeline class for the FLUX family."""
    if is_flux2:
        cls = getattr(diffusers, "Flux2KleinPipeline", None) or getattr(diffusers, "Flux2Pipeline", None)
        if cls is None:
            raise RuntimeError(
                "This is a FLUX.2 model but your diffusers build has no Flux2 pipeline. "
                "Update diffusers (pip install -U diffusers>=0.36) and retry.")
        return cls
    return getattr(diffusers, "FluxPipeline")


def _sha256_of_file(path: str, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def _first_gguf(path: Path) -> Optional[Path]:
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path
    if not path.is_dir():
        return None
    candidates = sorted(
        path.rglob("*.gguf"),
        key=lambda p: (
            "q4_k_m" not in p.name.lower(),
            "q4" not in p.name.lower(),
            "q5" not in p.name.lower(),
            p.name.lower(),
        ),
    )
    return candidates[0] if candidates else None


def _gguf_precision(path: Path) -> str:
    name = path.name.lower()
    if "q4" in name:
        return "q4_gguf"
    if "q5" in name:
        return "q5_gguf"
    if "q6" in name:
        return "q6_gguf"
    return "q8_gguf" if "q8" in name else "q4_gguf"


# â”€â”€ RuntimeHandle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@dataclass
class RuntimeHandle:
    """Opaque handle to a loaded pipeline + its active adapters."""

    model_id: str
    pipeline: Any = None
    base_kind: str = "sdxl"          # sdxl | sd15 | flux
    precision: str = "fp16"
    offloaded: bool = False
    active_loras: dict[str, float] = field(default_factory=dict)
    active_controlnet: Any = None
    active_ip_adapter: Optional[str] = None
    image_encoder: Any = None


# â”€â”€ ModelLoader â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class ModelLoader:
    """The single owner of GPU-resident models and adapters."""

    def __init__(self):
        self._lock = threading.RLock()
        self._handle: Optional[RuntimeHandle] = None
        self._cfg = config.RuntimeConfig.load()
        self._cache: dict[str, Any] = {}

    # â”€â”€ discovery / validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def scan_model_store(self) -> list[ModelIndex]:
        """scan_model_store() -> ModelIndex[] (spec required loader function).

        Detects three kinds of entries:
          - indexed folders  (contain a himura.json manifest)
          - Diffusers folders (have model_index.json / config.json)
          - single-file checkpoints (*.safetensors / *.ckpt) dropped into a kind folder
        """
        results: list[ModelIndex] = []
        for kind, _desc in config.MODEL_SUBDIRS.items():
            d = config.MODELS_ROOT / kind
            if not d.exists():
                continue
            for entry in sorted(d.iterdir()):
                # single-file checkpoint (e.g. a .safetensors dropped into base/)
                if entry.is_file() and entry.suffix in (".safetensors", ".ckpt", ".pt", ".bin", ".gguf"):
                    results.append(ModelIndex(
                        model_id=entry.stem,
                        display_name=entry.stem.replace("-", " ").replace("_", " "),
                        type=kind,
                        local_path=str(entry),
                        base_compatibility=_detect_base_compat(entry.name),
                        precision=_gguf_precision(entry) if entry.suffix.lower() == ".gguf" else "fp16",
                    ))
                    continue
                if not entry.is_dir():
                    continue
                # indexed folder (himura.json manifest written by the downloader)
                idx_file = entry / "himura.json"
                if idx_file.exists():
                    try:
                        raw = json.loads(idx_file.read_text(encoding="utf-8"))
                        gguf = _first_gguf(entry)
                        raw["local_path"] = str(gguf or entry)
                        if gguf:
                            raw["base_compatibility"] = _detect_base_compat(f"{raw.get('model_id', '')} {gguf.name}")
                            raw["precision"] = _gguf_precision(gguf)
                        results.append(ModelIndex(**raw))
                        continue
                    except Exception:
                        pass
                gguf = _first_gguf(entry)
                if gguf:
                    results.append(ModelIndex(
                        model_id=entry.name,
                        display_name=entry.name.replace("-", " ").replace("_", " "),
                        type=kind,
                        local_path=str(gguf),
                        base_compatibility=_detect_base_compat(f"{entry.name} {gguf.name}"),
                        precision=_gguf_precision(gguf),
                    ))
                    continue
                # infer from a Diffusers folder / bare directory
                results.append(ModelIndex(
                    model_id=entry.name,
                    display_name=entry.name.replace("-", " ").replace("_", " "),
                    type=kind,
                    local_path=str(entry),
                    base_compatibility=_detect_base_compat(entry.name),
                ))
        return results

    def validate_model_files(self, model_id: str) -> ValidationReport:
        """validate_model_files(model_id) -> ValidationReport."""
        idx = self._find_index(model_id)
        if idx is None:
            return ValidationReport(model_id=model_id, valid=False, errors=["model not found in store"])
        path = Path(idx.local_path)
        errors: list[str] = []
        warnings: list[str] = []
        present: list[str] = []
        missing: list[str] = []

        if idx.type == "base":
            needed = ["model_index.json", "unet", "text_encoder"]
            # diffusers dir, single safetensors, or a GGUF file inside a downloaded folder
            gguf = _first_gguf(path)
            is_safetensors = path.is_file() and path.suffix == ".safetensors"
            if is_safetensors or gguf:
                present.append((gguf or path).name)
            else:
                for n in needed:
                    (present if (path / n).exists() else missing).append(n)
                if (path / "config.json").exists():
                    present.append("config.json")
        elif idx.type == "lora":
            target = path if path.is_file() else (next(path.rglob("*.safetensors"), None) if path.exists() else None)
            ok = bool(target and target.is_file() and target.suffix == ".safetensors")
            (present if ok else missing).append(target.name if ok else (path.name if path.name else idx.model_id))
        elif idx.type == "controlnet":
            target = None
            if path.is_file() and path.suffix == ".safetensors":
                target = path
            elif path.exists():
                target = next(path.rglob("*.safetensors"), None)
            if target and target.exists():
                present.append(target.name)
            else:
                missing.append("diffusion_pytorch_model.safetensors")
        elif idx.type in {"ip_adapter", "motion"}:
            target = None
            if path.is_file() and path.suffix in (".safetensors", ".bin"):
                target = path
            elif path.exists():
                target = next(path.rglob("*.safetensors"), None) or next(path.rglob("*.bin"), None)
            if target and target.exists():
                present.append(target.name)
            else:
                missing.append(path.name if path.name else idx.model_id)
        else:
            if path.exists():
                present.append(path.name)
            else:
                missing.append(str(path))

        # sha verification if declared
        if idx.sha256:
            target = path if path.is_file() else (next(path.rglob("*.safetensors"), None) or _first_gguf(path))
            if target and target.is_file():
                actual = _sha256_of_file(str(target))
                if actual and actual != idx.sha256:
                    errors.append(f"sha256 mismatch: expected {idx.sha256[:12]}â€¦ got {actual[:12]}â€¦")
        if not idx.license:
            warnings.append("no license recorded â€” required for commercial use")

        return ValidationReport(
            model_id=model_id,
            valid=(not missing and not errors),
            errors=errors,
            warnings=warnings,
            files_present=present,
            files_missing=missing,
        )

    # â”€â”€ loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def load_base_model(self, model_id: str, precision: str = "fp16") -> RuntimeHandle:
        """load_base_model(model_id, precision) -> RuntimeHandle."""
        torch = _import_torch()
        diffusers = _import_diffusers()
        with self._lock:
            if self._handle and self._handle.model_id == model_id and self._handle.pipeline is not None:
                return self._handle

            # Enforce "keep only one base model loaded"
            if self._handle:
                self.unload_current_pipeline()

            idx = self._find_index(model_id) or ModelIndex(
                model_id=model_id, display_name=model_id, type="base",
                local_path=model_id, base_compatibility=_detect_base_compat(model_id),
            )
            base_kind = idx.base_compatibility[0] if idx.base_compatibility else "sdxl"
            cfg = self._cfg
            if base_kind == "ideogram_gguf":
                raise RuntimeError(
                    "Ideogram 4 GGUF is installed for the stable-diffusion.cpp backend. "
                    "The current in-app Diffusers runtime can download and validate it, "
                    "but generation support needs the planned stable-diffusion.cpp runner.")
            effective_precision = precision
            has_directml = _directml_device(cfg) is not None
            if not torch.cuda.is_available() and not has_directml and precision in ("fp16", "bf16"):
                # CPU float16/bfloat16 diffusion is both painfully slow and can
                # fail for unsupported ops. Use fp32 when CUDA/DirectML are unavailable.
                effective_precision = "fp32"
            dtype = _dtype_for(effective_precision)
            local_path = Path(idx.local_path)
            gguf_path = _first_gguf(local_path)
            if gguf_path and base_kind == "flux_optional_future":
                local_path = gguf_path
            is_single_file = local_path.is_file() and local_path.suffix in (".safetensors", ".ckpt", ".gguf")

            # A base must be a Diffusers pipeline (model_index.json) or hold a
            # single-file checkpoint / GGUF. Repos like ByteDance/SDXL-Lightning
            # ship only accelerator UNet/LoRA files and can't be a standalone
            # base — fail clearly instead of with a cryptic weights error.
            if not is_single_file and local_path.is_dir() and base_kind in ("sdxl", "sd15"):
                has_diffusers = (local_path / "model_index.json").exists()
                has_weights = bool(next(local_path.rglob("*.safetensors"), None))
                if not has_diffusers and not has_weights:
                    raise RuntimeError(
                        f"'{model_id}' has no loadable base weights (no model_index.json "
                        "and no .safetensors). If this is an accelerator like "
                        "SDXL-Lightning, install it as a LoRA (Models tab) and apply it on "
                        "top of an SDXL base instead of selecting it as the base model.")

            vae = None
            if base_kind == "sdxl":
                # Use the fp16-fix VAE to avoid NaNs (spec recommendation).
                try:
                    vae_idx = self._find_index("madebyollin/sdxl-vae-fp16-fix")
                    vae_source = vae_idx.local_path if vae_idx else "madebyollin/sdxl-vae-fp16-fix"
                    vae = diffusers.AutoencoderKL.from_pretrained(
                        vae_source, torch_dtype=dtype)
                except Exception:
                    vae = None
                if is_single_file:
                    pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
                        str(local_path), vae=vae, torch_dtype=dtype)
                else:
                    pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
                        str(local_path), vae=vae, torch_dtype=dtype, use_safetensors=True)
            elif base_kind == "sd15":
                if is_single_file:
                    pipe = diffusers.StableDiffusionPipeline.from_single_file(
                        str(local_path), torch_dtype=dtype)
                else:
                    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
                        str(local_path), torch_dtype=dtype, use_safetensors=True)
            elif base_kind == "flux_optional_future":
                is_flux2 = _is_flux2(f"{idx.model_id} {local_path.name}")
                if local_path.is_file() and local_path.suffix == ".gguf":
                    pipe = self._load_flux_gguf_pipeline(diffusers, local_path, idx, dtype)
                elif is_single_file:
                    Cls = _flux_pipeline_class(diffusers, is_flux2)
                    pipe = Cls.from_single_file(str(local_path), torch_dtype=dtype)
                else:
                    # diffusers folder: let DiffusionPipeline read _class_name
                    # (Flux2KleinPipeline vs FluxPipeline) from model_index.json.
                    pipe = diffusers.DiffusionPipeline.from_pretrained(str(local_path), torch_dtype=dtype)
            else:
                pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
                    str(local_path), vae=vae, torch_dtype=dtype, use_safetensors=True)

            # VRAM optimizations
            self._apply_vram_policy(pipe, cfg)

            self._handle = RuntimeHandle(
                model_id=model_id, pipeline=pipe, base_kind=base_kind, precision=effective_precision,
                offloaded=cfg.sequential_cpu_offload_when_low_vram,
            )
            return self._handle

    def _load_flux_gguf_pipeline(self, diffusers, local_path: Path, idx: ModelIndex, dtype):
        """Load a FLUX GGUF transformer and wrap it in the matching pipeline.

        FLUX.2-klein GGUFs (the unsloth/FLUX.2-klein-4B-GGUF the user has) must
        use ``Flux2Transformer2DModel`` + ``Flux2KleinPipeline``; using the
        FLUX.1 classes is what produced the "Unable to load weights" OSError.
        The pipeline's text-encoder/VAE/tokenizer are pulled from the matching
        Black Forest Labs base repo on first load (cached in the project).
        """
        GGUFQuantizationConfig = getattr(diffusers, "GGUFQuantizationConfig", None)
        if GGUFQuantizationConfig is None:
            raise RuntimeError(
                "Flux GGUF loading requires a diffusers build with GGUFQuantizationConfig. "
                "Update diffusers (pip install -U diffusers), then retry.")
        try:
            import gguf  # noqa: F401  (diffusers' GGUF reader needs this package)
        except Exception:
            raise RuntimeError(
                "Loading a .gguf model requires the 'gguf' package. Install it into the "
                "app venv: .venv\\Scripts\\python -m pip install gguf  (then restart the app).")

        name_blob = f"{idx.model_id} {local_path.name}".lower()
        is_flux2 = _is_flux2(name_blob)
        extras = self._cfg.extras if isinstance(self._cfg.extras, dict) else {}
        base_repo = extras.get("flux_gguf_base_repo")
        qconf = GGUFQuantizationConfig(compute_dtype=dtype)

        if is_flux2:
            TransformerCls = getattr(diffusers, "Flux2Transformer2DModel", None)
            PipelineCls = getattr(diffusers, "Flux2KleinPipeline", None) or getattr(diffusers, "Flux2Pipeline", None)
            if TransformerCls is None or PipelineCls is None:
                raise RuntimeError(
                    "This is a FLUX.2 GGUF but your diffusers build lacks Flux2 support. "
                    "Update diffusers (pip install -U diffusers>=0.36), then retry.")
            if not base_repo:
                base_repo = "black-forest-labs/FLUX.2-klein-9B" if "9b" in name_blob \
                    else "black-forest-labs/FLUX.2-klein-4B"
        else:
            TransformerCls = getattr(diffusers, "FluxTransformer2DModel", None)
            PipelineCls = getattr(diffusers, "FluxPipeline", None)
            if TransformerCls is None or PipelineCls is None:
                raise RuntimeError(
                    "Flux GGUF loading requires FluxTransformer2DModel/FluxPipeline. "
                    "Update diffusers, then retry.")
            if not base_repo:
                base_repo = "black-forest-labs/FLUX.1-dev"

        # Pull the transformer *config* from the (ungated) base repo's transformer
        # subfolder. Without this, from_single_file defaults to the gated
        # black-forest-labs/FLUX.2-dev config and fails with an access error.
        transformer = TransformerCls.from_single_file(
            str(local_path), quantization_config=qconf, torch_dtype=dtype,
            config=base_repo, subfolder="transformer")
        return PipelineCls.from_pretrained(
            base_repo, transformer=transformer, torch_dtype=dtype)

    def _apply_vram_policy(self, pipe, cfg) -> None:
        torch = _import_torch()
        try:
            if cfg.attention_slicing:
                pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            if cfg.vae_tiling and hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
        except Exception:
            pass
        cuda = torch.cuda.is_available()
        if cuda:
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # RTX 3060 12GB â†’ offload to keep base+adapter stable
            if cfg.sequential_cpu_offload_when_low_vram and vram < 14:
                try:
                    pipe.enable_model_cpu_offload()
                    return
                except Exception:
                    pass
            try:
                pipe.to("cuda")
            except Exception:
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pass
        else:
            dml = _directml_device(cfg)
            if dml is not None:
                try:
                    pipe.to(dml)
                    return
                except Exception:
                    pass
            try:
                pipe.to("cpu")
            except Exception:
                pass

    # â”€â”€ adapters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def attach_lora(self, model_id: str, weight: float = 0.8) -> None:
        """attach_lora(model_id, weight) -> None."""
        with self._lock:
            if not self._handle or not self._handle.pipeline:
                raise RuntimeError("load a base model before attaching a LoRA")
            idx = self._find_index(model_id) or ModelIndex(
                model_id=model_id, display_name=model_id, type="lora",
                local_path=model_id, base_compatibility=[])
            p = Path(idx.local_path)
            if p.is_file():
                self._handle.pipeline.load_lora_weights(str(p.parent), weight_name=p.name)
            else:
                self._handle.pipeline.load_lora_weights(str(p))
            self._handle.active_loras[model_id] = float(weight)

    def detach_lora(self, model_id: str) -> None:
        """detach_lora(model_id) -> None."""
        with self._lock:
            if self._handle and self._handle.pipeline:
                try:
                    self._handle.pipeline.unload_lora_weights()
                except Exception:
                    pass
                self._handle.active_loras.pop(model_id, None)

    def attach_ip_adapter(self, model_id: str, image_encoder_id: Optional[str] = None,
                          weight: float = 0.6) -> None:
        """attach_ip_adapter(model_id, image_encoder_id, weight) -> None."""
        with self._lock:
            if not self._handle or not self._handle.pipeline:
                raise RuntimeError("load a base model before attaching an IP-Adapter")
            pipe = self._handle.pipeline

            # FLUX uses a different IP-Adapter (XLabs) and load signature.
            if (self._handle.base_kind or "").startswith("flux"):
                idx = self._find_index(model_id) or self._find_index("XLabs-AI/flux-ip-adapter")
                flux_candidates: list[tuple[str, str]] = []
                if idx and Path(idx.local_path).exists():
                    flux_candidates.append((str(idx.local_path), "ip_adapter.safetensors"))
                flux_candidates.append(("XLabs-AI/flux-ip-adapter", "ip_adapter.safetensors"))
                last_error: Optional[Exception] = None
                for repo_or_path, weight_name in flux_candidates:
                    try:
                        pipe.load_ip_adapter(
                            repo_or_path, weight_name=weight_name,
                            image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14")
                        pipe.set_ip_adapter_scale(weight)
                        self._handle.active_ip_adapter = model_id or "XLabs-AI/flux-ip-adapter"
                        return
                    except Exception as exc:
                        last_error = exc
                raise RuntimeError(f"failed to load FLUX IP-Adapter: {last_error}") from last_error

            idx = self._find_index(model_id) or self._find_index("h94/IP-Adapter")
            requested_weight = str(Path(model_id).name or "ip-adapter_sdxl.safetensors")
            if not requested_weight.endswith(".safetensors"):
                requested_weight = "ip-adapter_sdxl.safetensors"

            load_candidates: list[tuple[str, str, str]] = []
            if idx:
                local_root = Path(idx.local_path)
                if local_root.exists():
                    load_candidates.append((str(local_root), "sdxl_models", requested_weight))
                    if requested_weight != "ip-adapter_sdxl.safetensors":
                        load_candidates.append((str(local_root), "sdxl_models", "ip-adapter_sdxl.safetensors"))
            load_candidates.append(("h94/IP-Adapter", "sdxl_models", requested_weight))
            if requested_weight != "ip-adapter_sdxl.safetensors":
                load_candidates.append(("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.safetensors"))

            last_error: Optional[Exception] = None
            for repo_or_path, subfolder, weight_name in load_candidates:
                try:
                    pipe.load_ip_adapter(repo_or_path, subfolder=subfolder, weight_name=weight_name)
                    pipe.set_ip_adapter_scale(weight)
                    self._handle.active_ip_adapter = model_id or "h94/IP-Adapter"
                    return
                except Exception as exc:
                    last_error = exc
            if last_error:
                raise RuntimeError(f"failed to load IP-Adapter from local install or fallback: {last_error}") from last_error
    def attach_controlnet(self, model_id: str, weight: float = 0.8) -> Any:
        """attach_controlnet(model_id, weight) -> loaded controlnet (None if N/A)."""
        diffusers = _import_diffusers()
        torch = _import_torch()
        with self._lock:
            if not self._handle:
                raise RuntimeError("load a base model before attaching a ControlNet")
            idx = self._find_index(model_id) or ModelIndex(
                model_id=model_id, display_name=model_id, type="controlnet",
                local_path=model_id)
            is_flux = (self._handle.base_kind or "").startswith("flux")
            CNCls = getattr(diffusers, "FluxControlNetModel", None) if is_flux else None
            if CNCls is None:
                CNCls = getattr(diffusers, "ControlNetModel", None)
            try:
                cn = CNCls.from_pretrained(
                    idx.local_path, torch_dtype=_dtype_for(self._handle.precision))
                self._handle.active_controlnet = cn
                self._cache["controlnet_weight"] = weight
                return cn
            except Exception:
                return None

    def attach_motion_adapter(self, model_id: str) -> None:
        """attach_motion_adapter(model_id) -> None (optional AnimateDiff draft)."""
        diffusers = _import_diffusers()
        torch = _import_torch()
        with self._lock:
            idx = self._find_index(model_id) or ModelIndex(
                model_id=model_id, display_name=model_id, type="motion",
                local_path=model_id)
            try:
                motion = diffusers.AnimateDiffSDXLPipeline.load_motion_adapter(
                    idx.local_path, torch_dtype=_dtype_for(
                        self._handle.precision if self._handle else "fp16"))
                self._cache["motion_adapter"] = motion
            except Exception:
                pass

    # â”€â”€ teardown / reports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def unload_current_pipeline(self) -> None:
        """unload_current_pipeline() -> None."""
        with self._lock:
            import gc
            if self._handle:
                self._handle.pipeline = None
                self._handle.active_loras.clear()
                self._handle.active_controlnet = None
                self._handle.active_ip_adapter = None
            self._handle = None
            self._cache.pop("controlnet_weight", None)
            self._cache.pop("motion_adapter", None)
            self._cache.pop("img2img_pipe", None)
            self._cache.pop("img2img_src", None)
            self._cache.pop("cn_pipe", None)
            self._cache.pop("cn_pipe_src", None)
            self._cache.pop("flux_cn_pipe", None)
            self._cache.pop("flux_cn_pipe_src", None)
            gc.collect()
            self.clear_cuda_cache()

    def clear_cuda_cache(self) -> None:
        """clear_cuda_cache() -> None."""
        try:
            torch = _import_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    def get_vram_report(self) -> VRAMReport:
        """get_vram_report() -> VRAMReport."""
        diagnostics: list[str] = []
        torch_version = None
        torch_cuda_version = None
        device = "cpu"
        try:
            torch = _import_torch()
            torch_version = getattr(torch, "__version__", None)
            torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
            cuda = torch.cuda.is_available()
            if cuda:
                device = "cuda"
                name = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserv = torch.cuda.memory_reserved() / 1024**3
                free = max(total - alloc, 0.0)
            else:
                total = alloc = reserv = free = None
                dml = _directml_device(self._cfg)
                if dml is not None:
                    device = "directml"
                    name = "DirectML GPU"
                    tdm = _import_torch_directml_optional()
                    try:
                        if hasattr(tdm, "device_name"):
                            name = f"DirectML GPU ({tdm.device_name(0)})"
                    except Exception:
                        pass
                    diagnostics.append(
                        "Using torch-directml for AMD, Intel, or non-CUDA GPU acceleration. "
                        "Some Diffusers features may fall back to CPU if DirectML lacks an operator."
                    )
                else:
                    name = "cpu"
                    if torch_cuda_version is None or (torch_version and "+cpu" in torch_version):
                        diagnostics.append(
                            "CPU-only PyTorch build installed. Re-run setup and choose CUDA for NVIDIA, "
                            "DirectML for AMD/Intel on Windows, or CPU fallback."
                        )
                    else:
                        diagnostics.append(
                            "PyTorch has a CUDA build, but no CUDA GPU is visible. "
                            "Check the NVIDIA driver or choose DirectML/CPU in setup."
                        )
        except Exception as e:
            cuda, name, total, alloc, reserv, free = False, "cpu", None, None, None, None
            diagnostics.append(f"Unable to query PyTorch runtime: {e}")
        loaded = self._handle.model_id if self._handle else None
        active = list(self._handle.active_loras.keys()) if self._handle else []
        if self._handle and self._handle.active_ip_adapter:
            active.append(self._handle.active_ip_adapter)
        device_name = name if device != "cpu" else f"cpu ({torch_version or 'torch unavailable'}, CUDA build {torch_cuda_version or 'none'})"
        return VRAMReport(
            device=device, device_name=device_name,
            total_vram_gb=total, allocated_gb=alloc, reserved_gb=reserv, free_gb=free,
            cuda_available=(device == "cuda"), torch_version=torch_version, torch_cuda_version=torch_cuda_version,
            diagnostics=diagnostics, loaded_model=loaded, active_adapters=active,
        )

    # pipeline assembly â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def build_pipeline(self, workflow_type: str, runtime_profile: Optional[dict] = None) -> RuntimeHandle:
        """build_pipeline(workflow_type, runtime_profile) -> RuntimeHandle.

        workflow_type âˆˆ {text_to_image, img2img, inpaint, controlnet, ip_adapter}.
        Ensures the base is loaded; for inpaint/controlnet it rebuilds the
        appropriate Diffusers pipeline class around the base.
        """
        profile = runtime_profile or {}
        base_id = profile.get("base_model_id") or self._cfg.last_base_model_id
        precision = profile.get("precision", self._cfg.precision)
        handle = self.load_base_model(base_id, precision=precision)

        if workflow_type in ("inpaint",):
            diffusers = _import_diffusers()
            torch = _import_torch()
            try:
                inp_pipe = diffusers.StableDiffusionXLInpaintPipeline(
                    vae=handle.pipeline.vae, unet=handle.pipeline.unet,
                    scheduler=handle.pipeline.scheduler,
                    tokenizer_1=handle.pipeline.tokenizer,
                    tokenizer_2=getattr(handle.pipeline, "tokenizer_2", None),
                    text_encoder_1=handle.pipeline.text_encoder,
                    text_encoder_2=getattr(handle.pipeline, "text_encoder_2", None),
                ).to(_dtype_for(precision))
                self._apply_vram_policy(inp_pipe, self._cfg)
                handle.pipeline = inp_pipe
            except Exception:
                pass
        return handle

    # â”€â”€ internals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _find_index(self, model_id: str) -> Optional[ModelIndex]:
        for m in self.scan_model_store():
            if m.model_id == model_id or Path(m.local_path).name == model_id:
                return m
        return None

    @property
    def handle(self) -> Optional[RuntimeHandle]:
        return self._handle

    @property
    def lock(self) -> threading.RLock:
        return self._lock







