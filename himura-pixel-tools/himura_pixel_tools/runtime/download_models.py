"""Model download / import manager.

Provides:
  - ``download_model(model_id, model_type=None, on_progress)`` â€” pulls a HF repo
    (by id like ``org/name`` OR full https URL) into the local model store and
    writes a ``himura.json`` index with sha + license metadata. Auto-detects the
    model type when not specified.
  - ``import_local_file(path, model_type, ...)`` â€” registers an existing file.
  - a CLI entry point (``python -m himura_pixel_tools.runtime.download_models``)
    used by the setup scripts' "download at startup" option and the Models menu.

Implements spec security rules: prefer safetensors, verify SHA-256 when
possible, store license metadata, never execute code from repos.

All downloads land INSIDE the project folder (config.MODELS_ROOT), and the HF
cache is redirected there too (config sets HF_HOME/HF_HUB_CACHE at import time)
so nothing leaks into ~/.cache.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from .. import config
from . import registry


def _on_progress_default(idx: int, msg: str) -> None:
    print(f"  [{idx}%] {msg}", flush=True)


# â”€â”€ repo id / URL normalization + type detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _is_civitai(source: str) -> bool:
    return "civitai.com" in (source or "").lower()


# Civitai baseModel string → our base_compatibility tag.
def _civitai_base_compat(base_model: str) -> list[str]:
    bm = (base_model or "").lower()
    if "flux" in bm:
        return ["flux_optional_future"]
    if any(k in bm for k in ("sd 1.5", "sd1.5", "sd 1.4")):
        return ["sd15"]
    # SDXL, Pony, Illustrious, NoobAI, SDXL Turbo/Lightning are all SDXL-arch.
    return ["sdxl"]


# Civitai model.type → our model store kind.
def _civitai_type(ctype: str) -> str:
    t = (ctype or "").lower()
    if t in ("lora", "locon", "lycoris", "dora"):
        return "lora"
    if t == "checkpoint":
        return "base"
    if t == "controlnet":
        return "controlnet"
    if t in ("upscaler", "esrgan"):
        return "upscale"
    return "lora"


def _civitai_resolve(source: str) -> dict:
    """Resolve a civitai.com model/version URL to a concrete file download.

    Returns {download_url, filename, mtype, base_compat, trigger, license,
    display_name, version_id}. Uses the public Civitai v1 API.
    """
    import json as _json
    import re
    s = source.strip()

    def _api_get(url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "himura-pixel-tools/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return _json.loads(r.read().decode("utf-8"))

    model_id = version_id = None
    m = re.search(r"/models/(\d+)", s)
    if m:
        model_id = m.group(1)
    mv = re.search(r"modelVersionId=(\d+)", s)
    if mv:
        version_id = mv.group(1)
    dl = re.search(r"/api/download/models/(\d+)", s)
    if dl:
        version_id = dl.group(1)

    model = ctype = "LORA"
    base_model = ""
    trigger = ""
    license_ = "see Civitai model page"
    display = None

    if model_id:
        data = _api_get(f"https://civitai.com/api/v1/models/{model_id}")
        ctype = data.get("type", "LORA")
        display = data.get("name")
        if data.get("allowCommercialUse"):
            license_ = "Civitai: " + ", ".join(
                data["allowCommercialUse"] if isinstance(data["allowCommercialUse"], list)
                else [str(data["allowCommercialUse"])])
        versions = data.get("modelVersions", []) or []
        ver = None
        if version_id:
            ver = next((v for v in versions if str(v.get("id")) == str(version_id)), None)
        ver = ver or (versions[0] if versions else {})
        version_id = ver.get("id", version_id)
        base_model = ver.get("baseModel", "")
        trigger = ", ".join((ver.get("trainedWords") or [])[:4])
        files = ver.get("files", []) or []
        prim = next((f for f in files if f.get("primary")),
                    files[0] if files else {})
        filename = prim.get("name") or f"civitai_{version_id}.safetensors"
        download_url = prim.get("downloadUrl") or f"https://civitai.com/api/download/models/{version_id}"
    elif version_id:
        ver = _api_get(f"https://civitai.com/api/v1/model-versions/{version_id}")
        ctype = (ver.get("model") or {}).get("type", "LORA")
        display = (ver.get("model") or {}).get("name")
        base_model = ver.get("baseModel", "")
        trigger = ", ".join((ver.get("trainedWords") or [])[:4])
        files = ver.get("files", []) or []
        prim = next((f for f in files if f.get("primary")), files[0] if files else {})
        filename = prim.get("name") or f"civitai_{version_id}.safetensors"
        download_url = prim.get("downloadUrl") or f"https://civitai.com/api/download/models/{version_id}"
    else:
        raise ValueError(f"Could not find a Civitai model/version id in: {source}")

    return {
        "download_url": download_url,
        "filename": filename,
        "mtype": _civitai_type(ctype),
        "base_compat": _civitai_base_compat(base_model),
        "trigger": trigger,
        "license": license_,
        "display_name": display or filename,
        "version_id": version_id,
        "base_model": base_model,
    }


def _download_civitai(source: str, model_type: Optional[str] = None,
                      on_progress: Optional[Callable[[int, str], None]] = None) -> str:
    """Download a single Civitai model file into the local store + index it."""
    on_progress = on_progress or _on_progress_default
    info = _civitai_resolve(source)
    mtype = model_type or info["mtype"]
    safe_name = Path(info["filename"]).stem.replace("/", "_")
    target_dir = config.model_dir(mtype) / f"civitai__{safe_name}"
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / info["filename"]

    if dest.exists() and dest.stat().st_size > 0:
        on_progress(100, f"already present: {dest.name}")
        _write_index(target_dir, info["display_name"], mtype, info["base_compat"],
                     info["license"], trigger=info.get("trigger"), source_url=source)
        return str(target_dir)

    url = info["download_url"]
    key = (config.RuntimeConfig.load().civitai_api_key or "").strip()
    if key:
        url += ("&" if "?" in url else "?") + "token=" + key

    on_progress(5, f"downloading from Civitai: {info['filename']} ({mtype})")
    req = urllib.request.Request(url, headers={"User-Agent": "himura-pixel-tools/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            read = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if total:
                    on_progress(min(95, int(read * 90 / total) + 5),
                                f"{read // (1 << 20)}/{total // (1 << 20)} MB")
    except urllib.error.HTTPError as e:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        if e.code in (401, 403):
            raise RuntimeError(
                "Civitai requires authentication for this download. Add your Civitai API "
                "key in Settings (civitai_api_key) and retry.") from e
        raise RuntimeError(f"Civitai download failed (HTTP {e.code})") from e

    if dest.stat().st_size < 1024:
        raise RuntimeError("Civitai download returned an unexpectedly small file "
                           "(likely an auth/HTML page). Set a Civitai API key in Settings.")
    on_progress(97, "indexing")
    _write_index(target_dir, info["display_name"], mtype, info["base_compat"],
                 info["license"], sha=compute_sha(dest), trigger=info.get("trigger"),
                 source_url=source)
    on_progress(100, f"done: {info['filename']}")
    return str(target_dir)


def normalize_repo_id(source: str) -> str:
    """Accept 'org/name', 'https://huggingface.co/org/name', or a resolve URL.

    Returns the canonical 'org/name' repo id. Raises if it isn't a HF repo.
    """
    s = source.strip()
    if s.startswith("http://") or s.startswith("https://"):
        parsed = urlparse(s)
        host = parsed.netloc.lower()
        if "huggingface.co" not in host:
            raise ValueError(f"Only Hugging Face URLs are supported, got host: {host}")
        path = parsed.path.lstrip("/")
        # /org/name/resolve/main/file.safetensors  ->  org/name
        if "/resolve/" in path:
            path = path.split("/resolve/")[0]
        if "/tree/" in path:
            path = path.split("/tree/")[0]
        parts = path.split("/")
        if len(parts) < 2:
            raise ValueError(f"Could not extract repo id from URL: {s}")
        return "/".join(parts[:2])
    # already an org/name id
    if "/" in s and " " not in s:
        return s
    # single word â€” treat as a user-less repo id (rare)
    return s


def auto_detect_type(repo_id: str) -> str:
    """Guess the model type from the repo/file name (base/lora/vae/controlnet/...)."""
    lc = repo_id.lower()
    if any(k in lc for k in ("lora", "dreambooth")):
        return "lora"
    if any(k in lc for k in ("vae",)):
        return "vae"
    if any(k in lc for k in ("controlnet", "control-net", "openpose", "canny", "lineart", "depth")):
        return "controlnet"
    if any(k in lc for k in ("ip-adapter", "ip_adapter", "ipadapter")):
        return "ip_adapter"
    if any(k in lc for k in ("animatediff", "motion", "motion-adapter")):
        return "motion"
    if any(k in lc for k in ("birefnet", "rembg", "segment", "background-removal")):
        return "segmentation"
    if any(k in lc for k in ("esrgan", "real-esrgan", "upscale", "scu-net")):
        return "upscale"
    # default: a full base model
    return "base"


# â”€â”€ safe file filters (avoid the multi-checkpoint / fp32 blowup) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _allow_ignore_patterns(mtype: str, repo_id: str = "") -> tuple[list[str], list[str]]:
    """Return (allow_patterns, ignore_patterns) for a safe, minimal download.

    HF repos ship redundant weights: standalone checkpoints, fp32 variants,
    legacy .bin, etc. We pull only the fp16 safetensors Diffusers layout.
    """
    rl = (repo_id or "").lower()
    ignore = [
        "*.bin", "*.pth", "*.ckpt",
        "*.fp32.*", "*_fp32.*",
        "*.onnx", "*.xml",
        "*/onnx/*", "*/ CoreML/*",
        "*.msgpack", "*.h5", "*.ot", "*.tflite",
        # full-precision safetensors (we want fp16 only for big weights)
        "unet/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "vae_1_0/diffusion_pytorch_model.safetensors",
        "vae_1_0/*",
        # standalone preview / multi-checkpoint dumps in the repo root
        "sd_xl_*", "sd_xl_base_*", "sd_xl_refiner_*",
        "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp",
        "*.mp4", "*.mov",
        "*.zip", "*.tar*", "*.7z",
    ]
    if mtype == "base" and "ideogram" in rl:
        allow = ["*.gguf", "*.json", "*.txt", "*.md"]
    elif mtype == "base":
        allow = [
            "*.fp16.safetensors", "*.gguf",
            "*.json", "*.txt",
            "tokenizer/*", "tokenizer_2/*",
            "feature_extractor/*", "scheduler/*",
        ]
    elif mtype == "vae":
        allow = ["config.json", "diffusion_pytorch_model.safetensors", "*.json", "*.txt"]
    elif mtype == "lora":
        allow = ["*.safetensors", "*.json"]
    elif mtype == "controlnet":
        allow = ["*.fp16.safetensors", "diffusion_pytorch_model.safetensors", "*.json", "*.txt"]
    elif mtype == "ip_adapter":
        if "flux" in rl or "xlabs" in rl:
            # FLUX IP-Adapter (XLabs): a single ip_adapter.safetensors at the
            # repo root; the image encoder (CLIP ViT-L/14) is fetched at load.
            allow = ["ip_adapter*.safetensors", "*.json", "*.txt"]
        else:
            # Himura runs SDXL by default. Pull only the SDXL adapter and image
            # encoder instead of the whole h94/IP-Adapter repo, which also
            # contains SD1.5 variants and multiple large duplicates.
            allow = [
                "sdxl_models/ip-adapter_sdxl.safetensors",
                "sdxl_models/image_encoder/config.json",
                "sdxl_models/image_encoder/model.safetensors",
                "*.json", "*.txt",
            ]
    elif mtype == "motion":
        allow = ["*.safetensors", "*.json", "*.txt"]
    elif mtype == "segmentation":
        allow = ["*.safetensors", "*.json", "*.txt", "*.py"]
    elif mtype == "upscale":
        allow = ["*.safetensors", "*.pth", "*.json", "*.txt"]
    else:
        allow = ["*.fp16.safetensors", "*.safetensors", "*.json", "*.txt"]
    return allow, ignore


# â”€â”€ core download â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def download_model(
    model_id: str,
    model_type: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
    revision: Optional[str] = None,
    display_name: Optional[str] = None,
) -> str:
    """Download a HF repo into the local model store. Returns the local path.

    ``model_id`` may be an ``org/name`` id or a full huggingface.co URL.
    ``model_type`` is auto-detected from the name if not provided.
    """
    on_progress = on_progress or _on_progress_default
    if _is_civitai(model_id):
        return _download_civitai(model_id, model_type=model_type, on_progress=on_progress)
    repo_id = normalize_repo_id(model_id)
    reg = registry.by_id(model_id) or registry.by_id(repo_id)
    mtype = model_type or (reg["type"] if reg else auto_detect_type(repo_id))
    base_compat = (reg["base_compatibility"] if reg else _base_compat_for(repo_id, mtype))
    license_ = reg.get("license") if reg else None
    notes = reg.get("notes", "") if reg else ""
    trigger = reg.get("trigger", "") if reg else ""
    source_url = model_id if str(model_id).startswith(("http://", "https://")) else f"https://huggingface.co/{repo_id}"
    name = display_name or (reg["id"] if reg else repo_id)

    target_dir = config.model_dir(mtype) / repo_id.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)

    # idempotent: if already present, just refresh the index
    if any(target_dir.iterdir()):
        has_weights = (any(target_dir.rglob("*.safetensors")) or
                       any(target_dir.rglob("*.pth")) or
                       any(target_dir.rglob("*.gguf")))
        if has_weights:
            on_progress(100, f"already present: {target_dir.name}")
            _write_index(target_dir, name, mtype, base_compat, license_, trigger=trigger, source_url=source_url, notes=notes)
            return str(target_dir)

    on_progress(5, f"downloading {repo_id} ({mtype}) -> {target_dir.name}")
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"huggingface_hub not available: {e}")

    allow, ignore = _allow_ignore_patterns(mtype, repo_id)

    # Case A: a direct single-file URL  (.../resolve/main/file.safetensors)
    direct_file = _extract_resolve_filename(model_id)
    try:
        if direct_file:
            on_progress(10, f"single file: {direct_file}")
            hf_hub_download(repo_id=repo_id, filename=direct_file,
                            local_dir=str(target_dir), revision=revision)
            path = str(target_dir)
        else:
            path = snapshot_download(
                repo_id=repo_id, local_dir=str(target_dir), revision=revision,
                allow_patterns=allow, ignore_patterns=ignore,
            )
    except Exception:
        # Case B: repo exposes only loose safetensors files (e.g. a LoRA repo)
        try:
            files = __import__("huggingface_hub").list_repo_files(repo_id, revision=revision)
            sfiles = [f for f in files if f.endswith((".safetensors", ".gguf"))]
            if not sfiles:
                raise
            for f in sfiles:
                hf_hub_download(repo_id=repo_id, filename=f, local_dir=str(target_dir),
                                revision=revision)
            path = str(target_dir)
        except Exception as e:
            on_progress(0, f"download failed for {repo_id}: {e}")
            raise

    on_progress(95, f"finalizing {repo_id}")
    _write_index(target_dir, name, mtype, base_compat, license_, trigger=trigger, source_url=source_url, notes=notes)
    on_progress(100, f"done: {repo_id}")
    return str(target_dir if isinstance(path, str) else path)


def _extract_resolve_filename(source: str) -> Optional[str]:
    """If source is a .../resolve/<rev>/<file> URL, return the file path."""
    if "/resolve/" not in source:
        return None
    after = source.split("/resolve/", 1)[1]
    # after = 'main/path/file.safetensors'
    if "/" in after:
        return after.split("/", 1)[1]
    return after or None


def _base_compat_for(repo_id: str, mtype: str) -> list[str]:
    """Infer base compatibility from the repo name."""
    lc = repo_id.lower()
    if mtype != "base":
        return []
    if "ideogram" in lc:
        return ["ideogram_gguf"]
    if "flux" in lc:
        return ["flux_optional_future"]
    if "xl" in lc or "sdxl" in lc:
        return ["sdxl"]
    if "stable-diffusion" in lc or "sd1" in lc or "1-5" in lc:
        return ["sd15"]
    return ["sdxl"]


def import_local_file(
    path: str,
    model_type: str,
    display_name: Optional[str] = None,
    license: Optional[str] = None,
    base_compatibility: Optional[list[str]] = None,
    move: bool = False,
) -> str:
    """Register (and optionally move) an existing local model file into the store."""
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    kind = model_type
    dest_dir = config.model_dir(kind) / src.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if src.resolve() != dest.resolve():
        if move:
            shutil.move(str(src), str(dest))
        else:
            shutil.copy2(str(src), str(dest))
    _write_index(dest_dir, display_name or src.stem, kind,
                 base_compatibility or _base_compat_for(src.stem, kind), license,
                 sha=compute_sha(dest))
    return str(dest)


def _write_index(dir_: Path, model_id: str, mtype: str, base_compat: list[str],
                 license_: Optional[str], sha: Optional[str] = None,
                 trigger: Optional[str] = None, source_url: Optional[str] = None,
                 notes: Optional[str] = None) -> None:
    import json
    from datetime import datetime, timezone
    safetensors = next(dir_.rglob("*.safetensors"), None)
    gguf = next(dir_.rglob("*.gguf"), None)
    target_weight = safetensors or gguf
    if sha is None and target_weight:
        sha = compute_sha(target_weight)
    precision = "fp16"
    if gguf:
        lname = gguf.name.lower()
        precision = "q4_gguf" if "q4" in lname else "q5_gguf" if "q5" in lname else "q6_gguf" if "q6" in lname else "q8_gguf"
    idx = {
        "model_id": model_id,
        "display_name": model_id.replace("/", " ").replace("-", " ").replace("_", " "),
        "type": mtype,
        "local_path": str(dir_),
        "source_url": source_url,
        "sha256": sha,
        "license": license_,
        "base_compatibility": base_compat,
        "precision": precision,
        "enabled": True,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "trigger": trigger or "",
        "notes": notes or "",
    }
    (dir_ / "himura.json").write_text(json.dumps(idx, indent=2), encoding="utf-8")


def compute_sha(path: Path) -> Optional[str]:
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def list_downloads() -> list[dict]:
    """Return all model records in the local store (used by the Models menu)."""
    from ..runtime.model_loader import ModelLoader
    return [m.model_dump() for m in ModelLoader().scan_model_store()]


def remove_model(model_id: str) -> bool:
    """Delete a downloaded model from the local store."""
    from ..runtime.model_loader import ModelLoader
    loader = ModelLoader()
    idx = loader._find_index(model_id)
    if idx is None:
        return False
    p = Path(idx.local_path)
    if p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    return True


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download models into the Himura Pixel Tools local model store.")
    parser.add_argument("--all", action="store_true",
                        help="download the full recommended set")
    parser.add_argument("--fast", action="store_true",
                        help="download a minimal fast set (base + vae + LCM-LoRA)")
    parser.add_argument("models", nargs="*",
                        help="specific model ids or huggingface.co URLs to download")
    parser.add_argument("--type", default=None,
                        help="override model type (base/lora/vae/controlnet/...)")
    parser.add_argument("--list", action="store_true", help="list the registry and exit")
    parser.add_argument("--installed", action="store_true", help="list installed models")
    args = parser.parse_args(argv)

    if args.list:
        print("Himura Pixel Tools â€” model registry")
        for r in registry.REGISTRY:
            print(f"  [{r['type']:11}] {r['id']:50} ({r['role']})")
        return 0
    if args.installed:
        print("Installed models:")
        for m in list_downloads():
            print(f"  [{m['type']:11}] {m['model_id']:50} {m['local_path']}")
        return 0

    config.ensure_dirs()
    ids: list[str] = list(args.models)
    if args.all:
        ids.extend(registry.DEFAULT_DOWNLOAD_SET)
    if args.fast:
        ids.extend(registry.FAST_DOWNLOAD_SET)
    if not ids:
        print("Nothing selected. Use --all, --fast, --list, --installed, or pass ids/URLs.")
        return 1

    seen = set()
    for mid in ids:
        if mid in seen:
            continue
        seen.add(mid)
        try:
            download_model(mid, model_type=args.type, on_progress=_on_progress_default)
        except Exception as e:
            print(f"  ! failed {mid}: {e}", file=sys.stderr)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






