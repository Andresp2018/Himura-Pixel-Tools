"""FastAPI server â€” the Himura Pixel Tools control layer.

Exposes every endpoint in spec.api_contract on 127.0.0.1 only, mounts the
desktop web UI, and serves the Streamable-HTTP MCP transport on /mcp.

Run with:
    python -m himura_pixel_tools.api.server --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import (Depends, FastAPI, HTTPException, Header, Request)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

from .. import __version__, config
from ..runtime import download_models, registry
from ..runtime.model_loader import ModelLoader
from . import orchestrator, security
from .jobs import get_queue
from ..schemas.jobs import (AnimateRequest, BatchRecipeRequest, CreateCharacterRequest, ExportPackRequest,
                            GenerateRequest, InpaintRequest, IsometricTileRequest, MapObjectRequest,
                            ObjectRequest, ObjectRotationRequest, ObjectStateRequest, PortraitRequest,
                            SidescrollerTilesetRequest, TilesetRequest, TilesProRequest,
                            TopdownTilesetRequest, TurnaroundRequest, UIAssetRequest,
                            ValidateAssetRequest)
from ..schemas.models import InstallModelRequest

# â”€â”€ app factory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def create_app() -> FastAPI:
    config.ensure_dirs()
    config.RuntimeConfig.load().save()
    runtime_report = orchestrator.get_loader().get_vram_report()
    print(
        "[Himura Runtime] "
        f"device={runtime_report.device_name} "
        f"torch={runtime_report.torch_version or 'unknown'} "
        f"cuda_build={runtime_report.torch_cuda_version or 'none'} "
        f"cuda_available={runtime_report.cuda_available}"
    )
    for diagnostic in runtime_report.diagnostics:
        print(f"[Himura Runtime][WARN] {diagnostic}")
    _model_progress: dict[str, dict] = {}   # tracks background model downloads
    app = FastAPI(title="Himura Pixel Tools", version=__version__,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
        allow_headers=["*"], allow_credentials=False,
    )

    # â”€â”€ static + UI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ui_dir = Path(__file__).resolve().parent.parent / "desktop"
    static_dir = ui_dir / "static"

    @app.middleware("http")
    async def _no_cache_ui(request, call_next):
        """Keep the desktop UI fresh: never cache the HTML/JS/CSS, so a server
        restart (e.g. after an edit) is reflected without a hard-reload.
        API/asset responses keep default caching."""
        resp = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
        return resp

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        idx = ui_dir / "templates" / "index.html"
        if idx.exists():
            html = idx.read_text(encoding="utf-8")
            # Inject the bearer token into the page so the same-origin desktop
            # UI can authenticate its own /api/* calls (and image <img> tags,
            # which can't set headers). The token is secrets.token_urlsafe(),
            # so it only contains [A-Za-z0-9_-] and is safe to embed in JS.
            token = security.get_or_create_token()
            inject = f'<script>window.HIMURA_TOKEN = "{token}";</script>'
            if "<!--HIMURA_TOKEN-->" in html:
                html = html.replace("<!--HIMURA_TOKEN-->", inject, 1)
            else:
                html = html.replace("</head>", inject + "</head>", 1)
            return HTMLResponse(html)
        return HTMLResponse("<h1>Himura Pixel Tools</h1><p>UI not built.</p>")

    # â”€â”€ auth dependency â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def require_token(authorization: Optional[str] = Header(default=None),
                      token: Optional[str] = None):
        cfg = config.RuntimeConfig.load()
        if not cfg.mcp_require_token:
            return True
        provided = None
        if authorization and authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        elif token:
            # query-param fallback, so <img>/<a download> URLs can authenticate
            provided = token
        if not security.check_token(provided):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")
        return True

    # â”€â”€ health / version â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @app.get("/api/health")
    def health():
        loader = orchestrator.get_loader()
        vram = loader.get_vram_report()
        return {
            "status": "healthy",
            "version": __version__,
            "device": vram.device_name,
            "cuda_available": vram.cuda_available,
            "torch_version": vram.torch_version,
            "torch_cuda_version": vram.torch_cuda_version,
            "diagnostics": vram.diagnostics,
            "model_loaded": loader.handle.model_id if loader.handle else None,
            "mcp_http_enabled": config.RuntimeConfig.load().mcp_http_enabled,
        }

    # â”€â”€ models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @app.get("/api/models")
    def list_models(_=Depends(require_token)):
        loader = orchestrator.get_loader()
        installed = loader.scan_model_store()
        installed_ids = {m.model_id for m in installed}

        def recommendation_installed(r: dict) -> bool:
            rid = r["id"]
            try:
                repo_id = download_models.normalize_repo_id(rid)
            except Exception:
                repo_id = rid
            direct_file = download_models._extract_resolve_filename(rid) or ""
            direct_name = Path(direct_file).name if direct_file else ""
            is_civitai = "civitai.com" in rid.lower()
            for model in installed:
                local_name = Path(model.local_path).name
                if is_civitai and getattr(model, "source_url", None) == rid:
                    return True
                if (rid in (model.model_id, model.local_path)
                        or repo_id == model.model_id
                        or rid in model.model_id
                        or (direct_name and direct_name == local_name)):
                    return True
            return rid in installed_ids or repo_id in installed_ids

        recommendations = [
            {"id": r["id"], "type": r["type"], "role": r["role"],
             "license": r.get("license"),
             "base_compatibility": r.get("base_compatibility", []),
             "trigger": r.get("trigger"),
             "installed": recommendation_installed(r)}
            for r in registry.REGISTRY
        ]
        active = config.RuntimeConfig.load().last_base_model_id
        return {"installed": [m.model_dump() for m in installed],
                "recommendations": recommendations,
                "vram": loader.get_vram_report().model_dump(),
                "active_base_model": active,
                "models_root": str(config.MODELS_ROOT)}

    @app.get("/api/loras")
    def list_loras(_=Depends(require_token)):
        """Installed LoRAs + the active base, so each tab can show a base-aware
        LoRA selector."""
        loader = orchestrator.get_loader()
        loras = [
            {"model_id": m.model_id, "display_name": m.display_name,
             "base_compatibility": m.base_compatibility,
             "trigger": getattr(m, "trigger", "") or "",
             "source_url": getattr(m, "source_url", "") or "",
             "notes": getattr(m, "notes", "") or "",
             "local_path": getattr(m, "local_path", "") or ""}
            for m in loader.scan_model_store() if m.type == "lora"
        ]
        return {"loras": loras,
                "active_base_model": config.RuntimeConfig.load().last_base_model_id}

    @app.post("/api/models/install")
    def install_model(req: InstallModelRequest, _=Depends(require_token)):
        try:
            if req.local_path and Path(req.local_path).exists():
                path = download_models.import_local_file(
                    req.local_path, req.model_type, req.display_name,
                    req.license, req.base_compatibility)
            else:
                path = download_models.download_model(
                    req.source_url or req.display_name, model_type=req.model_type)
            return {"ok": True, "local_path": path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/models/download")
    def download_one(body: dict, _=Depends(require_token)):
        """Download a single model by HF id or URL. Runs in background."""
        source = (body or {}).get("source") or (body or {}).get("model_id")
        if not source:
            raise HTTPException(status_code=400, detail="source required (HF id or URL)")
        mtype = (body or {}).get("model_type")
        import threading

        # track progress so the UI can poll
        progress_key = source
        _model_progress[progress_key] = {"status": "downloading", "message": "started", "pct": 0}

        def _bg():
            def _cb(idx, msg):
                pct = max(0, min(100, int(idx)))
                _model_progress[progress_key] = {"status": "downloading", "message": msg, "pct": pct}
            try:
                path = download_models.download_model(source, model_type=mtype, on_progress=_cb)
                _model_progress[progress_key] = {"status": "done", "message": "complete",
                                                  "pct": 100, "local_path": path}
            except Exception as e:
                _model_progress[progress_key] = {"status": "error", "message": str(e), "pct": 0}

        threading.Thread(target=_bg, daemon=True).start()
        return {"ok": True, "source": source, "tracking": progress_key}

    @app.get("/api/models/download-status")
    def download_status(source: str, _=Depends(require_token)):
        return _model_progress.get(source, {"status": "idle", "message": "no download", "pct": 0})

    @app.post("/api/models/download-all")
    def download_all(_=Depends(require_token)):
        """Download the full recommended set (startup option)."""
        import threading
        progress_key = "__all__"
        _model_progress[progress_key] = {"status": "downloading", "message": "started", "pct": 0}

        def _bg():
            total = len(registry.DEFAULT_DOWNLOAD_SET)
            for i, mid in enumerate(registry.DEFAULT_DOWNLOAD_SET):
                _model_progress[progress_key] = {"status": "downloading",
                                                  "message": f"downloading {mid}",
                                                  "pct": int(i * 100 / total)}
                try:
                    download_models.download_model(mid)
                except Exception:
                    pass
            _model_progress[progress_key] = {"status": "done", "message": "complete", "pct": 100}

        threading.Thread(target=_bg, daemon=True).start()
        return {"ok": True, "downloading": registry.DEFAULT_DOWNLOAD_SET}

    @app.post("/api/models/remove")
    def remove_model(body: dict, _=Depends(require_token)):
        model_id = (body or {}).get("model_id")
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id required")
        ok = download_models.remove_model(model_id)
        return {"ok": ok}

    @app.post("/api/models/set-active")
    def set_active(body: dict, _=Depends(require_token)):
        """Choose which installed base model is used for generation."""
        model_id = (body or {}).get("model_id")
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id required")
        cfg = config.RuntimeConfig.load()
        cfg.last_base_model_id = model_id
        cfg.save()
        # unload the current pipeline so the new one loads on next generation
        try:
            orchestrator.get_loader().unload_current_pipeline()
        except Exception:
            pass
        return {"ok": True, "active_base_model": model_id}

    # â”€â”€ runtime load-profile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @app.post("/api/runtime/load-profile")
    def load_profile(body: dict = None, _=Depends(require_token)):
        body = body or {}
        loader = orchestrator.get_loader()
        base = body.get("base_model_id") or config.RuntimeConfig.load().last_base_model_id
        precision = body.get("precision", config.RuntimeConfig.load().precision)
        h = loader.load_base_model(base, precision=precision)
        return {"ok": True, "model_id": h.model_id, "vram": loader.get_vram_report().model_dump()}

    @app.post("/api/runtime/unload")
    def unload(_=Depends(require_token)):
        orchestrator.get_loader().unload_current_pipeline()
        return {"ok": True}

    @app.get("/api/runtime/vram")
    def vram(_=Depends(require_token)):
        return orchestrator.get_loader().get_vram_report().model_dump()

    # â”€â”€ job endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    queue = get_queue()

    def _run(job_type, request_model, fn):
        def _wrapped(job):
            return fn(job, request_model)
        job = queue.enqueue(job_type, request_model.model_dump(), _wrapped)
        return job.response()

    @app.post("/api/jobs/generate")
    def jobs_generate(req: GenerateRequest, _=Depends(require_token)):
        return _run("generate", req, orchestrator.generate_asset)

    @app.post("/api/jobs/create-character")
    def jobs_create_character(req: CreateCharacterRequest, _=Depends(require_token)):
        return _run("create_character", req, orchestrator.create_character)

    @app.post("/api/jobs/turnaround")
    def jobs_turnaround(req: TurnaroundRequest, _=Depends(require_token)):
        return _run("turnaround", req, orchestrator.generate_turnaround)

    @app.post("/api/jobs/animate-character")
    def jobs_animate(req: AnimateRequest, _=Depends(require_token)):
        return _run("animate", req, orchestrator.animate_character)

    @app.post("/api/jobs/portrait")
    def jobs_portrait(req: PortraitRequest, _=Depends(require_token)):
        return _run("portrait", req, orchestrator.generate_portrait)

    @app.post("/api/jobs/inpaint")
    def jobs_inpaint(req: InpaintRequest, _=Depends(require_token)):
        return _run("inpaint", req, orchestrator.inpaint_asset)

    @app.post("/api/jobs/tileset")
    def jobs_tileset(req: TilesetRequest, _=Depends(require_token)):
        return _run("tileset", req, orchestrator.create_tileset)

    @app.post("/api/jobs/object")
    def jobs_object(req: ObjectRequest, _=Depends(require_token)):
        return _run("object", req, orchestrator.create_1_direction_object)

    @app.post("/api/jobs/object-8dir")
    def jobs_object_8dir(req: ObjectRotationRequest, _=Depends(require_token)):
        return _run("object_8dir", req, orchestrator.create_8_direction_object)

    @app.post("/api/jobs/object-state")
    def jobs_object_state(req: ObjectStateRequest, _=Depends(require_token)):
        return _run("object_state", req, orchestrator.create_object_state)

    @app.post("/api/jobs/map-object")
    def jobs_map_object(req: MapObjectRequest, _=Depends(require_token)):
        return _run("map_object", req, orchestrator.create_map_object)

    @app.post("/api/jobs/ui-asset")
    def jobs_ui_asset(req: UIAssetRequest, _=Depends(require_token)):
        return _run("ui_asset", req, orchestrator.create_ui_asset)

    @app.post("/api/jobs/topdown-tileset")
    def jobs_topdown_tileset(req: TopdownTilesetRequest, _=Depends(require_token)):
        return _run("topdown_tileset", req, orchestrator.create_topdown_tileset)

    @app.post("/api/jobs/sidescroller-tileset")
    def jobs_sidescroller_tileset(req: SidescrollerTilesetRequest, _=Depends(require_token)):
        return _run("sidescroller_tileset", req, orchestrator.create_sidescroller_tileset)

    @app.post("/api/jobs/isometric-tile")
    def jobs_isometric_tile(req: IsometricTileRequest, _=Depends(require_token)):
        return _run("isometric_tile", req, orchestrator.create_isometric_tile)

    @app.post("/api/jobs/tiles-pro")
    def jobs_tiles_pro(req: TilesProRequest, _=Depends(require_token)):
        return _run("tiles_pro", req, orchestrator.create_tiles_pro)

    @app.post("/api/jobs/batch-recipe")
    def jobs_batch_recipe(req: BatchRecipeRequest, _=Depends(require_token)):
        return _run("batch_recipe", req, orchestrator.create_batch_recipe)

    @app.get("/api/jobs")
    def jobs_list(_=Depends(require_token)):
        return [j.to_dict() for j in queue.list_jobs()]

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, _=Depends(require_token)):
        j = queue.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="job not found")
        return j.to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str, _=Depends(require_token)):
        return {"ok": queue.cancel(job_id)}

    @app.post("/api/jobs/exclude-saved-outputs")
    def jobs_exclude_saved_outputs(_=Depends(require_token)):
        count = queue.exclude_all_saved_outputs()
        return {"ok": True, "jobs_updated": count}

    @app.post("/api/jobs/{job_id}/exclude-saved-output")
    def job_exclude_saved_output(job_id: str, _=Depends(require_token)):
        ok = queue.exclude_saved_outputs(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail="job not found")
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/stream")
    def job_stream(job_id: str, _=Depends(require_token)):
        """SSE stream of job progress."""
        import time
        def ev():
            last = None
            for _ in range(3600):
                j = queue.get(job_id)
                if j:
                    d = j.to_dict()
                    if d != last:
                        yield f"data: {json.dumps(d)}\n\n"
                        last = d
                    if j.status in ("succeeded", "failed", "cancelled", "needs_review"):
                        return
                time.sleep(0.5)
        return StreamingResponse(ev(), media_type="text/event-stream")

    # â”€â”€ assets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @app.get("/api/assets")
    def assets_list(_=Depends(require_token)):
        from .. import db
        from sqlmodel import select
        with db.get_session() as s:
            rows = s.exec(select(db.AssetRow)).all()
            return [r.model_dump() for r in rows]

    @app.get("/api/assets/{asset_id}")
    def asset_meta(asset_id: str, _=Depends(require_token)):
        from .. import db
        from sqlmodel import select
        with db.get_session() as s:
            row = s.exec(select(db.AssetRow).where(db.AssetRow.asset_id == asset_id)).first()
            if not row:
                raise HTTPException(status_code=404, detail="asset not found")
            return row.model_dump()

    def _asset_rows(types: set[str] | None = None):
        from .. import db
        from sqlmodel import select
        with db.get_session() as s:
            rows = s.exec(select(db.AssetRow)).all()
            if types:
                rows = [r for r in rows if r.asset_type in types]
            return [r.model_dump() for r in rows]

    @app.get("/api/objects")
    def list_objects(_=Depends(require_token)):
        return _asset_rows({"prop", "item", "rotation_sheet"})

    @app.get("/api/objects/{object_id}")
    def get_object(object_id: str, _=Depends(require_token)):
        res = orchestrator.pixellab_asset_response(object_id)
        if res.get("error"):
            raise HTTPException(status_code=404, detail=res["error"])
        return res

    @app.get("/api/map-objects")
    def list_map_objects(_=Depends(require_token)):
        return _asset_rows({"map_object"})

    @app.get("/api/map-objects/{object_id}")
    def get_map_object(object_id: str, _=Depends(require_token)):
        res = orchestrator.pixellab_asset_response(object_id)
        if res.get("error"):
            raise HTTPException(status_code=404, detail=res["error"])
        return res

    @app.get("/api/ui-assets")
    def list_ui_assets(_=Depends(require_token)):
        return _asset_rows({"ui_element", "ui_pack"})

    @app.get("/api/ui-assets/{ui_asset_id}")
    def get_ui_asset(ui_asset_id: str, _=Depends(require_token)):
        res = orchestrator.pixellab_asset_response(ui_asset_id)
        if res.get("error"):
            raise HTTPException(status_code=404, detail=res["error"])
        return res

    @app.get("/api/topdown-tilesets")
    def list_topdown_tilesets(_=Depends(require_token)):
        return _asset_rows({"tileset_top_down", "tiles_pro"})

    @app.get("/api/sidescroller-tilesets")
    def list_sidescroller_tilesets(_=Depends(require_token)):
        return _asset_rows({"tileset_sidescroller"})

    @app.get("/api/isometric-tiles")
    def list_isometric_tiles(_=Depends(require_token)):
        return _asset_rows({"isometric_tile"})

    @app.get("/api/files/{path:path}")
    def serve_file(path: str, _=Depends(require_token)):
        """Serve a generated asset file from the projects root."""
        full = (config.PROJECTS_ROOT / path).resolve()
        try:
            full.relative_to(config.PROJECTS_ROOT.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="outside sandbox")
        if not full.exists():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(str(full))

    @app.post("/api/reveal")
    def reveal_path(body: dict, _=Depends(require_token)):
        """Open a generated-output folder (or a file's parent folder) in the OS
        file browser. Restricted to the projects sandbox."""
        raw = (body or {}).get("path")
        if not raw:
            raise HTTPException(status_code=400, detail="path required")
        target = Path(raw).expanduser().resolve()
        try:
            target.relative_to(config.PROJECTS_ROOT.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="outside sandbox")
        if not target.exists():
            raise HTTPException(status_code=404, detail="path not found")
        folder = target if target.is_dir() else target.parent
        try:
            import subprocess
            import sys
            if sys.platform.startswith("win"):
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"ok": True, "opened": str(folder)}

    @app.post("/api/snap")
    def snap_true_pixels(body: dict, _=Depends(require_token)):
        """True-pixel snapper (spritefusion-style): auto-detect the pixel grid of
        an image and snap it to genuine uniform pixels. Accepts either a sandboxed
        ``path`` (under the projects root) or a base64 ``image`` data URL."""
        import base64
        import io
        import uuid as _uuid
        from PIL import Image as _Image
        from ..pixel.snap import snap_to_true_pixels
        from ..pixel import exporters

        raw_path = (body or {}).get("path")
        data_url = (body or {}).get("image")
        img = None
        if raw_path:
            full = (config.PROJECTS_ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() \
                else Path(raw_path).resolve()
            try:
                full.relative_to(config.PROJECTS_ROOT.resolve())
            except ValueError:
                raise HTTPException(status_code=403, detail="path outside sandbox")
            if not full.exists():
                raise HTTPException(status_code=404, detail="file not found")
            img = _Image.open(str(full)).convert("RGBA")
        elif data_url:
            b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
            try:
                img = _Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"bad image data: {e}")
        else:
            raise HTTPException(status_code=400, detail="provide 'path' or 'image'")

        pixel_size = (body or {}).get("pixel_size")
        k_colors = int((body or {}).get("k_colors") or 0)
        res = snap_to_true_pixels(img, pixel_size=float(pixel_size) if pixel_size else None,
                                  k_colors=k_colors)
        out_dir = (config.PROJECTS_ROOT / "snapped").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        aid = "snap_" + _uuid.uuid4().hex[:10]
        prod = str(out_dir / f"{aid}.png")
        exporters.save_production_png(res.image, prod)
        prev = str(out_dir / f"{aid}_preview_8x.png")
        exporters.save_preview_png(res.image, prev, scale=8)
        return {
            "ok": True,
            "production_png": prod,
            "preview_png": prev,
            "detected_pixel_size": {"x": round(res.pixel_size_x, 2), "y": round(res.pixel_size_y, 2)},
            "output_size": {"width": res.out_width, "height": res.out_height},
            "colors": res.colors,
        }

    # â”€â”€ export â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @app.post("/api/export/pack")
    def export_pack_route(req: ExportPackRequest, _=Depends(require_token)):
        def _wrapped(job):
            return orchestrator.export_pack(job, req.asset_ids, req.engine, req.output_folder)
        job = queue.enqueue("export", req.model_dump(), _wrapped)
        return job.response()

    # â”€â”€ validate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @app.post("/api/validate/asset")
    def validate_asset_route(req: ValidateAssetRequest, _=Depends(require_token)):
        from ..pixel.validate import validate_file
        return validate_file(req.asset_path, req.expected_width,
                             req.expected_height, req.palette_limit).model_dump()

    # â”€â”€ characters / styles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @app.get("/api/characters")
    def list_chars(_=Depends(require_token)):
        return [c.model_dump() for c in orchestrator.get_chars().list_profiles()]

    @app.get("/api/characters/{character_id}")
    def get_char(character_id: str, _=Depends(require_token)):
        c = orchestrator.get_chars().get_profile(character_id)
        if not c:
            raise HTTPException(status_code=404, detail="character not found")
        return c.model_dump()

    @app.get("/api/styles")
    def list_styles(_=Depends(require_token)):
        from .. import db
        from sqlmodel import select
        with db.get_session() as s:
            rows = s.exec(select(db.StyleProfileRow)).all()
            return [{"style_profile_id": r.style_profile_id, "name": r.name,
                     "prompt_prefix": r.prompt_prefix, "negative_prompt": r.negative_prompt}
                    for r in rows]

    # â”€â”€ MCP token helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @app.get("/api/mcp-token")
    def mcp_token(_=Depends(require_token)):
        import sys
        from pathlib import Path as _P
        scripts = _P(sys.executable).parent
        exe = scripts / ("himura-pixel-tools-mcp.exe" if os.name == "nt" else "himura-pixel-tools-mcp")
        stdio_command = str(exe) if exe.exists() else "himura-pixel-tools-mcp"
        return {"token": security.get_or_create_token(),
                "http_url": f"http://127.0.0.1:{_port()}/mcp",
                "stdio_command": stdio_command}

    @app.api_route("/mcp", methods=["GET", "POST"])
    @app.api_route("/mcp/", methods=["GET", "POST"])
    async def mcp_http(request: Request, _=Depends(require_token)):
        """Minimal MCP JSON-RPC endpoint served by the main API process.

        The standalone ``himura-pixel-tools-mcp --transport http`` command still
        exists, but the desktop UI advertises ``127.0.0.1:<api-port>/mcp``. This
        route keeps that advertised URL real and forwards JSON-RPC MCP messages
        to the same tool dispatcher used by stdio.
        """
        if not config.RuntimeConfig.load().mcp_http_enabled:
            raise HTTPException(status_code=404, detail="MCP HTTP is disabled")
        if request.method == "GET":
            from ..mcp import TOOLS
            return {"status": "ok", "server": "himura-pixel-tools", "tools": len(TOOLS)}
        try:
            msg = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON-RPC body")
        from ..mcp.server import _handle_jsonrpc
        resp = await _handle_jsonrpc(msg)
        return JSONResponse(resp or {})

    # â”€â”€ config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @app.get("/api/config")
    def get_config(_=Depends(require_token)):
        return config.RuntimeConfig.load().to_dict()

    @app.post("/api/config")
    def set_config(body: dict, _=Depends(require_token)):
        cfg = config.RuntimeConfig.load()
        for k, v in body.items():
            if k in cfg.__dataclass_fields__:
                setattr(cfg, k, v)
        cfg.save()
        return cfg.to_dict()

    return app


def _port() -> int:
    return int(os.environ.get("HIMURA_PORT", config.DEFAULT_PORT))


app = create_app()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Himura Pixel Tools API server")
    parser.add_argument("--host", default=config.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    os.environ.setdefault("HIMURA_HOST", args.host)
    os.environ.setdefault("HIMURA_PORT", str(args.port))

    # Optionally trigger startup model download.
    if config.RuntimeConfig.load().auto_download_models_at_startup:
        import threading
        def _bg():
            for mid in registry.DEFAULT_DOWNLOAD_SET:
                try:
                    download_models.download_model(mid)
                except Exception:
                    pass
        threading.Thread(target=_bg, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


