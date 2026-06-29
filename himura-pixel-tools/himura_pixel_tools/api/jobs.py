"""In-process job queue with progress tracking.

Honors spec.local_model_loader_spec.vram_policy.max_parallel_jobs_on_rtx3060
(=1): jobs run sequentially through a single worker thread. The API can poll
status or subscribe via SSE for progress.
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .. import config, db
from ..schemas.jobs import JobOutputs, JobResponse, JobStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output_path_strings(outputs: JobOutputs) -> list[str]:
    values: list[str] = []

    def add(value) -> None:
        if value and isinstance(value, str):
            values.append(value)

    for attr in (
        "production_png", "preview_png", "preview_png_8x", "sprite_sheet_png",
        "metadata_json", "gif_preview", "webp_preview", "zip_path",
    ):
        add(getattr(outputs, attr, None))
    for value in outputs.files:
        add(value)
    for value in outputs.final_files:
        add(value)
    for value in outputs.named_files.values():
        add(value)
    for value in outputs.turnaround_pngs.values():
        add(value)
    for value in outputs.extra.values():
        add(value)
    return values


def _safe_project_path(value: str) -> Optional[Path]:
    try:
        path = Path(value).expanduser().resolve()
        path.relative_to(config.PROJECTS_ROOT.resolve())
        return path
    except Exception:
        return None


def _unique_existing_paths(outputs: JobOutputs) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []

    def add(value) -> None:
        if not value or not isinstance(value, str):
            return
        try:
            p = Path(value).expanduser().resolve()
        except Exception:
            return
        key = str(p).lower()
        if key in seen or not p.exists() or not p.is_file():
            return
        seen.add(key)
        paths.append(p)

    for attr in (
        "production_png", "preview_png", "preview_png_8x", "sprite_sheet_png",
        "metadata_json", "gif_preview", "webp_preview", "zip_path",
    ):
        add(getattr(outputs, attr, None))
    for value in outputs.files:
        add(value)
    for value in outputs.named_files.values():
        add(value)
    for value in outputs.turnaround_pngs.values():
        add(value)
    for value in outputs.extra.values():
        add(value)
    return paths


def _request_excludes_saved_outputs(request: dict) -> bool:
    return bool((request or {}).get("exclude_from_saved_outputs"))


def _saved_outputs_enabled(job: "Job") -> bool:
    cfg = config.RuntimeConfig.load()
    return bool(getattr(cfg, "mirror_outputs_to_final", True)) and not _request_excludes_saved_outputs(job.request)


def _mirror_to_final_output(job: "Job") -> None:
    files = _unique_existing_paths(job.outputs)
    if not files:
        return
    dest_dir = (config.PROJECTS_ROOT / "final_output" / job.job_type / job.job_id).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    final_files: list[str] = []
    used_names: set[str] = set()
    for src in files:
        name = src.name
        if name.lower() in used_names:
            stem = src.stem
            suffix = src.suffix
            i = 2
            while f"{stem}_{i}{suffix}".lower() in used_names:
                i += 1
            name = f"{stem}_{i}{suffix}"
        used_names.add(name.lower())
        dst = dest_dir / name
        try:
            shutil.copy2(src, dst)
            final_files.append(str(dst))
        except Exception as exc:
            job.warnings.append(f"final output copy failed for {src.name}: {exc}")

    job.outputs.final_output_folder = str(dest_dir)
    job.outputs.final_files = final_files



def _job_from_row(row: db.JobRow) -> "Job":
    try:
        request = json.loads(row.request_json or "{}")
    except Exception:
        request = {}
    try:
        outputs = JobOutputs(**json.loads(row.result_json or "{}"))
    except Exception:
        outputs = JobOutputs()
    return Job(
        job_id=row.job_id, job_type=row.job_type, request=request,
        status=row.status, progress=row.progress, outputs=outputs,
        error=row.error, created_at=row.created_at, updated_at=row.updated_at,
    )


def _path_in_projects(path: Path) -> bool:
    try:
        path.resolve().relative_to(config.PROJECTS_ROOT.resolve())
        return True
    except Exception:
        return False


def _delete_sibling_output_files(path: Path) -> None:
    if not _path_in_projects(path):
        return
    try:
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)
        parent = path.parent
        if not parent.exists() or not parent.is_dir():
            return
        stem = path.stem
        suffixes = ("", "_preview", "_preview_4x", "_preview_8x", "_metadata")
        for sibling in parent.iterdir():
            if not sibling.is_file():
                continue
            if sibling.stem == stem or any(sibling.stem == f"{stem}{suffix}" for suffix in suffixes):
                sibling.unlink(missing_ok=True)
    except Exception:
        pass


def _delete_output_files(outputs: JobOutputs) -> None:
    for value in _output_path_strings(outputs):
        path = _safe_project_path(value)
        if path:
            _delete_sibling_output_files(path)
    folder = outputs.final_output_folder
    if folder:
        try:
            target = Path(folder).expanduser().resolve()
            target.relative_to((config.PROJECTS_ROOT / "final_output").resolve())
            if target.exists() and target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
        except Exception:
            pass


def _resolve_output_path_set(outputs: JobOutputs) -> set[str]:
    paths: set[str] = set()
    for value in _output_path_strings(outputs):
        path = _safe_project_path(value)
        if path:
            paths.add(str(path).lower())
    return paths


def _delete_asset_rows_for_outputs(outputs: JobOutputs) -> set[str]:
    paths = _resolve_output_path_set(outputs)
    stems = {Path(p).stem.lower() for p in paths}
    character_ids: set[str] = set()
    try:
        with db.get_session() as s:
            rows = s.exec(db.select(db.AssetRow)).all()
            for row in rows:
                row_paths = []
                for raw in [row.production_path, row.preview_path, row.metadata_path]:
                    safe = _safe_project_path(raw) if raw else None
                    if safe:
                        row_paths.append(str(safe).lower())
                matches = (row.asset_id.lower() in stems or any(path in paths for path in row_paths))
                if matches:
                    if row.character_profile_id:
                        character_ids.add(row.character_profile_id)
                    for raw in [row.production_path, row.preview_path, row.metadata_path]:
                        safe = _safe_project_path(raw) if raw else None
                        if safe:
                            _delete_sibling_output_files(safe)
                    s.delete(row)
            s.commit()
    except Exception:
        pass
    return character_ids


def _character_ids_for_job(job: "Job") -> set[str]:
    ids: set[str] = set()
    for source in (job.request or {}, job.outputs.extra or {}):
        value = source.get("character_profile_id") if isinstance(source, dict) else None
        if isinstance(value, str) and value:
            ids.add(value)
    paths = _resolve_output_path_set(job.outputs)
    if paths:
        try:
            with db.get_session() as s:
                for row in s.exec(db.select(db.CharacterProfileRow)).all():
                    lock_paths: list[str] = []
                    try:
                        lock = json.loads(row.reference_lock_json or "{}")
                        lock_paths.extend(lock.get("reference_image_paths") or [])
                    except Exception:
                        pass
                    try:
                        approved = json.loads(row.approved_assets_json or "[]")
                        lock_paths.extend(approved if isinstance(approved, list) else [])
                    except Exception:
                        pass
                    for raw in lock_paths:
                        safe = _safe_project_path(raw) if raw else None
                        if safe and str(safe).lower() in paths:
                            ids.add(row.character_profile_id)
                            break
        except Exception:
            pass
    return ids


def _delete_character_profiles(character_ids: set[str]) -> None:
    if not character_ids:
        return
    try:
        with db.get_session() as s:
            for cid in character_ids:
                char = s.exec(db.select(db.CharacterProfileRow).where(
                    db.CharacterProfileRow.character_profile_id == cid)).first()
                if char:
                    try:
                        lock = json.loads(char.reference_lock_json or "{}")
                        for raw in lock.get("reference_image_paths") or []:
                            safe = _safe_project_path(raw) if raw else None
                            if safe:
                                _delete_sibling_output_files(safe)
                    except Exception:
                        pass
                    s.delete(char)
                assets = s.exec(db.select(db.AssetRow).where(
                    db.AssetRow.character_profile_id == cid)).all()
                for asset in assets:
                    for raw in [asset.production_path, asset.preview_path, asset.metadata_path]:
                        safe = _safe_project_path(raw) if raw else None
                        if safe:
                            _delete_sibling_output_files(safe)
                    s.delete(asset)
            s.commit()
    except Exception:
        pass


def _purge_project_outputs() -> None:
    root = config.PROJECTS_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for child in list(root.iterdir()):
        try:
            child.resolve().relative_to(root)
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception:
            pass
    root.mkdir(parents=True, exist_ok=True)


def _purge_generated_database_rows() -> int:
    count = 0
    try:
        with db.get_session() as s:
            for table in (db.AssetRow, db.CharacterProfileRow, db.JobRow, db.ExportRow):
                rows = s.exec(db.select(table)).all()
                count += len(rows)
                for row in rows:
                    s.delete(row)
            s.commit()
    except Exception:
        pass
    return count


def _delete_job_row(job_id: str) -> None:
    try:
        with db.get_session() as s:
            row = s.exec(db.select(db.JobRow).where(db.JobRow.job_id == job_id)).first()
            if row:
                s.delete(row)
                s.commit()
    except Exception:
        pass


@dataclass
class Job:
    job_id: str
    job_type: str
    request: dict
    status: JobStatus = "queued"
    progress: float = 0.0
    outputs: JobOutputs = field(default_factory=JobOutputs)
    warnings: list[str] = field(default_factory=list)
    model_profile_id: Optional[str] = None
    seed: Optional[int] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    log_lines: list[str] = field(default_factory=list)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def response(self) -> JobResponse:
        return JobResponse(
            job_id=self.job_id, status=self.status, progress=self.progress,
            outputs=self.outputs, warnings=self.warnings,
            model_profile_id=self.model_profile_id, seed=self.seed,
            error=self.error, created_at=self.created_at,
        )

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id, "job_type": self.job_type,
            "status": self.status, "progress": self.progress,
            "outputs": self.outputs.model_dump(),
            "warnings": self.warnings, "model_profile_id": self.model_profile_id,
            "seed": self.seed, "error": self.error,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "logs": list(self.log_lines),
            "saved_outputs_excluded": not _saved_outputs_enabled(self),
        }


JobFn = Callable[[Job], dict]


class JobQueue:
    """Single-worker queue. Concurrent enqueues are fine; execution is serial."""

    def __init__(self):
        self._q: queue.Queue[tuple[str, JobFn]] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._max_parallel = config.RuntimeConfig.load().max_parallel_jobs
        self._worker = threading.Thread(target=self._run, daemon=True, name="himura-worker")
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        try:
            self._q.put_nowait(("__stop", lambda j: {}))
        except Exception:
            pass

    def enqueue(self, job_type: str, request: dict, fn: JobFn) -> Job:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        job = Job(job_id=job_id, job_type=job_type, request=request)
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job, new=True)
        self._q.put((job_id, fn))
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            found = self._jobs.get(job_id)
        if found:
            return found
        try:
            with db.get_session() as s:
                row = s.exec(db.select(db.JobRow).where(db.JobRow.job_id == job_id)).first()
                return _job_from_row(row) if row else None
        except Exception:
            return None

    def list_jobs(self) -> list[Job]:
        jobs: dict[str, Job] = {}
        try:
            with db.get_session() as s:
                for row in s.exec(db.select(db.JobRow)).all():
                    jobs[row.job_id] = _job_from_row(row)
        except Exception:
            pass
        with self._lock:
            jobs.update(self._jobs)
        return sorted(jobs.values(), key=lambda j: j.created_at or "")

    def exclude_all_saved_outputs(self) -> int:
        jobs = list(self.list_jobs())
        db_count = _purge_generated_database_rows()
        _purge_project_outputs()
        with self._lock:
            memory_count = len(self._jobs)
            self._jobs.clear()
        return max(len(jobs), db_count, memory_count)

    def exclude_saved_outputs(self, job_id: str) -> bool:
        j = self.get(job_id)
        if not j:
            return False
        character_ids = _character_ids_for_job(j)
        character_ids.update(_delete_asset_rows_for_outputs(j.outputs))
        _delete_output_files(j.outputs)
        _delete_character_profiles(character_ids)
        _delete_job_row(job_id)
        with self._lock:
            self._jobs.pop(job_id, None)
        return True

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            j = self._jobs.get(job_id)
        if j and j.status in ("queued", "running"):
            j._cancel.set()
            j.status = "cancelled"
            j.updated_at = _now()
            self._persist(j)
            return True
        return False

    def _run(self) -> None:
        while self._running:
            job_id, fn = self._q.get()
            if job_id == "__stop":
                break
            with self._lock:
                job = self._jobs.get(job_id)
            if job is None:
                continue
            try:
                job.status = "running"
                job.updated_at = _now()
                self._persist(job)
                result = fn(job)
                job.outputs = JobOutputs(**(result.get("outputs", {}) if isinstance(result, dict) else {}))
                if isinstance(result, dict):
                    if "production_png" in result and not job.outputs.production_png:
                        job.outputs.production_png = result.get("production_png")
                    if "preview_png" in result and not job.outputs.preview_png:
                        job.outputs.preview_png = result.get("preview_png")
                    if "preview_png_8x" in result and not job.outputs.preview_png_8x:
                        job.outputs.preview_png_8x = result.get("preview_png_8x")
                    if "sprite_sheet_png" in result and not job.outputs.sprite_sheet_png:
                        job.outputs.sprite_sheet_png = result.get("sprite_sheet_png")
                    if "metadata_json" in result and not job.outputs.metadata_json:
                        job.outputs.metadata_json = result.get("metadata_json")
                    if "gif_preview" in result and not job.outputs.gif_preview:
                        job.outputs.gif_preview = result.get("gif_preview")
                    if "webp_preview" in result and not job.outputs.webp_preview:
                        job.outputs.webp_preview = result.get("webp_preview")
                    if "zip_path" in result and not job.outputs.zip_path:
                        job.outputs.zip_path = result.get("zip_path")
                    if "directions" in result and not job.outputs.turnaround_pngs:
                        dirs = result.get("directions") or {}
                        if isinstance(dirs, dict):
                            job.outputs.turnaround_pngs = {str(k): str(v) for k, v in dirs.items() if v}
                    if not job.outputs.files:
                        files = result.get("files") or result.get("output_files") or []
                        if isinstance(files, list):
                            job.outputs.files = [str(v) for v in files if v]
                    if not job.outputs.named_files:
                        named = result.get("named_files") or {}
                        if isinstance(named, dict):
                            job.outputs.named_files = {str(k): str(v) for k, v in named.items() if v}
                    if "seed" in result:
                        job.seed = result["seed"]
                    if "model_profile_id" in result:
                        job.model_profile_id = result["model_profile_id"]
                    if "warnings" in result:
                        job.warnings = result["warnings"]
                    for key in ("asset_id", "character_profile_id", "object_id", "ui_asset_id"):
                        value = result.get(key)
                        if value is not None:
                            job.outputs.extra[key] = str(value)
                if not job.outputs.files:
                    job.outputs.files = [str(p) for p in _unique_existing_paths(job.outputs)]
                if _saved_outputs_enabled(job):
                    _mirror_to_final_output(job)
                else:
                    job.outputs.final_files = []
                    job.outputs.final_output_folder = None
                job.status = "needs_review" if job.warnings and any("review" in w.lower() for w in job.warnings) else "succeeded"
                job.progress = 1.0
            except Exception as e:
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.log_lines.append(traceback.format_exc())
            finally:
                job.updated_at = _now()
                self._persist(job)

    def _persist(self, job: Job, new: bool = False) -> None:
        try:
            with db.get_session() as s:
                row = s.exec(db.select(db.JobRow).where(db.JobRow.job_id == job.job_id)).first()
                payload = dict(
                    job_id=job.job_id, job_type=job.job_type, status=job.status,
                    progress=job.progress, request_json=json.dumps(job.request),
                    result_json=json.dumps(job.outputs.model_dump()),
                    error=job.error, updated_at=job.updated_at,
                )
                if row:
                    for k, v in payload.items():
                        setattr(row, k, v)
                    s.add(row)
                else:
                    s.add(db.JobRow(**payload))
                s.commit()
        except Exception:
            pass


_queue: Optional[JobQueue] = None
_queue_lock = threading.Lock()


def get_queue() -> JobQueue:
    global _queue
    with _queue_lock:
        if _queue is None:
            _queue = JobQueue()
            _queue.start()
        return _queue
