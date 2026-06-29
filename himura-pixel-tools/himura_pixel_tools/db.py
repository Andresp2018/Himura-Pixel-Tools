"""SQLite database layer for Himura Pixel Tools.

Uses SQLModel to implement every table in ``spec.database_schema.tables``:
projects, models, style_profiles, character_profiles, assets, jobs, exports.

The DB is created lazily on first use under ``config.DB_PATH``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from . import config

_engine = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Table models (spec.database_schema) ───────────────────────────────────────


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    root_path: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class ModelRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: str = Field(index=True, unique=True)
    display_name: str
    type: str
    local_path: str
    source_url: Optional[str] = None
    sha256: Optional[str] = None
    license: Optional[str] = None
    base_compatibility: str = "[]"          # JSON list
    precision: str = "fp16"
    enabled: bool = True
    notes: str = ""
    created_at: str = Field(default_factory=_now)


class StyleProfileRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    style_profile_id: str = Field(index=True, unique=True)
    name: str
    prompt_prefix: str = ""
    negative_prompt: str = ""
    palette_id: Optional[str] = None
    settings_json: str = "{}"
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class CharacterProfileRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    character_profile_id: str = Field(index=True, unique=True)
    name: str
    description: str = ""
    base_seed: int = 0
    style_profile_id: Optional[str] = None
    palette_id: Optional[str] = None
    canonical_size_json: str = '{"width":64,"height":64}'
    traits_json: str = "[]"
    reference_lock_json: str = "{}"
    directions_supported: str = "[4,8]"
    approved_assets_json: str = "[]"
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class AssetRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    character_profile_id: Optional[str] = Field(default=None, index=True)
    asset_id: str = Field(index=True, unique=True)
    asset_type: str
    prompt: str = ""
    seed: Optional[int] = None
    width: int = 0
    height: int = 0
    production_path: Optional[str] = None
    preview_path: Optional[str] = None
    metadata_path: Optional[str] = None
    validation_json: str = "{}"
    created_at: str = Field(default_factory=_now)


class JobRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    job_id: str = Field(index=True, unique=True)
    job_type: str
    status: str = "queued"
    progress: float = 0.0
    request_json: str = "{}"
    result_json: str = "{}"
    logs_path: Optional[str] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class ExportRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    engine: str
    zip_path: str
    asset_ids_json: str = "[]"
    created_at: str = Field(default_factory=_now)


# ── Engine / session management ───────────────────────────────────────────────


def get_engine():
    global _engine
    if _engine is None:
        config.ensure_dirs()
        url = f"sqlite:///{config.DB_PATH}"
        _engine = create_engine(url, connect_args={"check_same_thread": False})
        SQLModel.metadata.create_all(_engine)
        _seed_defaults()
    return _engine


def get_session() -> Session:
    return Session(get_engine())


def _seed_defaults() -> None:
    """Create a default project and default style on first run."""
    with get_session() as s:
        if s.exec(select(Project)).first() is None:
            s.add(Project(name="Default", root_path=str(config.PROJECTS_ROOT / "default")))
        if s.exec(select(StyleProfileRow)).first() is None:
            s.add(StyleProfileRow(
                style_profile_id="style_default_rpg_64",
                name="Default RPG 64",
                prompt_prefix="pixel art, RPG sprite, clean outline, limited palette, game asset",
                negative_prompt=config.DEFAULT_NEGATIVE_PROMPT,
                settings_json=json.dumps({
                    "internal_generation_size": {"width": 768, "height": 768},
                    "default_sampler": "dpm++",
                    "pixel_rules": ["limited palette", "hard edges", "transparent background"],
                }),
            ))
        s.commit()


# ── Helpers (JSON columns <-> python) ─────────────────────────────────────────


def _jload(s: str, default: Any):
    try:
        return json.loads(s) if s else default
    except Exception:
        return default


def _jdump(v: Any) -> str:
    return json.dumps(v, default=str)
