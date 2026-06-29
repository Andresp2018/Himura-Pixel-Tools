"""Tests for model loader validation logic + schemas (no torch needed)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json

from himura_pixel_tools.schemas.models import ModelIndex, VRAMReport, ValidationReport
from himura_pixel_tools.schemas.characters import CharacterProfile, ReferenceLock, StyleProfile
from himura_pixel_tools.schemas.jobs import GenerateRequest, JobResponse, TargetSize, ObjectStateRequest
from himura_pixel_tools.runtime.model_loader import ModelLoader, _detect_base_compat


def test_model_index_roundtrip():
    m = ModelIndex(model_id="stabilityai/stable-diffusion-xl-base-1.0",
                   display_name="SDXL Base", type="base",
                   local_path="/models/base/sdxl", base_compatibility=["sdxl"])
    d = m.model_dump()
    m2 = ModelIndex(**d)
    assert m2.model_id == m.model_id


def test_detect_base_compat():
    assert _detect_base_compat("stabilityai/stable-diffusion-xl-base-1.0") == ["sdxl"]
    assert _detect_base_compat("runwayml/stable-diffusion-v1-5") == ["sd15"]
    assert _detect_base_compat("black-forest-labs/FLUX.1-dev") == ["flux_optional_future"]
    assert _detect_base_compat("flux-2-klein-4b-Q4_K_M.gguf") == ["flux_optional_future"]
    assert _detect_base_compat("leejet/ideogram-4-GGUF") == ["ideogram_gguf"]


def test_loader_scan_finds_gguf_base(tmp_path, monkeypatch):
    import himura_pixel_tools.config as cfg
    monkeypatch.setattr(cfg, "MODELS_ROOT", tmp_path)
    base_dir = tmp_path / "base"
    base_dir.mkdir(parents=True)
    gguf = base_dir / "flux-2-klein-4b-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    loader = ModelLoader()
    found = loader.scan_model_store()
    item = next(m for m in found if m.local_path == str(gguf))
    assert item.type == "base"
    assert item.base_compatibility == ["flux_optional_future"]
    report = loader.validate_model_files(item.model_id)
    assert report.valid is True
    assert gguf.name in report.files_present


def test_loader_scan_finds_indexed(tmp_path, monkeypatch):
    # point the model store at a temp dir with one indexed model
    import himura_pixel_tools.config as cfg
    monkeypatch.setattr(cfg, "MODELS_ROOT", tmp_path)
    base = tmp_path / "base" / "my__model"
    base.mkdir(parents=True)
    (base / "himura.json").write_text(json.dumps({
        "model_id": "my/model", "display_name": "My Model", "type": "base",
        "local_path": str(base), "base_compatibility": ["sdxl"],
    }))
    loader = ModelLoader()
    found = loader.scan_model_store()
    ids = [m.model_id for m in found]
    assert "my/model" in ids


def test_loader_validate_missing(tmp_path, monkeypatch):
    import himura_pixel_tools.config as cfg
    monkeypatch.setattr(cfg, "MODELS_ROOT", tmp_path)
    loader = ModelLoader()
    report = loader.validate_model_files("does-not-exist")
    assert report.valid is False
    assert report.errors


def test_vram_report_shape():
    r = VRAMReport(device="cpu", device_name="cpu", cuda_available=False)
    assert r.active_adapters == []


def test_character_profile_defaults():
    p = CharacterProfile(character_id="c1", name="Knight", description="x", base_seed=1,
                         style_profile_id="s1")
    assert "idle" in p.animation_templates
    assert "palette" in p.visual_locked_traits
    assert p.canonical_size["width"] == 64


def test_object_state_accepts_upload_without_saved_object():
    req = ObjectStateRequest(edit_description="mossy", source_image="data:image/png;base64,AAAA")
    assert req.object_id is None
    assert req.source_image.startswith("data:image/png")

def test_generate_request_defaults():
    req = GenerateRequest(prompt="sword")
    assert req.target_size.width == 64
    assert req.generate_preview_scale == [4, 8]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))


