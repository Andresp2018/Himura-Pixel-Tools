"""Unit tests for the research-driven upgrades (no torch required):

  - retro palette presets
  - ordered (Bayer) dithering with edge-safe palette mapping
  - rare-colour (outline/highlight) protection in palette extraction
  - the true-pixel pipeline honouring palette_preset + dither + protect_extremes
  - the optional Gemini provider's parsing/guard logic (no network)
  - RuntimeConfig.gemini_enabled gating
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from himura_pixel_tools.pixel import palettes, dither, pixelate
from himura_pixel_tools.pixel.pipeline import PixelPipelineOptions, run_true_pixel_pipeline
from himura_pixel_tools.runtime import external
from himura_pixel_tools import config


# â”€â”€ palette presets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_palette_presets_known():
    assert len(palettes.get_preset("gameboy")) == 4
    assert len(palettes.get_preset("pico8")) == 16
    assert len(palettes.get_preset("c64")) == 16
    assert palettes.get_preset(None) is None
    assert palettes.get_preset("does-not-exist") is None
    # every entry is a 3-channel 0..255 colour
    for pal in palettes.PALETTE_PRESETS.values():
        for c in pal:
            assert len(c) == 3 and all(0 <= v <= 255 for v in c)


# â”€â”€ dithering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _opaque_colors(img):
    arr = np.array(img.convert("RGBA"))
    op = arr[arr[..., 3] >= 16][:, :3]
    return {tuple(c) for c in op}


def test_dither_outputs_only_palette_colors():
    grad = np.tile(np.linspace(0, 255, 64, dtype=np.uint8)[None, :, None], (16, 1, 3))
    img = Image.fromarray(grad, "RGB")
    pal = [[0, 0, 0], [255, 255, 255]]
    out = dither.apply_palette_dithered(img, pal, matrix="bayer4", strength=0.9, edge_safe=False)
    assert out.size == img.size
    colors = _opaque_colors(out)
    assert colors.issubset({(0, 0, 0), (255, 255, 255)})
    # a smooth gradient mapped through a dither should use both palette colours
    assert len(colors) == 2


def test_dither_preserves_alpha():
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[2:6, 2:6] = [120, 120, 120, 255]
    img = Image.fromarray(arr, "RGBA")
    out = dither.apply_palette_dithered(img, palettes.get_preset("gameboy"), strength=0.7)
    a = np.array(out)[..., 3]
    assert set(np.unique(a)).issubset({0, 255})
    assert a[0, 0] == 0 and a[3, 3] == 255


# â”€â”€ rare-colour protection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_protect_extremes_keeps_outline_and_glint():
    arr = np.zeros((1, 104, 3), dtype=np.uint8)
    arr[0, :100] = [128, 128, 128]   # dominant mid grey
    arr[0, 100:102] = [5, 5, 5]      # rare dark outline
    arr[0, 102:104] = [250, 250, 250]  # rare highlight glint
    img = Image.fromarray(arr, "RGB")

    protected = pixelate.extract_palette(img, max_colors=4, protect_extremes=True)
    assert [5, 5, 5] in protected
    assert [250, 250, 250] in protected
    assert len(protected) <= 4


# â”€â”€ pipeline integration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_pipeline_locks_to_preset_with_dither():
    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((128, 128, 3)) * 255).astype("uint8"), "RGB")
    out_dir = str(config.CACHE_ROOT / "test_research")
    res = run_true_pixel_pipeline(
        img, out_dir, "preset test", asset_type="item",
        options=PixelPipelineOptions(
            target_width=32, target_height=32, transparent=False,
            palette_limit=16, palette_preset="gameboy", dither="bayer4",
            protect_extremes=True, generate_preview_scale=[], seed=1),
        seed=1,
    )
    produced = Image.open(res.production_path).convert("RGBA")
    assert produced.size == (32, 32)
    gb = {tuple(c) for c in palettes.get_preset("gameboy")}
    assert _opaque_colors(produced).issubset(gb)
    assert res.validation.exact_size_ok


# â”€â”€ Gemini provider (no network) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_gemini_requires_key():
    try:
        external.generate_image("a knight", api_key="")
        assert False, "expected GeminiError"
    except external.GeminiError:
        pass


def test_gemini_inline_image_parsing():
    import base64
    png = Image.new("RGB", (2, 2), (10, 20, 30))
    import io
    buf = io.BytesIO(); png.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    payload = {"candidates": [{"content": {"parts": [
        {"text": "here you go"},
        {"inlineData": {"mimeType": "image/png", "data": b64}},
    ]}}]}
    raw = external._extract_inline_image(payload)
    assert raw is not None
    assert external._extract_inline_image({"candidates": []}) is None


def test_gemini_compose_prompt_folds_negative():
    p = external._compose_prompt("a sword", "blurry, smooth", 32, 32)
    assert "a sword" in p
    assert "Avoid: blurry, smooth" in p


# â”€â”€ config gating â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_gemini_enabled_gating():
    cfg = config.RuntimeConfig()
    assert cfg.gemini_enabled() is False           # default local
    cfg.generation_provider = "gemini"
    assert cfg.gemini_enabled() is False           # no key yet
    cfg.gemini_api_key = "secret"
    assert cfg.gemini_enabled() is True


# â”€â”€ FLUX loading + adapters + registry fixes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_is_flux2_detection():
    from himura_pixel_tools.runtime import model_loader as ml
    assert ml._is_flux2("unsloth/FLUX.2-klein-4B-GGUF flux-2-klein-4b-Q4_K_M.gguf")
    assert ml._is_flux2("flux2-klein")
    assert not ml._is_flux2("black-forest-labs/FLUX.1-dev")
    assert not ml._is_flux2("stabilityai/stable-diffusion-xl-base-1.0")


def test_flux_pipeline_class_selection():
    from himura_pixel_tools.runtime import model_loader as ml

    class FakeDiffusers:
        Flux2KleinPipeline = "Flux2KleinPipeline"
        Flux2Pipeline = "Flux2Pipeline"
        FluxPipeline = "FluxPipeline"

    fd = FakeDiffusers()
    assert ml._flux_pipeline_class(fd, True) == "Flux2KleinPipeline"
    assert ml._flux_pipeline_class(fd, False) == "FluxPipeline"


def test_flux_ip_adapter_download_filter():
    from himura_pixel_tools.runtime import download_models as dm
    allow, _ = dm._allow_ignore_patterns("ip_adapter", "XLabs-AI/flux-ip-adapter")
    assert any("ip_adapter" in a for a in allow)
    # SDXL ip-adapter still uses the sdxl_models-scoped filter
    allow_sdxl, _ = dm._allow_ignore_patterns("ip_adapter", "h94/IP-Adapter")
    assert any("sdxl_models" in a for a in allow_sdxl)


def test_registry_fixes():
    from himura_pixel_tools.runtime import registry
    types_by_role = {r["role"]: r["type"] for r in registry.REGISTRY}
    # SDXL-Lightning is now a LoRA, not a base
    assert any(r["type"] == "lora" and "Lightning" in r["role"] for r in registry.REGISTRY)
    assert not any(r["type"] == "base" and "Lightning" in r.get("role", "") for r in registry.REGISTRY)
    # FLUX.1 base is available because FLUX.1 LoRAs are not compatible with FLUX.2/Klein.
    assert any(r["type"] == "base" and "FLUX.1" in r["id"] for r in registry.REGISTRY)
    # FLUX controlnet + ip-adapter registered with flux compatibility
    flux_cn = [r for r in registry.REGISTRY if r["type"] == "controlnet"
               and "flux_optional_future" in r.get("base_compatibility", [])]
    flux_ip = [r for r in registry.REGISTRY if r["type"] == "ip_adapter"
               and "flux_optional_future" in r.get("base_compatibility", [])]
    assert flux_cn and flux_ip


# â”€â”€ Civitai download helpers + per-tab LoRA + seam validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_civitai_helpers():
    from himura_pixel_tools.runtime import download_models as dm
    assert dm._is_civitai("https://civitai.com/models/945266/pixel-game-assets-flux-by-dever")
    assert not dm._is_civitai("https://huggingface.co/nerijs/pixel-art-xl")
    assert dm._civitai_type("LORA") == "lora"
    assert dm._civitai_type("Checkpoint") == "base"
    assert dm._civitai_base_compat("Flux.1 D") == ["flux_optional_future"]
    assert dm._civitai_base_compat("SD 1.5") == ["sd15"]
    assert dm._civitai_base_compat("Illustrious") == ["sdxl"]


def test_registry_has_civitai_loras():
    from himura_pixel_tools.runtime import registry
    civitai = [r for r in registry.REGISTRY
               if r["type"] == "lora" and "civitai.com" in r["id"]]
    assert len(civitai) >= 4
    # the requested FLUX assets LoRA is present and tagged flux
    dever = next(r for r in registry.REGISTRY if "945266" in r["id"])
    assert dever["base_compatibility"] == ["flux_optional_future"]


def test_savedoutput_carries_lora_fields():
    from himura_pixel_tools.schemas.jobs import GenerateRequest, TilesetRequest
    g = GenerateRequest(prompt="x", lora_id="nerijs/pixel-art-xl", lora_weight=0.8)
    assert g.lora_id == "nerijs/pixel-art-xl" and g.lora_weight == 0.8
    t = TilesetRequest(description="grass")
    assert t.lora_id is None  # default


def test_seam_score():
    from himura_pixel_tools.pixel import tiles
    # a flat tile repeats perfectly â†’ ~0 seam score
    flat = Image.new("RGB", (16, 16), (40, 120, 60))
    assert tiles.seam_score(flat) < 1.0
    assert tiles.is_seamless(flat)
    # left edge black, right edge white â†’ large mismatch
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[:, -1] = 255
    bad = Image.fromarray(arr, "RGB")
    assert tiles.seam_score(bad) > 12.0
    assert not tiles.is_seamless(bad)


def test_flux2_rejects_flux1_lora_for_attachment():
    from types import SimpleNamespace
    from himura_pixel_tools.api import orchestrator

    class FakeLoader:
        handle = SimpleNamespace(
            model_id="unsloth/FLUX.2-klein-4B-GGUF",
            base_kind="flux_optional_future",
        )

        def _find_index(self, model_id):
            return SimpleNamespace(
                model_id=model_id,
                display_name="Pixel Game Assets FLUX",
                source_url="https://civitai.com/models/945266/pixel-game-assets-flux-by-dever",
                local_path="models/lora/pixel-game-assets-flux",
                trigger="dvr-pixel-flux",
                notes="FLUX.1-dev pixel LoRA",
                base_compatibility=["flux_optional_future"],
            )

    assert orchestrator._lora_base_compatible(FakeLoader(), "dever-flux1") is False


def test_flux1_accepts_flux1_lora_for_attachment():
    from types import SimpleNamespace
    from himura_pixel_tools.api import orchestrator

    class FakeLoader:
        handle = SimpleNamespace(
            model_id="black-forest-labs/FLUX.1-dev",
            base_kind="flux_optional_future",
        )

        def _find_index(self, model_id):
            return SimpleNamespace(
                model_id=model_id,
                display_name="Pixel Game Assets FLUX",
                source_url="https://civitai.com/models/945266/pixel-game-assets-flux-by-dever",
                local_path="models/lora/pixel-game-assets-flux",
                trigger="dvr-pixel-flux",
                notes="FLUX.1-dev pixel LoRA",
                base_compatibility=["flux_optional_future"],
            )

    assert orchestrator._lora_base_compatible(FakeLoader(), "dever-flux1") is True


def test_ideogram_registry_and_download_filters():
    from himura_pixel_tools.runtime import download_models as dm
    from himura_pixel_tools.runtime import registry

    item = registry.by_id("https://huggingface.co/leejet/ideogram-4-GGUF/tree/main")
    assert item is not None
    assert item["base_compatibility"] == ["ideogram_gguf"]
    allow, _ = dm._allow_ignore_patterns("base", "leejet/ideogram-4-GGUF")
    assert "*.gguf" in allow
    assert dm._base_compat_for("leejet/ideogram-4-GGUF", "base") == ["ideogram_gguf"]


def test_ui_pack_tiny_renderer_no_inverted_rectangles():
    from himura_pixel_tools.api import orchestrator

    img = orchestrator._draw_pixel_ui("health bar", 12, 8, "tiny")
    assert img.size == (12, 8)

# -- batch recipes -------------------------------------------------------------

def test_batch_recipe_schema_defaults():
    from himura_pixel_tools.schemas.jobs import BatchRecipeRequest
    req = BatchRecipeRequest(theme="crystal forest", lora_id="some/lora")
    assert req.recipe == "item_set"
    assert req.count == 6
    assert req.animations == ["idle", "walk"]
    assert req.lora_id == "some/lora"


def test_batch_recipe_aggregates_mocked_item_outputs(monkeypatch):
    from himura_pixel_tools import config
    from himura_pixel_tools.api import orchestrator
    from himura_pixel_tools.schemas.jobs import BatchRecipeRequest

    project_root = config.CACHE_ROOT / "test_batch_recipe_projects"
    project_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "PROJECTS_ROOT", project_root)

    class Job:
        progress = 0.0
        warnings = []
        log_lines = []

    def fake_generate(job, req):
        stem = req.prompt.split()[0] + "_" + str(req.seed)
        prod = str(project_root / f"{stem}.png")
        meta = str(project_root / f"{stem}.json")
        return {
            "asset_id": stem,
            "production_png": prod,
            "metadata_json": meta,
            "outputs": {
                "production_png": prod,
                "metadata_json": meta,
                "files": [prod, meta],
                "named_files": {"production": prod, "metadata": meta},
            },
        }

    monkeypatch.setattr(orchestrator, "generate_asset", fake_generate)
    res = orchestrator.create_batch_recipe(
        Job(), BatchRecipeRequest(
            recipe="item_set", theme="crystal", count=2, size=32,
            seed=10, output_folder="batch_test", lora_id="test/lora"))

    outputs = res["outputs"]
    assert outputs["production_png"].endswith(".png")
    assert outputs["metadata_json"].endswith("_manifest.json")
    assert Path(outputs["metadata_json"]).exists()
    assert len([p for p in outputs["files"] if p.endswith(".png")]) == 2
    assert "item_01_sword_production" in outputs["named_files"]


def test_prompt_keeps_pixel_style_before_long_user_text():
    long_prompt = " ".join(f"descriptor{i}" for i in range(120))
    prompt = config.build_prompt(long_prompt, "character")
    assert prompt.startswith("pixel art, game sprite, limited palette, hard edges")
    assert "descriptor0" in prompt
    assert "descriptor80" not in prompt
    assert "plain background" in prompt


def test_controlnet_validation_rejects_incomplete_openpose(monkeypatch):
    import json
    import uuid
    from himura_pixel_tools.runtime.model_loader import ModelLoader

    root = config.CACHE_ROOT / f"test_controlnet_validation_{uuid.uuid4().hex[:8]}"
    bad = root / "controlnet" / "thibaud__controlnet-openpose-sdxl-1.0"
    good = root / "controlnet" / "xinsir__controlnet-openpose-sdxl-1.0"
    bad.mkdir(parents=True)
    good.mkdir(parents=True)
    (bad / "config.json").write_text(json.dumps({"model": "openpose"}), encoding="utf-8")
    (good / "diffusion_pytorch_model.safetensors").write_bytes(b"safe weights placeholder")
    monkeypatch.setattr(config, "MODELS_ROOT", root)

    loader = ModelLoader()
    bad_report = loader.validate_model_files("thibaud__controlnet-openpose-sdxl-1.0")
    good_report = loader.validate_model_files("xinsir__controlnet-openpose-sdxl-1.0")
    assert bad_report.valid is False
    assert "diffusion_pytorch_model.safetensors" in bad_report.files_missing
    assert good_report.valid is True


def test_ui_pack_recipe_uses_deterministic_ui_renderer(monkeypatch):
    import json
    import uuid
    from PIL import Image
    from himura_pixel_tools.api import orchestrator
    from himura_pixel_tools.schemas.jobs import BatchRecipeRequest

    project_root = config.CACHE_ROOT / f"test_ui_recipe_projects_{uuid.uuid4().hex[:8]}"
    project_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "PROJECTS_ROOT", project_root)
    monkeypatch.setattr(orchestrator, "_record_file_asset", lambda *args, **kwargs: None)

    class Job:
        progress = 0.0
        warnings = []
        log_lines = []

    res = orchestrator.create_batch_recipe(
        Job(), BatchRecipeRequest(
            recipe="ui_pack", theme="crystal forest RPG", count=2,
            size=64, seed=7, output_folder="ui_recipe_test"))

    pngs = [Path(p) for p in res["outputs"]["files"] if str(p).endswith(".png") and "preview" not in str(p)]
    metas = [Path(p) for p in res["outputs"]["files"] if str(p).endswith(".json") and not str(p).endswith("_manifest.json")]
    assert len(pngs) == 2
    assert Image.open(pngs[0]).size == (256, 128)
    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    assert meta["asset_type"] == "ui_element"
    assert meta["extra"]["deterministic_ui"] is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

