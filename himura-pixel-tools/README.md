# Himura Pixel Tools

A **self-contained, fully local** pixel-art game-asset generator: text-to-sprite,
character consistency, animation, tilesets, exact-size true-pixel export, and a
local MCP server â€” all running on your own GPU. **No ComfyUI, no cloud
generation, no external workflow UI.**

Built from `himura_pixel_tools_engineer_spec_no_comfyui.json`. Diffusers/PyTorch
are used as libraries inside the app's own model loader, pipeline orchestration,
job system, UI, and export logic.

---

## What it does

- **Exact-size true-pixel export** â€” production PNG is *always* the requested
  canvas (16Ã—16 â€¦ 128Ã—128 or custom). Previews are nearest-neighbor upscales.
  Alpha is snapped to {0,255}; palette is quantized; orphans cleaned up.
- **Character consistency** â€” `CharacterProfile` + `ReferenceLock` keep a
  persistent identity (palette, silhouette hash, seed pack, IP-Adapter
  reference) across directions, animations, and inpaint edits.
- **Animation** â€” idle/walk/run/attack/hurt/death/cast, 4 or 8 directions,
  aligned frames, fixed pivot, sprite-sheet + Aseprite/Unity/Godot/Phaser
  metadata + GIF/WebP previews.
- **Tilesets** â€” top_down / sidescroller / isometric / Wang, exact tile size.
- **Recipes** - one-click item sets, UI packs, tileset packs, and character packs
  using shared seed/LoRA settings.
- **Local MCP** â€” 10 safe tools exposed to Claude Code, Google Antigravity, and
  OpenAI Codex over stdio + Streamable HTTP, bearer-token protected.
- **Model management** â€” scan / validate (SHA-256) / load / hot-swap adapters
  with VRAM policy tuned for 12 GB cards (one base resident, sequential jobs).

## Architecture (ACT)

| Layer | Process | Stack |
|-------|---------|-------|
| **A** Application | Desktop web UI | HTML/CSS/JS served by FastAPI |
| **C** Control | `himura-pixel-tools-api` | FastAPI + SQLite + job queue + MCP bridge |
| **T** Tool runtime | `himura-pixel-tools-runtime` | PyTorch + Diffusers + Pillow/NumPy pixel tools |

Single local desktop app launches a local backend + model runtime sidecar on
`127.0.0.1:8765` only.

---

## Quick start

### 1. Setup (creates the venv, installs everything)

**Windows** (PowerShell):
```powershell
cd himura_pixel_tools
.\setup_windows.ps1
```
â€¦or just double-click **`himura.bat`** (runs setup on first launch, then starts the app).

**Linux / macOS**:
```bash
cd himura_pixel_tools
./setup.sh
```

The setup script creates a `.venv/`, installs PyTorch for the selected backend,
then installs the runtime dependencies and the package itself in editable mode. It offers
to **download the recommended models** (SDXL base + fp16-fix VAE) into the local
model store. You can defer this and download later.

### 2. Download models

At setup time (offered automatically), or any time:

```bash
# full recommended set
.venv/bin/python -m himura_pixel_tools.runtime.download_models --all
# minimal fast set (base + vae + LCM-LoRA)
.venv/bin/python -m himura_pixel_tools.runtime.download_models --fast
# specific models (by id or full huggingface.co URL)
.venv/bin/python -m himura_pixel_tools.runtime.download_models stabilityai/stable-diffusion-xl-base-1.0
# list the registry / installed
.venv/bin/python -m himura_pixel_tools.runtime.download_models --list
.venv/bin/python -m himura_pixel_tools.runtime.download_models --installed
```

You can also enable **auto-download at startup** in the UI â†’ Settings, or from
the Models tab.

### 3. Run

**Windows:** `START.BAT` (double-click) or `.\start_windows.ps1`
**Linux/macOS:** `./start.sh`

Opens the desktop UI at `http://127.0.0.1:8765/` and starts the API.

---

## Using a custom / different diffusion model

You can use **any** Stable Diffusion model (SD 1.5, SDXL, or any community
checkpoint) in two ways:

### Option A â€” Drop it in the folder

1. Find the model store (shown at the top of the **Models** tab, default:
   `himura_pixel_tools\himura_data\models\`).
2. Drop your file:
   - A **single-file checkpoint** (`*.safetensors`) â†’ into `models\base\`
   - A **LoRA** â†’ into `models\lora\`
   - A **VAE / ControlNet / IP-Adapter** â†’ into the matching subfolder
3. Click **Refresh** on the Models tab â€” it's detected automatically.
4. Select it in **Active base model â†’ Set active**.

### Option B â€” Paste a link in the UI

1. Open the **Models** tab â†’ "Download a model individually".
2. Paste a Hugging Face **repo id** (`org/name`) or a **direct file URL**
   (`https://huggingface.co/org/name/resolve/main/file.safetensors`).
3. Pick a type (or leave on *auto-detect*), click **Download**.
4. Progress is polled live; when done, set it active.

The type is **auto-detected** from the name (lora / vae / controlnet / â€¦), and
the download is filtered to pull **only fp16 safetensors** (no fp32 / .bin
bloat). Custom base models are loaded via Diffusers' `from_single_file` for
standalone checkpoints, or `from_pretrained` for Diffusers folders.

---

## Optional: Google Gemini as the image source

Everything ships **fully local** by default. If you'd rather generate the
source image with Google Gemini *instead of* the local SDXL/FLUX model, paste a
Gemini API key in **Settings**:

- **Generation provider** â†’ `gemini`
- **Gemini API key** â†’ your key (from Google AI Studio)
- **Gemini model** â†’ defaults to `gemini-2.5-flash-image`

This is the only non-local path and it is strictly opt-in: it is used only when
the provider is `gemini` **and** a key is present. The local true-pixel pipeline
(segmentation, grid-snap, palette/dither, export) still runs on your machine, so
output is the same exact-size pixel art. If a Gemini call fails the job
transparently falls back to the local model. Only your prompt is sent to Google,
and only when this is enabled.

## Pixel-art LoRAs (per-tab selector + Civitai)

- **Per-tab LoRA selector:** every generating tab (Generate, Objects, UI,
  Advanced Tiles, Characters, Animation, Tilesets) has a **Style LoRA** dropdown.
  It only lists LoRAs compatible with the **currently active base** (SDXL LoRAs
  for an SDXL base, FLUX LoRAs for a FLUX base), and overrides the global default.
  FLUX.1 and FLUX.2/Klein LoRAs are also separated because their transformer
  layer shapes are not interchangeable.
- **Global default:** set **Default pixel-art LoRA id** in Settings as a fallback
  for tabs left on "(none / global default)".
- **Civitai downloads:** paste a `https://civitai.com/models/...` URL in the
  **Models** tab to download a LoRA/checkpoint straight from Civitai (the type and
  base compatibility are read from the Civitai API). Some Civitai files need a
  token â€” set **Civitai API key** in Settings. The recommended registry already
  includes several Civitai pixel LoRAs (Dever FLUX game assets, Perfect PixelArt
  x8, Character Sprite generator, 2D Pixel Toolkit) plus `nerijs/pixel-art-xl`.

## Using the desktop UI

- **Generate** â€” prompt â†’ exact-size item/character/UI/background, with size
  presets, palette limit, seed, transparent background, fast draft toggle, and
  **retro palette presets** (Game Boy / NES / C64 / PICO-8 â€¦), **Bayer
  dithering**, and **outline/highlight color protection**.
- **Characters** â€” create a persistent character, then generate 4/8-direction
  turnarounds.
- **Animation** â€” pick a character + animation template â†’ aligned sprite sheet.
- **Tilesets** â€” seamless tiles at exact dimensions.
- **Models** â€” VRAM report, **download each model/LoRA individually**, paste any
  HF link, set the active base model, remove models.
- **Jobs** â€” queue status, progress, outputs.
- **Settings** â€” precision, VRAM policy, MCP token, auto-download.


## Using MCP (Claude / Codex / Antigravity)

**Full step-by-step guide: [`MCP_SETUP.md`](MCP_SETUP.md)** â€” covers Claude Code,
Claude Desktop, OpenAI Codex and Google Antigravity, with both stdio and HTTP
transports. Quick version:

```bash
# get your token (after starting the app) â€” or copy it from the Settings tab
# Windows: Get-Content "E:\Himura Pixel Tools\himura_data\mcp_token"

# Claude Code (HTTP)
claude mcp add --transport http himura-pixel-tools http://127.0.0.1:8765/mcp \
  --header "Authorization: Bearer <token>"

# Claude Code (stdio â€” no token needed)
claude mcp add himura-pixel-tools -- \
  "E:\Himura Pixel Tools\himura_pixel_tools\.venv\Scripts\himura-pixel-tools-mcp.exe" --transport stdio
```

Tools: `generate_asset` (item/prop/character/**building**/ui/background),
`create_character`, `generate_turnaround`, `generate_portrait`,
`animate_character`, `inpaint_asset`, `create_tileset`, `export_pack`,
`get_job_status`, `list_models`, `validate_asset`. The REST API also exposes
`/api/jobs/batch-recipe` for UI/item/tile/character packs. Only these high-level tools
are exposed â€” never shell tools. Output paths are sandboxed.

> The MCP bridge forwards to the local API, so **keep the app (START.BAT)
> running** while your agent uses these tools.

## REST API

Base: `http://127.0.0.1:8765` Â· docs at `/api/docs`. Bearer token required
on all `/api/*` routes except `/api/health`. Full endpoint list in the spec's
`api_contract`; key ones: `POST /api/jobs/generate`, `/api/jobs/create-character`,
`/api/jobs/turnaround`, `/api/jobs/animate-character`, `/api/jobs/inpaint`,
`/api/jobs/tileset`, `/api/export/pack`, `/api/validate/asset`.

## Data layout

Everything lives **inside the project folder** â€” fully self-contained and
portable. Nothing is written to `%LOCALAPPDATA%` or `~/.cache`.

```
himura_pixel_tools/
â”œâ”€â”€ .venv/                      # the virtualenv (created by SETUP)
â””â”€â”€ himura_data/                # ALL app data + models (default; override with HIMURA_DATA_DIR)
    â”œâ”€â”€ models/{base,vae,lora,controlnet,ip_adapter,motion,segmentation,upscale}/
    â”œâ”€â”€ hf_cache/               # Hugging Face cache (redirected here, not ~/.cache)
    â”œâ”€â”€ projects/<name>/â€¦       # generated assets, sandboxed
    â”œâ”€â”€ cache/  logs/  himura.db  config.json  mcp_token
```

To move everything to another drive, set `HIMURA_DATA_DIR` before starting:

```bat
set HIMURA_DATA_DIR=E:\HimuraData
START.BAT
```

## Tests

```bash
python -m pytest tests/ -v     # pure-Python pixel + schema tests, no torch needed
```

## Project layout

```
himura_pixel_tools/
â”œâ”€â”€ api/          control layer: FastAPI server, job queue, auth, sandboxing, orchestrator
â”œâ”€â”€ runtime/      model loader, pipelines, conditioning, model registry + downloader
â”œâ”€â”€ pixel/        true-pixel pipeline: pixelate, masks, cleanup, validate, spritesheet, exporters
â”œâ”€â”€ characters.py character consistency system (profiles, reference lock, animation)
â”œâ”€â”€ schemas/      Pydantic models (models, characters, jobs, pixel)
â”œâ”€â”€ mcp/          MCP server (stdio + Streamable HTTP), 10 tools
â”œâ”€â”€ desktop/      web UI (templates + static css/js)
â”œâ”€â”€ db.py         SQLModel tables (projects, models, styles, characters, assets, jobs, exports)
â”œâ”€â”€ config.py     paths, runtime config, exact-size targets, asset types
â”œâ”€â”€ requirements.txt  pyproject.toml  setup_*.sh/.ps1  start_*.sh/.ps1  himura.bat
examples/mcp/     client config examples (Claude Code, Antigravity, Codex)
tests/            unit tests
```

## Notes

- Designed for an **RTX 3060 12 GB**: one base model resident, adapters
  hot-swapped per job, sequential generation, CPU offload when low on VRAM.
- SDXL is the default base; the fp16-fix VAE is used to avoid NaN issues.
- **FLUX**: loading a `.gguf` model needs the `gguf` package (now in
  `requirements.txt`; `pip install gguf` if you set up before this). FLUX.2-klein
  uses the `Flux2KleinPipeline` (Qwen3 text encoder) â€” on first load the app pulls
  the text-encoder/VAE from `black-forest-labs/FLUX.2-klein-4B` (~8 GB, one-time).
  FLUX.1 LoRAs require a FLUX.1 base such as `black-forest-labs/FLUX.1-dev`;
  they are not compatible with FLUX.2/Klein transformer shapes.
  ControlNet and IP-Adapter now work on FLUX too (FLUX union ControlNet + XLabs
  FLUX IP-Adapter); install them from the **Models** tab (FLUX stack).
- **SDXL-Lightning is not a base model** â€” it ships only accelerator UNet/LoRA
  files. Install it as the 4-step LoRA (Models tab â†’ FLUX/SDXL stack) and apply it
  on top of an SDXL base, rather than selecting it as the active base.
- Character consistency relies on IP-Adapter reference locks + manual approval;
  the spec is explicit that this is an approximation, not a copy of any
  proprietary system. PixelLab is used only as a public feature benchmark.
- License: MIT.

