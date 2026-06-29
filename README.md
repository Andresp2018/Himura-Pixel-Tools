# Himura Pixel Tools Version 1

Himura Pixel Tools is a local pixel art asset generator for game production. Version 1 focuses on prompt driven sprites, objects, UI packs, recipes, tiles, animation sheets, model download helpers, and an MCP bridge for tool use.

## What is included

- Local FastAPI backend and browser based desktop UI.
- Model manager with SDXL, SD 1.5, FLUX, and Ideogram 4 GGUF indexing support.
- CUDA setup for NVIDIA GPUs.
- DirectML setup for AMD and Intel GPUs on Windows.
- CPU fallback when no supported GPU backend is available.
- Character turnaround and animation workflows with direction aware prompts and mirrored direction reuse.
- Object workflows with saved object selection or uploaded image state variants.
- Recipe generation for item sets and UI packs.
- MCP server over stdio or local HTTP, with clear messages when the app is not running.
- Source copy of the app in `GITHUB/himura-pixel-tools` (all Python files, setup, and start scripts, without the virtual environment or downloaded models).
- Native desktop launcher source in `GITHUB/tauri-launcher`, built with Tauri.

## Quick start

1. Run `SETUP.BAT` from the repository root.
2. Pick model download from the Models tab or run `.\himura_pixel_tools\.venv\Scripts\python.exe -m himura_pixel_tools.runtime.download_models --all`.
3. Run `START.BAT`.
4. Open `http://127.0.0.1:8765/` if the browser does not open automatically.

To force a backend, set `HIMURA_GPU_BACKEND` before setup:

```bat
set HIMURA_GPU_BACKEND=cuda
SETUP.BAT
```

Valid values are `auto`, `cuda`, `directml`, and `cpu`.

## Ideogram 4 GGUF

The model downloader can index and download `leejet/ideogram-4-GGUF`. The current Diffusers runtime validates the files and stores metadata. Generation from this GGUF bundle needs the planned stable-diffusion.cpp execution runner.

## Repository copy

`GITHUB/himura-pixel-tools` is a clean copy of the application source: all Python
files, templates, static assets, `SETUP.BAT`, `START.BAT`, and the docs. It does
not include the virtual environment, the bundled Python runtime, downloaded
models, or the local database. Run `SETUP.BAT` inside that folder to create a
fresh environment.

## Desktop launcher (Tauri)

The native launcher source is in `GITHUB/tauri-launcher`. It is a small Tauri and
Rust desktop shell that starts the local backend, waits for it, shows the desktop
UI in a WebView2 window, and stops the backend when the window closes. The
frontend is a single static splash page, so no Node or npm is needed.

Build it with the Rust toolchain:

```bat
cd GITHUB\tauri-launcher\src-tauri
cargo build --release
```

The executable is written to
`GITHUB\tauri-launcher\src-tauri\target\release\himura-pixel-tools.exe`. You can
also run `GITHUB\build_launcher.bat`, which builds the launcher and copies the
exe into the `GITHUB` folder next to the source copy.

Recommended portable layout:

```
HimuraPixelTools\
  himura-pixel-tools.exe        the launcher
  himura-pixel-tools\           the app: source plus SETUP.BAT and START.BAT
  python312-runtime\            optional bundled Python for offline setup
```

On first run, if no virtual environment exists yet, the launcher opens
`SETUP.BAT` in a console so you can install, then you can reopen it. After setup
the launcher starts the backend on `127.0.0.1:8765` automatically.

## Version 1 status

Version 1 is usable for local production experiments and asset batch creation. It is not yet a polished consumer installer. The highest value next steps are bundled backend packaging, stronger model compatibility checks, more deterministic character rigs, and preset galleries.

## License

This project is source-available, not open source. See [LICENSE](LICENSE). You may view, run, and modify it for personal, non-commercial use and evaluation. Commercial use, resale, hosting as a service, and redistribution need written permission. Third-party models downloaded by the app keep their own licenses, and no model weights are shipped in this repository.
