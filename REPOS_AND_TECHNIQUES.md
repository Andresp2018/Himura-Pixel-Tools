# Repos And Techniques

This project combines local generation, game asset post-processing, and desktop tooling. The following open source projects and techniques shape the current architecture.

## Repositories and libraries

- Hugging Face Diffusers: image generation pipelines for SDXL, SD 1.5, and FLUX style workflows.
- Hugging Face Hub: model download, snapshot indexing, and local cache aware fetches.
- PyTorch: tensor runtime and CUDA execution.
- torch-directml: Windows DirectML path for AMD and Intel GPUs.
- Pillow: image loading, drawing, compositing, and export.
- NumPy, OpenCV, scikit-image, scipy: pixel analysis, masks, palette work, and cleanup passes.
- FastAPI and Uvicorn: local API server and browser UI hosting.
- SQLModel: local asset record storage.
- MCP Python SDK: tool bridge for stdio and local HTTP workflows.
- Tauri: native desktop shell around the local web UI.
- stable-diffusion.cpp: planned execution path for GGUF image models such as Ideogram 4 GGUF.

## Techniques used

- True pixel post-processing after model generation.
- Palette extraction and palette locking for consistency.
- Dither options for controlled retro output.
- Segmentation and cleanup passes for transparent sprites.
- Direction specific prompts and negatives for turnaround generation.
- Mirrored direction reuse for matching left and right character poses.
- Seeded batch recipes for reproducible asset packs.
- Local model metadata indexes to track compatibility, source URL, trigger words, and notes.
- Frontend model filtering that surfaces LoRAs while backend validation protects incompatible attachment.

## Why this architecture works

The image model does the broad creative pass. The pixel pipeline then forces production constraints: exact size, transparency, palette limits, metadata, previews, and batch manifests. This split makes the tool practical for game asset iteration because it treats generation as an input to a production pipeline, not the final step.

## Inspiration

This project is an independent, local implementation. Several existing tools and projects inspired its direction and workflow.

- PixelLab inspired the asset focused workflow, including character turnarounds with 4 and 8 directions, object state variants, tilesets, and animation sets. This project rebuilds that kind of workflow to run fully on local hardware.
- Aseprite is the pixel art editor common on game teams. The export formats, sprite sheets, and frame metadata here target that kind of pipeline.
- ComfyUI and the Stable Diffusion web interfaces inspired the local first approach to generation and model management on a user's own GPU.
- ControlNet, IP-Adapter, and AnimateDiff inspired the pose control, reference identity locking, and animation passes.
- BiRefNet and similar background removal projects inspired the transparent sprite cleanup step.
- Retro hardware palettes such as Game Boy, NES, and PICO-8 inspired the optional palette presets for authentic low color output.

These are credited as inspiration only. This project does not bundle, depend on, or redistribute any of them.
