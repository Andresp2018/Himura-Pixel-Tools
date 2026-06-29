# Limitations Q&A

## Can it run fully offline?

Generation can run offline after models are installed. Model downloads, package installation, and first setup need internet access.

## Does every listed model generate inside the app today?

No. SDXL, SD 1.5, and supported FLUX paths are handled by the current runtime. Ideogram 4 GGUF is downloadable and indexed, but generation needs the planned stable-diffusion.cpp runner.

## Why do some LoRAs show a compatibility note?

FLUX.1 and FLUX.2 or Klein LoRAs are not always interchangeable. The UI lets users select them so they are visible and manageable, while the backend blocks unsafe attachment when the active base is incompatible.

## Why can character directions still vary?

The app now mirrors matching right facing directions from left facing outputs when possible. New model generated directions can still drift because image diffusion is not a rigged character system. Better pose locks, reference adapters, and skeleton controlled passes are planned.

## Can AMD and Intel GPUs run it?

On Windows, setup can choose DirectML for AMD and Intel GPUs. CUDA remains the best tested path for NVIDIA. CPU works as a fallback but is slower.

## Is the Tauri folder a finished installer?

No. It is a source build scaffold for the desktop shell. A full installer should bundle Python, the virtual environment bootstrap, the backend launcher, and model storage permissions.

## What are the main risks for Version 1?

- Model licenses and usage rules vary by checkpoint.
- GGUF generation needs a separate runner path.
- Animation quality depends heavily on model, prompt, seed, and reference quality.
- Large model downloads can fail on unstable connections.
- CPU fallback is practical for testing but slow for production generation.
