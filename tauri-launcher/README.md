# Himura Pixel Tools Launcher (Tauri)

This is a small native desktop shell for Himura Pixel Tools. It starts the local
Python backend, waits for it to come up, and then shows the desktop UI in a
WebView2 window. When you close the window it stops the backend.

It is a pure Rust and Tauri project. The frontend is a single static splash page
in `dist/index.html`, so you do not need Node or npm to build it.

## Requirements

- Windows 10 or 11 with the WebView2 runtime (already present on current Windows).
- Rust toolchain (`rustup`, stable, MSVC host). Install from https://rustup.rs.
- The Himura Pixel Tools app folder available next to the built executable, or in
  a parent folder. See "How it finds the app" below.

## Build

From this folder:

```bat
cd src-tauri
cargo build --release
```

The executable is written to:

```
src-tauri\target\release\himura-pixel-tools.exe
```

You can also run `build_launcher.bat` in the `GITHUB` folder, which builds the
launcher and copies the exe next to the portable app folder.

## How it finds the app

On launch the exe searches itself and its parent folders for either:

- a sibling `himura-pixel-tools` folder that contains `START.BAT`, or
- a folder that contains `START.BAT` directly.

It prefers a folder that already has a working virtual environment at
`himura_pixel_tools\.venv`. If none has a venv yet, it opens `SETUP.BAT` in a
console for the first run, then you can reopen the launcher.

Recommended portable layout:

```
HimuraPixelTools\
  himura-pixel-tools.exe        (this launcher)
  himura-pixel-tools\           (the app: source + SETUP.BAT + START.BAT)
  python312-runtime\            (optional bundled Python for offline setup)
```

## Notes

- Everything stays on `127.0.0.1`. The launcher never makes network calls itself.
- The backend is the same FastAPI server that `START.BAT` runs, on port 8765.
- If the backend takes a while on first run while models load, the splash keeps
  waiting and shows a hint after a short delay.
