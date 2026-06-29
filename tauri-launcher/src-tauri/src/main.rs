// Himura Pixel Tools desktop launcher.
//
// A thin Tauri (WebView2) shell that:
//   1. locates the Himura Pixel Tools app folder relative to this executable,
//   2. starts the local FastAPI backend (the same one START.BAT launches),
//   3. shows a splash that waits for the backend, then loads the desktop UI,
//   4. stops the backend when the window closes.
//
// It never talks to the network itself; everything stays on 127.0.0.1.

#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;

use tauri::{Manager, WindowEvent};

/// Start the process without a console window (CreateProcess flag). We still
/// give it real stdout/stderr files so Python's sys.stdout is not None.
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Holds the spawned backend process so we can stop it on exit.
struct BackendState(Mutex<Option<Child>>);

/// Candidate app folders, searched from the exe location upward. We check both
/// a sibling `himura-pixel-tools` folder (portable layout) and the directory
/// itself (running from inside a project tree).
fn candidate_dirs(exe_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut cur = Some(exe_dir.to_path_buf());
    let mut depth = 0;
    while let Some(dir) = cur {
        out.push(dir.join("himura-pixel-tools"));
        out.push(dir.clone());
        depth += 1;
        if depth > 10 {
            break;
        }
        cur = dir.parent().map(|p| p.to_path_buf());
    }
    out
}

fn is_app_dir(p: &Path) -> bool {
    p.join("START.BAT").exists()
        || p.join("himura_pixel_tools").join("api").join("server.py").exists()
}

/// Returns the venv python. We prefer python.exe (not pythonw.exe): pythonw has
/// no console, so sys.stdout/stderr are None and the API server crashes at
/// startup. We instead run python.exe with CREATE_NO_WINDOW and redirect output
/// to a log file, which keeps stdout valid and shows no console window.
fn venv_python(app: &Path) -> Option<PathBuf> {
    let scripts = app.join("himura_pixel_tools").join(".venv").join("Scripts");
    for name in ["python.exe", "pythonw.exe"] {
        let p = scripts.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Pick the best app folder: prefer one that already has a working venv, then
/// fall back to any folder that looks like the app (first run / needs setup).
fn find_app_dir(exe_dir: &Path) -> Option<PathBuf> {
    let cands = candidate_dirs(exe_dir);
    for c in &cands {
        if is_app_dir(c) && venv_python(c).is_some() {
            return Some(c.clone());
        }
    }
    for c in &cands {
        if is_app_dir(c) {
            return Some(c.clone());
        }
    }
    None
}

/// Start the backend. If the venv is present we launch the API directly; if not
/// (first run), we open SETUP.BAT in a console so the user can install, and
/// return None so the splash shows its first-run hint.
fn spawn_backend() -> Option<Child> {
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?.to_path_buf();
    let app = find_app_dir(&exe_dir)?;

    if let Some(py) = venv_python(&app) {
        let log_dir = app.join("himura_data").join("logs");
        let _ = std::fs::create_dir_all(&log_dir);
        let mut cmd = Command::new(py);
        cmd.args([
            "-m",
            "himura_pixel_tools.api.server",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
        ])
        .current_dir(&app)
        .creation_flags(CREATE_NO_WINDOW);
        // Give the child real stdout/stderr so Python does not crash on a None
        // stream; keep the output for troubleshooting.
        if let Ok(out) = std::fs::File::create(log_dir.join("launcher_backend.log")) {
            if let Ok(err) = out.try_clone() {
                cmd.stdout(out).stderr(err);
            }
        }
        return cmd.spawn().ok();
    }

    // No venv yet: open the setup script in its own console window.
    let setup = app.join("SETUP.BAT");
    if setup.exists() {
        let _ = Command::new("cmd")
            .args(["/c", "start", "", "cmd", "/k"])
            .arg(setup.as_os_str())
            .current_dir(&app)
            .spawn();
    }
    None
}

fn main() {
    tauri::Builder::default()
        .manage(BackendState(Mutex::new(None)))
        .setup(|app| {
            let child = spawn_backend();
            if let Some(state) = app.try_state::<BackendState>() {
                if let Ok(mut guard) = state.0.lock() {
                    *guard = child;
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                if let Some(state) = window.app_handle().try_state::<BackendState>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(child) = guard.as_mut() {
                            let _ = child.kill();
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running the Himura Pixel Tools launcher");
}
