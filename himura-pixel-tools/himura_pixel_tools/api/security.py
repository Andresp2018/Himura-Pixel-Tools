"""Security helpers: bearer-token auth + filesystem sandboxing.

Implements spec.mcp_server_spec.mcp_security:
  - Require bearer token for HTTP transport.
  - Sandbox output paths to explicit project asset folders.
  - Bind HTTP server to 127.0.0.1 by default.
  - Never expose shell tools.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from .. import config


def get_or_create_token() -> str:
    """Return the persistent MCP bearer token (created on first launch)."""
    if config.TOKEN_PATH.exists():
        tok = config.TOKEN_PATH.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = secrets.token_urlsafe(32)
    config.ensure_dirs()
    config.TOKEN_PATH.write_text(tok, encoding="utf-8")
    return tok


def check_token(provided: str | None) -> bool:
    if not config.RuntimeConfig.load().mcp_require_token:
        return True
    if not provided:
        return False
    expected = get_or_create_token()
    return secrets.compare_digest(provided or "", expected)


def resolve_output_folder(requested: str | None, project_name: str = "default") -> Path:
    """Resolve and sandbox an output folder under the projects root.

    Refuses absolute paths outside the projects root; relative paths are
    joined under the project folder. Always returns an absolute path inside
    the sandbox.
    """
    root = config.PROJECTS_ROOT.resolve()
    if requested:
        p = Path(requested).expanduser()
        # If absolute, must already live under the projects root (or be allowed)
        if p.is_absolute():
            try:
                rel = p.resolve().relative_to(root)
                out = root / rel
            except ValueError:
                # outside sandbox → place under project dir with a sanitized name
                safe = "_".join(str(p).replace(":", "").replace("\\", "/").split("/")[-3:])
                out = root / project_name / safe
        else:
            out = (root / project_name / requested).resolve()
            try:
                out.relative_to(root)
            except ValueError:
                out = root / project_name
    else:
        out = root / project_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def assert_path_inside_sandbox(path: str | Path, sandbox: Path) -> Path:
    """Raise if ``path`` resolves outside ``sandbox``. Returns resolved path."""
    p = Path(path).expanduser().resolve()
    sandbox = Path(sandbox).resolve()
    try:
        p.relative_to(sandbox)
    except ValueError as e:
        raise PermissionError(f"path {p} is outside sandbox {sandbox}") from e
    return p
