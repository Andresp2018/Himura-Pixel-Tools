# Connecting Himura Pixel Tools to an AI agent (MCP)

Himura Pixel Tools ships a local **MCP server** that exposes its safe, high‑level
asset‑generation tools to AI coding agents — **Claude** (Claude Code & Claude
Desktop), **OpenAI Codex**, and **Google Antigravity**. The agent can then create
characters, turnarounds, portraits, animations, tilesets, buildings and export
packs for you, entirely on your own GPU.

> Everything runs on `127.0.0.1`. Nothing is sent to the cloud. The HTTP
> transport is protected by a bearer token; the stdio transport runs as a local
> child process.

---

## 0. Prerequisites

1. **Install once** (creates the venv): run `SETUP.BAT`.
2. **Start the app** (starts the API the MCP bridge talks to): run `START.BAT`.
   - The desktop UI opens at <http://127.0.0.1:8765/>.
   - **The API must be running** for the MCP tools to work — the MCP server is a
     thin bridge that forwards each tool call to this local API.

### Where things live (Windows)

| Thing | Path |
|---|---|
| MCP server executable | `E:\Himura Pixel Tools\himura_pixel_tools\.venv\Scripts\himura-pixel-tools-mcp.exe` |
| Bearer token (for HTTP transport) | `E:\Himura Pixel Tools\himura_data\mcp_token` |
| HTTP MCP endpoint (served by the running app) | `http://127.0.0.1:8765/mcp` |

Get your token from the **Settings** tab in the app (there's a *Copy* button), or:

```powershell
Get-Content "E:\Himura Pixel Tools\himura_data\mcp_token"
```

There are two transports — pick whichever your agent supports best:

- **stdio** – the agent launches the MCP server as a child process. No token
  needed (it reads the token file itself). The app/API still has to be running.
- **HTTP** – the agent connects to `http://127.0.0.1:8765/mcp` with the bearer
  token. Good for remote/containerized agents.

---

## 1. Claude Code (CLI)

**HTTP transport (recommended):**

```bash
# PowerShell
$env:HIMURA_MCP_TOKEN = Get-Content "E:\Himura Pixel Tools\himura_data\mcp_token"
claude mcp add --transport http himura-pixel-tools http://127.0.0.1:8765/mcp `
  --header "Authorization: Bearer $env:HIMURA_MCP_TOKEN"
```

**stdio transport (no token needed):**

```bash
claude mcp add himura-pixel-tools -- "E:\Himura Pixel Tools\himura_pixel_tools\.venv\Scripts\himura-pixel-tools-mcp.exe" --transport stdio
```

Verify: `claude mcp list` → you should see `himura-pixel-tools` and its tools.

---

## 2. Claude Desktop

Edit `claude_desktop_config.json`
(Settings → Developer → Edit Config, or
`%APPDATA%\Claude\claude_desktop_config.json`) and add:

```json
{
  "mcpServers": {
    "himura-pixel-tools": {
      "command": "E:\\Himura Pixel Tools\\himura_pixel_tools\\.venv\\Scripts\\himura-pixel-tools-mcp.exe",
      "args": ["--transport", "stdio"]
    }
  }
}
```

Restart Claude Desktop. The Himura tools appear in the 🔌 tools menu.

---

## 3. OpenAI Codex (CLI)

Codex reads MCP servers from `~/.codex/config.toml`
(`C:\Users\<you>\.codex\config.toml`). Add:

```toml
[mcp_servers.himura-pixel-tools]
command = "E:\\Himura Pixel Tools\\himura_pixel_tools\\.venv\\Scripts\\himura-pixel-tools-mcp.exe"
args = ["--transport", "stdio"]
```

Then start Codex and ask it to list tools — `generate_asset`, `create_character`,
`generate_portrait`, etc. should be available.

---

## 4. Google Antigravity

Antigravity manages MCP servers from its **MCP settings panel** (Agent/Tools →
*Manage MCP servers* → *Add custom server* / *Edit mcp_config.json*). Paste this
server entry:

```json
{
  "mcpServers": {
    "himura-pixel-tools": {
      "command": "E:\\Himura Pixel Tools\\himura_pixel_tools\\.venv\\Scripts\\himura-pixel-tools-mcp.exe",
      "args": ["--transport", "stdio"]
    }
  }
}
```

Save and reload the server in the panel. If you prefer HTTP, add it as a
streamable‑HTTP server pointing at `http://127.0.0.1:8765/mcp` with header
`Authorization: Bearer <your token>`.

---

## 5. Any other MCP client (generic)

- **stdio:** run
  `"E:\Himura Pixel Tools\himura_pixel_tools\.venv\Scripts\himura-pixel-tools-mcp.exe" --transport stdio`
  (equivalently `python -m himura_pixel_tools.mcp.server --transport stdio` inside
  the venv).
- **HTTP:** `POST http://127.0.0.1:8765/mcp` with
  `Authorization: Bearer <token>` (Streamable‑HTTP MCP).
- **Standalone HTTP MCP** on its own port:
  `himura-pixel-tools-mcp --transport http --port 8766` →
  `http://127.0.0.1:8766/mcp`.

---

## Available tools

| Tool | What it does |
|---|---|
| `generate_asset` | Exact‑size sprite (item / prop / character / **building** / ui_element / background) |
| `create_character` | Persistent character + canonical reference (identity lock) |
| `generate_turnaround` | 4/8 directional sprites, locked to the character's identity |
| `generate_portrait` | Face/bust portrait matched to a character (dialogue/UI art) |
| `animate_character` | Aligned animation frames + sprite sheet + GIF/WebP preview |
| `inpaint_asset` | Edit a sprite while preserving style/identity |
| `create_tileset` | Seamless top‑down / sidescroller / isometric / Wang tiles |
| `snap_true_pixels` | Snap any image to a clean auto‑detected true‑pixel grid |
| `export_pack` | Bundle assets + engine metadata (Godot/Unity/Phaser/Aseprite) into a ZIP |
| `get_job_status` | Poll a long‑running generation job |
| `list_models` | List installed local models |
| `validate_asset` | Check exact‑size / palette / alpha constraints |

Only these high‑level tools are exposed — never shell access. All output paths
are sandboxed to the project's asset folders.

### Example agent prompts

- *"Create a character called 'Blue Knight' (small knight, blue cape, silver
  armor, top‑down RPG), then generate a 4‑direction turnaround and a smiling
  portrait."*
- *"Generate a 32×32 health‑potion item and a 64×64 stone tower building."*
- *"Make a seamless 16×16 grass tileset with 8 tiles, then export everything as a
  Godot pack."*

> **Tip:** most tools take an `output_folder`. Pass a subfolder name (e.g.
> `"blue_knight"`) — it is created under the sandboxed project assets root.
