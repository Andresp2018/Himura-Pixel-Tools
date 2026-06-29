# Himura Pixel Tools — MCP client config examples

Himura Pixel Tools exposes an MCP server (`himura-pixel-tools`) with 10 safe tools.
The API + MCP HTTP endpoint run on **127.0.0.1** only.

## Get your bearer token

Start Himura Pixel Tools, then:

```bash
curl http://127.0.0.1:8765/api/mcp-token -H "Authorization: Bearer $HIMURA_MCP_TOKEN"
# or read the token file directly:
#   Windows:  <project>\himura_data\mcp_token   (e.g. E:\Himura Pixel Tools\himura_data\mcp_token)
#   Linux:    <project>/himura_data/mcp_token
# The Settings tab shows the token + a ready copy-paste config for each AI CLI.
```

Set it in your shell:

```bash
export HIMURA_MCP_TOKEN="paste-token-here"
```

---

## Claude Code

### Streamable HTTP (recommended — shared local server)

```bash
claude mcp add --transport http himura-pixel-tools http://127.0.0.1:8765/mcp \
  --header "Authorization: Bearer $HIMURA_MCP_TOKEN"
```

### stdio (per-project subprocess)

```bash
claude mcp add --transport stdio himura-pixel-tools -- himura-pixel-tools-mcp --transport stdio
```

---

## Google Antigravity

`.antigravity/mcp.json` (or the project MCP config):

```json
{
  "mcpServers": {
    "himura-pixel-tools": {
      "httpUrl": "http://127.0.0.1:8765/mcp",
      "headers": {
        "Authorization": "Bearer ${HIMURA_MCP_TOKEN}"
      },
      "timeout": 600000
    }
  }
}
```

---

## OpenAI Codex

`~/.codex/config.toml` — Streamable HTTP:

```toml
[mcp_servers.himura_pixel_tools]
url = "http://127.0.0.1:8765/mcp"
bearer_token_env_var = "HIMURA_MCP_TOKEN"
startup_timeout_sec = 20
tool_timeout_sec = 600
```

…or stdio:

```toml
[mcp_servers.himura_pixel_tools]
command = "himura-pixel-tools-mcp"
args = ["--transport", "stdio"]
startup_timeout_sec = 20
tool_timeout_sec = 600
```

---

## Available tools

`generate_asset`, `create_character`, `generate_turnaround`,
`animate_character`, `inpaint_asset`, `create_tileset`, `export_pack`,
`get_job_status`, `list_models`, `validate_asset`.

Only these high-level tools are exposed — never shell or filesystem tools.
All output paths are sandboxed under the project asset folder.
