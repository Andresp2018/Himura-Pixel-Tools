"""MCP server entry point with stdio + Streamable HTTP transports.

CLI:
    himura-pixel-tools-mcp --transport stdio
    himura-pixel-tools-mcp --transport http --port 8766

Uses the official ``mcp`` Python SDK when available. Falls back to a minimal
JSON-RPC-over-stdio implementation if the SDK isn't installed, so the bridge
still works for quick local testing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from typing import Any

from .. import config
from . import TOOLS


# â”€â”€ dispatch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def dispatch_tool(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool: {name}"}
    _desc, _schema, handler = TOOLS[name]
    try:
        result = await handler(args or {})
        return result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


# â”€â”€ stdio transport (minimal JSON-RPC, SDK-free fallback) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def _stdio_loop() -> int:
    """Minimal MCP-over-stdio: respond to tools/list and tools/call.

    Uses a background thread for stdin reads so it works on Windows, where
    asyncio pipe readers are not consistently available for console handles.
    """
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            break
        try:
            msg = json.loads(line)
        except Exception:
            continue
        resp = await _handle_jsonrpc(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


async def _handle_jsonrpc(msg) -> dict | list | None:
    if isinstance(msg, list):
        replies = []
        for item in msg:
            resp = await _handle_jsonrpc(item)
            if resp is not None:
                replies.append(resp)
        return replies
    if not isinstance(msg, dict):
        return {"jsonrpc": "2.0", "id": None,
                "error": {"code": -32600, "message": "invalid JSON-RPC message"}}

    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params", {}) or {}

    if method == "initialize":
        protocol = params.get("protocolVersion") or "2024-11-05"
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": protocol,
            "serverInfo": {"name": "himura-pixel-tools", "version": config.APP_VERSION},
            "capabilities": {"tools": {"listChanged": False}, "resources": {}, "prompts": {}},
            "instructions": "Use tools/list and tools/call to generate and inspect local Himura Pixel Tools assets.",
        }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": [
            {"name": n, "description": d, "inputSchema": s}
            for n, (d, s, _h) in TOOLS.items()
        ]}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        result = await dispatch_tool(name, args)
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}],
            "isError": bool(isinstance(result, dict) and result.get("error")),
        }}
    if method in {"resources/list", "prompts/list"}:
        key = "resources" if method == "resources/list" else "prompts"
        return {"jsonrpc": "2.0", "id": msg_id, "result": {key: []}}
    if method == "notifications/initialized":
        return None
    if msg_id is not None:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"method not found: {method}"}}
    return None


# Streamable HTTP transport â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def _http_server(port: int) -> int:
    """A tiny Starlette/ASGI app exposing the MCP HTTP transport at /mcp.

    If the ``mcp`` SDK is installed we use its StreamableHTTPServerTransport;
    otherwise we expose the same JSON-RPC over a plain POST endpoint.
    """
    try:
        from mcp.server import Server
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        import mcp.types as types
    except Exception:
        return await _http_server_minimal(port)

    server = Server("himura-pixel-tools")

    @server.list_tools()
    async def _list() -> list[types.Tool]:
        return [
            types.Tool(name=n, description=d, inputSchema=s)
            for n, (d, s, _h) in TOOLS.items()
        ]

    @server.call_tool()
    async def _call(name: str, args: dict) -> list[types.TextContent]:
        result = await dispatch_tool(name, args)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    manager = StreamableHTTPSessionManager(app=server)

    from starlette.applications import Starlette
    from starlette.routing import Mount
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    import contextlib

    async def _asgi(scope, receive, send):
        if scope["type"] == "lifespan":
            async with manager.run():
                await _noop_lifespan(scope, receive, send)
            return
        # /mcp endpoint
        if scope.get("path") == "/mcp":
            await manager.handle_request(scope, receive, send)
            return
        # health
        await JSONResponse({"status": "ok", "server": "himura-pixel-tools"})(scope, receive, send)

    async def _noop_lifespan(scope, receive, send):
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup.complete" or msg["type"] == "lifespan.shutdown.complete":
                break

    config_uvicorn = __import__("uvicorn").Config(_asgi, host="127.0.0.1", port=port, log_level="info")
    server_inst = __import__("uvicorn").Server(config_uvicorn)
    await server_inst.serve()
    return 0


async def _http_server_minimal(port: int) -> int:
    """Fallback: plain JSON-RPC over POST at /mcp when the MCP SDK is absent."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    async def _mcp(request: Request):
        if request.method == "GET":
            return JSONResponse({"status": "ok", "server": "himura-pixel-tools", "tools": len(TOOLS)})
        try:
            msg = await request.json()
        except Exception:
            return JSONResponse({"jsonrpc": "2.0", "id": None,
                                 "error": {"code": -32700, "message": "invalid JSON"}}, status_code=400)
        resp = await _handle_jsonrpc(msg)
        return JSONResponse(resp or {})

    async def _health(_):
        return JSONResponse({"status": "ok", "server": "himura-pixel-tools (minimal)"})

    app = Starlette(routes=[Route("/mcp", _mcp, methods=["POST", "GET"]),
                            Route("/mcp/", _mcp, methods=["POST", "GET"]),
                            Route("/", _health)])
    config_ = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config_)
    await server.serve()
    return 0


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Himura Pixel Tools MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)

    config.ensure_dirs()
    try:
        if args.transport == "stdio":
            return asyncio.run(_stdio_loop())
        # Prefer the SDK-based Streamable HTTP transport so strict MCP HTTP
        # clients (e.g. ``claude mcp add --transport http``) work. _http_server()
        # falls back to the minimal JSON-RPC server when the mcp SDK is absent.
        return asyncio.run(_http_server(args.port))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())



