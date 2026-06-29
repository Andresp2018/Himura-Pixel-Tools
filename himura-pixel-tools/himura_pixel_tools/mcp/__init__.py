"""Himura Pixel Tools MCP server.

Exposes the 10 safe asset-generation tools from spec.mcp_server_spec.tools:
  generate_asset, create_character, generate_turnaround, animate_character,
  inpaint_asset, create_tileset, export_pack, get_job_status, list_models,
  validate_asset.

The MCP bridge forwards every tool call to the local himura-pixel-tools-api over HTTP
on 127.0.0.1 — it never exposes shell tools and enforces output-folder
sandboxing (handled by the API). Two transports:
  - stdio            : ``himura-pixel-tools-mcp --transport stdio``
  - streamable_http  : served by the FastAPI app at /mcp

Security: only high-level Himura Pixel Tools tools are exposed; output paths are
sandboxed to project asset folders; HTTP transport requires a bearer token.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Optional

from .. import config


def _api_url() -> str:
    host = os.environ.get("HIMURA_HOST", config.DEFAULT_HOST)
    port = os.environ.get("HIMURA_PORT", str(config.DEFAULT_PORT))
    return f"http://{host}:{port}"


def _token() -> Optional[str]:
    return config.TOKEN_PATH.read_text(encoding="utf-8").strip() if config.TOKEN_PATH.exists() else None


async def _http(method: str, path: str, body: dict | None = None) -> dict:
    """Forward a call to the local API."""
    import httpx
    headers = {}
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        async with httpx.AsyncClient(base_url=_api_url(), timeout=600.0) as client:
            if method == "GET":
                r = await client.get(path, headers=headers)
            else:
                r = await client.request(method, path, json=body or {}, headers=headers)
    except httpx.ConnectError:
        return {"error": "Himura Pixel Tools is not running. Start it with START.BAT "
                         f"(the MCP tools talk to the local API at {_api_url()}), then retry."}
    except httpx.HTTPError as e:
        return {"error": f"could not reach the Himura API at {_api_url()}: {e}. "
                         "Make sure START.BAT is running."}
    if r.status_code == 401:
        return {"error": "the Himura API rejected the MCP token (HTTP 401). Restart the app "
                         "so the MCP bridge and the API share the same token file, then retry."}
    if r.status_code >= 400:
        return {"error": f"HTTP {r.status_code}", "detail": r.text}
    return r.json()


# ── tool implementations (async, return JSON) ─────────────────────────────────


async def tool_generate_asset(args: dict) -> dict:
    req = {
        "asset_type": args.get("asset_type", "item"),
        "prompt": args["prompt"],
        "target_size": {"width": args.get("width", 64), "height": args.get("height", 64)},
        "transparent": args.get("transparent", True),
        "style_profile_id": args.get("style_profile_id"),
        "character_profile_id": args.get("character_profile_id"),
        "palette_limit": args.get("palette_limit", 24),
        "output_root": args.get("output_folder"),
        "seed": args.get("seed"),
        "palette_preset": args.get("palette_preset"),
        "dither": args.get("dither", "none"),
        "protect_extremes": args.get("protect_extremes", False),
        "lora_id": args.get("lora_id"),
        "lora_weight": args.get("lora_weight"),
    }
    return await _http("POST", "/api/jobs/generate", req)


async def tool_create_character(args: dict) -> dict:
    req = {
        "name": args["name"], "description": args["description"],
        "width": args.get("width", 64), "height": args.get("height", 64),
        "directions": args.get("directions", 4),
        "style_profile_id": args.get("style_profile_id"),
        "output_folder": args.get("output_folder"),
    }
    return await _http("POST", "/api/jobs/create-character", req)


async def tool_generate_turnaround(args: dict) -> dict:
    req = {
        "character_profile_id": args["character_profile_id"],
        "directions": args.get("directions", 4),
        "width": args.get("width"), "height": args.get("height"),
        "output_folder": args.get("output_folder"),
    }
    return await _http("POST", "/api/jobs/turnaround", req)


async def tool_animate_character(args: dict) -> dict:
    req = {
        "character_profile_id": args["character_profile_id"],
        "animation": args.get("animation", "idle"),
        "directions": args.get("directions", 4),
        "frames_per_direction": args.get("frames_per_direction"),
        "output_folder": args.get("output_folder"),
    }
    return await _http("POST", "/api/jobs/animate-character", req)


async def tool_generate_portrait(args: dict) -> dict:
    req = {
        "character_profile_id": args["character_profile_id"],
        "width": args.get("width", 128), "height": args.get("height", 128),
        "palette_limit": args.get("palette_limit", 32),
        "expression": args.get("expression"),
        "transparent": args.get("transparent", True),
        "output_folder": args.get("output_folder"),
        "seed": args.get("seed"),
    }
    return await _http("POST", "/api/jobs/portrait", req)


async def tool_inpaint_asset(args: dict) -> dict:
    req = {
        "asset_path": args["asset_path"], "mask_path": args["mask_path"],
        "edit_prompt": args["edit_prompt"],
        "character_profile_id": args.get("character_profile_id"),
        "output_folder": args.get("output_folder"),
    }
    return await _http("POST", "/api/jobs/inpaint", req)


async def tool_create_tileset(args: dict) -> dict:
    req = {
        "description": args["description"],
        "tile_width": args.get("tile_width", 16),
        "tile_height": args.get("tile_height", 16),
        "tileset_type": args.get("tileset_type", "top_down"),
        "tile_count": args.get("tile_count", 8),
        "output_folder": args.get("output_folder"),
    }
    return await _http("POST", "/api/jobs/tileset", req)



async def tool_create_1_direction_object(args: dict) -> dict:
    req = {
        "description": args["description"],
        "size": args.get("size", 64),
        "view": args.get("view", "top-down"),
        "output_folder": args.get("output_folder"),
        "seed": args.get("seed"),
    }
    return await _http("POST", "/api/jobs/object", req)


async def tool_create_8_direction_object(args: dict) -> dict:
    req = {
        "description": args["description"],
        "size": args.get("size", 64),
        "view": args.get("view", "low top-down"),
        "output_folder": args.get("output_folder"),
        "seed": args.get("seed"),
    }
    return await _http("POST", "/api/jobs/object-8dir", req)


async def tool_create_object_state(args: dict) -> dict:
    req = {
        "object_id": args["object_id"],
        "edit_description": args["edit_description"],
        "output_folder": args.get("output_folder"),
        "seed": args.get("seed"),
    }
    return await _http("POST", "/api/jobs/object-state", req)


async def tool_get_object(args: dict) -> dict:
    return await _http("GET", f"/api/objects/{args['object_id']}")


async def tool_list_objects(args: dict) -> dict:
    return await _http("GET", "/api/objects")


async def tool_create_map_object(args: dict) -> dict:
    req = {
        "description": args["description"],
        "width": args.get("width", 64),
        "height": args.get("height", 64),
        "view": args.get("view", "low top-down"),
        "detail": args.get("detail"),
        "output_folder": args.get("output_folder"),
        "seed": args.get("seed"),
    }
    return await _http("POST", "/api/jobs/map-object", req)


async def tool_get_map_object(args: dict) -> dict:
    return await _http("GET", f"/api/map-objects/{args['object_id']}")


async def tool_create_ui_asset(args: dict) -> dict:
    req = {
        "description": args["description"],
        "name": args.get("name"),
        "width": args.get("width", 256),
        "height": args.get("height", 128),
        "color_palette": args.get("color_palette"),
        "elements": args.get("elements") or [],
        "no_background": args.get("no_background", True),
        "output_folder": args.get("output_folder"),
        "seed": args.get("seed"),
    }
    return await _http("POST", "/api/jobs/ui-asset", req)


async def tool_get_ui_asset(args: dict) -> dict:
    return await _http("GET", f"/api/ui-assets/{args['ui_asset_id']}")


async def tool_create_topdown_tileset(args: dict) -> dict:
    req = {
        "lower_description": args["lower_description"],
        "upper_description": args["upper_description"],
        "transition_description": args.get("transition_description"),
        "tile_size": args.get("tile_size") or {"width": 16, "height": 16},
        "transition_size": args.get("transition_size", 0.5),
        "view": args.get("view", "low top-down"),
        "mode": args.get("mode", "standard"),
        "output_folder": args.get("output_folder"),
        "seed": args.get("seed"),
    }
    return await _http("POST", "/api/jobs/topdown-tileset", req)


async def tool_create_sidescroller_tileset(args: dict) -> dict:
    req = {
        "lower_description": args["lower_description"],
        "transition_description": args["transition_description"],
        "tile_size": args.get("tile_size") or {"width": 16, "height": 16},
        "transition_size": args.get("transition_size", 0.25),
        "output_folder": args.get("output_folder"),
        "seed": args.get("seed"),
    }
    return await _http("POST", "/api/jobs/sidescroller-tileset", req)


async def tool_create_isometric_tile(args: dict) -> dict:
    req = {
        "description": args["description"],
        "size": args.get("size", 32),
        "tile_shape": args.get("tile_shape", "thick tile"),
        "output_folder": args.get("output_folder"),
        "seed": args.get("seed"),
    }
    return await _http("POST", "/api/jobs/isometric-tile", req)


async def tool_create_tiles_pro(args: dict) -> dict:
    req = {
        "description": args["description"],
        "tile_size": args.get("tile_size", 32),
        "tile_type": args.get("tile_type", "square_topdown"),
        "tile_view": args.get("tile_view", "top-down"),
        "output_folder": args.get("output_folder"),
        "seed": args.get("seed"),
    }
    return await _http("POST", "/api/jobs/tiles-pro", req)


async def tool_export_pack(args: dict) -> dict:
    req = {
        "asset_ids": args.get("asset_ids", []),
        "engine": args.get("engine", "generic"),
        "output_folder": args.get("output_folder"),
    }
    return await _http("POST", "/api/export/pack", req)


async def tool_get_job_status(args: dict) -> dict:
    return await _http("GET", f"/api/jobs/{args['job_id']}")


async def tool_list_models(args: dict) -> dict:
    return await _http("GET", "/api/models")


async def tool_snap_true_pixels(args: dict) -> dict:
    req = {
        "path": args.get("path"),
        "pixel_size": args.get("pixel_size"),
        "k_colors": args.get("k_colors", 0),
    }
    return await _http("POST", "/api/snap", req)


async def tool_validate_asset(args: dict) -> dict:
    req = {
        "asset_path": args["asset_path"],
        "expected_width": args.get("expected_width"),
        "expected_height": args.get("expected_height"),
        "palette_limit": args.get("palette_limit"),
    }
    return await _http("POST", "/api/validate/asset", req)


# Tool registry: name -> (description, input schema, handler)
TOOLS: dict[str, tuple[str, dict, Any]] = {
    "generate_asset": (
        "Generate an exact-size true-pixel asset from text and optional reference.",
        {
            "type": "object",
            "properties": {
                "asset_type": {"type": "string", "default": "item",
                               "enum": ["item", "prop", "character", "building",
                                        "ui_element", "background_scene"]},
                "prompt": {"type": "string"},
                "width": {"type": "integer", "default": 64},
                "height": {"type": "integer", "default": 64},
                "style_profile_id": {"type": "string"},
                "character_profile_id": {"type": "string"},
                "transparent": {"type": "boolean", "default": True},
                "palette_limit": {"type": "integer"},
                "output_folder": {"type": "string"},
                "seed": {"type": "integer"},
                "palette_preset": {"type": "string",
                                   "enum": ["gameboy", "gameboy_pocket", "pico8",
                                            "nes", "c64", "cga", "sweetie16"],
                                   "description": "lock output to a retro hardware palette"},
                "dither": {"type": "string",
                           "enum": ["none", "bayer2", "bayer4", "bayer8"],
                           "default": "none",
                           "description": "ordered (Bayer) dithering when mapping to the palette"},
                "protect_extremes": {"type": "boolean", "default": False,
                                     "description": "keep outline/highlight colours when reducing palette"},
                "lora_id": {"type": "string",
                            "description": "installed style LoRA model id to attach (base-compatible)"},
                "lora_weight": {"type": "number", "description": "LoRA strength (default 1.0)"},
            },
            "required": ["prompt", "output_folder"],
        },
        tool_generate_asset,
    ),
    "create_character": (
        "Create a persistent CharacterProfile with canonical reference sprite and metadata.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "width": {"type": "integer", "default": 64},
                "height": {"type": "integer", "default": 64},
                "directions": {"type": "integer", "enum": [4, 8], "default": 4},
                "style_profile_id": {"type": "string"},
                "output_folder": {"type": "string"},
            },
            "required": ["name", "description", "output_folder"],
        },
        tool_create_character,
    ),
    "generate_turnaround": (
        "Generate 4/8 directional sprites for an existing CharacterProfile.",
        {
            "type": "object",
            "properties": {
                "character_profile_id": {"type": "string"},
                "directions": {"type": "integer", "enum": [4, 8], "default": 4},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "output_folder": {"type": "string"},
            },
            "required": ["character_profile_id", "output_folder"],
        },
        tool_generate_turnaround,
    ),
    "animate_character": (
        "Generate aligned sprite animation frames and sprite sheet for a CharacterProfile.",
        {
            "type": "object",
            "properties": {
                "character_profile_id": {"type": "string"},
                "animation": {"type": "string",
                              "enum": ["idle", "walk", "run", "attack", "hurt", "death", "cast", "custom"],
                              "default": "idle"},
                "directions": {"type": "integer", "enum": [4, 8], "default": 4},
                "frames_per_direction": {"type": "integer"},
                "output_folder": {"type": "string"},
            },
            "required": ["character_profile_id", "animation", "output_folder"],
        },
        tool_animate_character,
    ),
    "generate_portrait": (
        "Generate a face/bust portrait of an existing character, matched to its "
        "sprite colors and design (dialogue/UI art).",
        {
            "type": "object",
            "properties": {
                "character_profile_id": {"type": "string"},
                "width": {"type": "integer", "default": 128},
                "height": {"type": "integer", "default": 128},
                "palette_limit": {"type": "integer", "default": 32},
                "expression": {"type": "string",
                               "description": "neutral, smiling, angry, sad, etc."},
                "transparent": {"type": "boolean", "default": True},
                "output_folder": {"type": "string"},
                "seed": {"type": "integer"},
            },
            "required": ["character_profile_id", "output_folder"],
        },
        tool_generate_portrait,
    ),
    "inpaint_asset": (
        "Edit an existing sprite while preserving style/identity.",
        {
            "type": "object",
            "properties": {
                "asset_path": {"type": "string"},
                "mask_path": {"type": "string"},
                "edit_prompt": {"type": "string"},
                "character_profile_id": {"type": "string"},
                "output_folder": {"type": "string"},
            },
            "required": ["asset_path", "mask_path", "edit_prompt", "output_folder"],
        },
        tool_inpaint_asset,
    ),
    "create_tileset": (
        "Create a seamless tileset or Wang tileset with exact tile size.",
        {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "tile_width": {"type": "integer", "default": 16},
                "tile_height": {"type": "integer", "default": 16},
                "tileset_type": {"type": "string",
                                 "enum": ["top_down", "sidescroller", "isometric", "wang"],
                                 "default": "top_down"},
                "tile_count": {"type": "integer"},
                "output_folder": {"type": "string"},
            },
            "required": ["description", "output_folder"],
        },
        tool_create_tileset,
    ),
    "create_1_direction_object": (
        "Queue a single-direction pixel art object with transparent background.",
        {"type": "object", "properties": {
            "description": {"type": "string"}, "size": {"type": "integer", "default": 64},
            "view": {"type": "string", "enum": ["top-down", "sidescroller", "low top-down", "high top-down", "side"]},
            "output_folder": {"type": "string"}, "seed": {"type": "integer"}},
         "required": ["description", "output_folder"]},
        tool_create_1_direction_object,
    ),
    "create_8_direction_object": (
        "Queue an eight-direction object rotation sheet.",
        {"type": "object", "properties": {
            "description": {"type": "string"}, "size": {"type": "integer", "default": 64},
            "view": {"type": "string", "enum": ["low top-down", "high top-down", "side"]},
            "output_folder": {"type": "string"}, "seed": {"type": "integer"}},
         "required": ["description", "output_folder"]},
        tool_create_8_direction_object,
    ),
    "create_object_state": (
        "Create an edited variant/state of a completed local object asset.",
        {"type": "object", "properties": {
            "object_id": {"type": "string"}, "edit_description": {"type": "string"},
            "output_folder": {"type": "string"}, "seed": {"type": "integer"}},
         "required": ["object_id", "edit_description", "output_folder"]},
        tool_create_object_state,
    ),
    "get_object": (
        "Get a local object asset by object_id.",
        {"type": "object", "properties": {"object_id": {"type": "string"}}, "required": ["object_id"]},
        tool_get_object,
    ),
    "list_objects": (
        "List local object assets.",
        {"type": "object", "properties": {}},
        tool_list_objects,
    ),
    "create_map_object": (
        "Create a transparent map object sprite.",
        {"type": "object", "properties": {
            "description": {"type": "string"}, "width": {"type": "integer", "default": 64},
            "height": {"type": "integer", "default": 64}, "view": {"type": "string"},
            "detail": {"type": "string"}, "output_folder": {"type": "string"}, "seed": {"type": "integer"}},
         "required": ["description", "output_folder"]},
        tool_create_map_object,
    ),
    "get_map_object": (
        "Get a local map object asset by object_id.",
        {"type": "object", "properties": {"object_id": {"type": "string"}}, "required": ["object_id"]},
        tool_get_map_object,
    ),
    "create_ui_asset": (
        "Create a pixel art UI panel or interface element.",
        {"type": "object", "properties": {
            "description": {"type": "string"}, "name": {"type": "string"},
            "width": {"type": "integer", "default": 256}, "height": {"type": "integer", "default": 128},
            "color_palette": {"type": "string"}, "elements": {"type": "array", "items": {"type": "string"}},
            "no_background": {"type": "boolean", "default": True}, "output_folder": {"type": "string"}, "seed": {"type": "integer"}},
         "required": ["description", "output_folder"]},
        tool_create_ui_asset,
    ),
    "get_ui_asset": (
        "Get a local UI asset by ui_asset_id.",
        {"type": "object", "properties": {"ui_asset_id": {"type": "string"}}, "required": ["ui_asset_id"]},
        tool_get_ui_asset,
    ),
    "create_topdown_tileset": (
        "Create a top-down Wang/autotile terrain tileset.",
        {"type": "object", "properties": {
            "lower_description": {"type": "string"}, "upper_description": {"type": "string"},
            "transition_description": {"type": "string"}, "tile_size": {"type": "object"},
            "transition_size": {"type": "number"}, "view": {"type": "string"}, "mode": {"type": "string"},
            "output_folder": {"type": "string"}, "seed": {"type": "integer"}},
         "required": ["lower_description", "upper_description", "output_folder"]},
        tool_create_topdown_tileset,
    ),
    "create_sidescroller_tileset": (
        "Create a side-view platformer terrain tileset.",
        {"type": "object", "properties": {
            "lower_description": {"type": "string"}, "transition_description": {"type": "string"},
            "tile_size": {"type": "object"}, "transition_size": {"type": "number"},
            "output_folder": {"type": "string"}, "seed": {"type": "integer"}},
         "required": ["lower_description", "transition_description", "output_folder"]},
        tool_create_sidescroller_tileset,
    ),
    "create_isometric_tile": (
        "Create a single isometric tile.",
        {"type": "object", "properties": {
            "description": {"type": "string"}, "size": {"type": "integer"},
            "tile_shape": {"type": "string"}, "output_folder": {"type": "string"}, "seed": {"type": "integer"}},
         "required": ["description", "output_folder"]},
        tool_create_isometric_tile,
    ),
    "create_tiles_pro": (
        "Create multiple shaped tile variations.",
        {"type": "object", "properties": {
            "description": {"type": "string"}, "tile_size": {"type": "integer"},
            "tile_type": {"type": "string"}, "tile_view": {"type": "string"},
            "output_folder": {"type": "string"}, "seed": {"type": "integer"}},
         "required": ["description", "output_folder"]},
        tool_create_tiles_pro,
    ),
    "export_pack": (
        "Package generated assets and metadata for a game engine.",
        {
            "type": "object",
            "properties": {
                "asset_ids": {"type": "array", "items": {"type": "string"}},
                "engine": {"type": "string",
                           "enum": ["godot", "unity", "phaser", "generic"], "default": "generic"},
                "output_folder": {"type": "string"},
            },
            "required": ["output_folder"],
        },
        tool_export_pack,
    ),
    "get_job_status": (
        "Check job status and retrieve output paths.",
        {"type": "object", "properties": {"job_id": {"type": "string"}},
         "required": ["job_id"]},
        tool_get_job_status,
    ),
    "list_models": (
        "List installed local models and compatibility profiles.",
        {"type": "object", "properties": {}},
        tool_list_models,
    ),
    "snap_true_pixels": (
        "Snap an image to a clean true-pixel grid (auto-detects native pixel size). "
        "Takes a sandboxed file path under the projects root.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "image path under the projects root"},
                "pixel_size": {"type": "number", "description": "override; omit to auto-detect"},
                "k_colors": {"type": "integer", "default": 0,
                             "description": "0 keeps colors; >1 also k-means quantizes"},
            },
            "required": ["path"],
        },
        tool_snap_true_pixels,
    ),
    "validate_asset": (
        "Validate exact-size true-pixel output constraints.",
        {
            "type": "object",
            "properties": {
                "asset_path": {"type": "string"},
                "expected_width": {"type": "integer"},
                "expected_height": {"type": "integer"},
                "palette_limit": {"type": "integer"},
            },
            "required": ["asset_path"],
        },
        tool_validate_asset,
    ),
}
