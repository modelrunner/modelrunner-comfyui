"""Schema -> ComfyUI widget mapping. Pure: imports nothing from ComfyUI.

Every typed node is built from one catalog entry's Input schema. `plan_fields`
turns each property into a FieldPlan (which ComfyUI input to show, and how to
turn the widget value back into API input). Anything the mapper cannot show as
a widget lands in the node's `extra_json` input; `fallback_report` counts those
so a catalog refresh can be checked before it ships.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

UNSET = "(default)"  # COMBO choice meaning "leave this field out"
INT_MAX = 2**53 - 1
SEED_MAX = 2**31 - 1

IMAGE_HINT = re.compile(r"image|img|mask|face|photo|frame|picture|portrait|sketch|style_ref", re.I)
AUDIO_HINT = re.compile(r"audio|voice|speech|music|song|sound", re.I)
VIDEO_HINT = re.compile(r"video|clip|footage|motion", re.I)
MULTILINE_HINT = re.compile(r"prompt|text|lyrics|script|instruction|description|negative", re.I)


@dataclass
class FieldPlan:
    name: str
    kind: str  # enum | bool | int | float | string | image | images | audio | video | url | urls | object
    required: bool
    comfy_type: Any  # "INT", "IMAGE", ... or a list of COMBO choices
    options: dict = field(default_factory=dict)
    enum_values: list = field(default_factory=list)  # original typed enum values
    children: list = field(default_factory=list)  # flattened object properties
    advanced: bool = False


def normalize(schema: dict) -> dict:
    """Collapse allOf, unwrap nullable anyOf/oneOf, prefer a preset enum over a
    free-form object. Returns a plain schema, or {"x-json": True} when the
    union has no single widget form."""
    s = dict(schema or {})
    if "allOf" in s:
        branches = s.pop("allOf")
        merged: dict = {}
        for b in branches:
            merged.update(normalize(b))
        merged.update(s)
        s = merged
    for key in ("anyOf", "oneOf"):
        if key not in s:
            continue
        branches = s.pop(key)
        if s.get("type") in ("integer", "number", "string", "boolean"):
            continue  # constraint-only union on a typed field: keep the type
        live = [normalize(b) for b in branches if b.get("type") != "null"]
        nullable = len(live) < len(branches)
        enums = [b for b in live if "enum" in b]
        if len(live) == 1:
            pick = live[0]
        elif enums and all(b.get("type") == "object" or "enum" in b for b in live):
            pick = enums[0]  # e.g. a size preset OR a custom {width,height}: show the preset
        else:
            return {**s, "x-json": True}
        out = {**pick, **{k: v for k, v in s.items() if k not in pick or k in ("default", "description")}}
        if nullable:
            out["x-nullable"] = True
        s = out
    return s


def _is_uri(s: dict) -> bool:
    return s.get("type") == "string" and s.get("format") in ("uri", "url", "uri-reference")


def _media_kind(name: str, input_modalities: list[str]) -> str:
    if AUDIO_HINT.search(name):
        return "audio"
    if VIDEO_HINT.search(name) and not IMAGE_HINT.search(name):
        return "video"
    if IMAGE_HINT.search(name):
        return "image"
    media = [m for m in input_modalities if m in ("image", "audio", "video")]
    if media == ["image"]:
        return "image"
    if media == ["audio"]:
        return "audio"
    return "url"


def _number_opts(s: dict, is_int: bool) -> dict:
    opts: dict = {}
    lo = s.get("minimum", s.get("exclusiveMinimum"))
    hi = s.get("maximum", s.get("exclusiveMaximum"))
    if lo is not None:
        opts["min"] = int(lo) if is_int else float(lo)
    if hi is not None:
        opts["max"] = int(hi) if is_int else float(hi)
    if not is_int:
        opts["step"] = float(s.get("multipleOf", 0.01))
    if "default" in s and isinstance(s["default"], (int, float)) and not isinstance(s["default"], bool):
        opts["default"] = int(s["default"]) if is_int else float(s["default"])
    else:
        opts["default"] = opts.get("min", 0) if is_int else float(opts.get("min", 0.0))
    if is_int:
        opts.setdefault("min", -INT_MAX)
        opts.setdefault("max", INT_MAX)
    return opts


def plan_field(name: str, raw: dict, required: bool, input_modalities: list[str], depth: int = 0) -> FieldPlan:
    s = normalize(raw)
    has_default = "default" in s and s["default"] is not None
    optional_unset = not required and not has_default
    tooltip = (s.get("description") or "")[:300]
    base = {"tooltip": tooltip} if tooltip else {}

    if s.get("x-json"):
        return FieldPlan(name, "object", required, "JSON")

    if "enum" in s:
        values = list(s["enum"])
        choices = [str(v) for v in values]
        if optional_unset:
            choices = [UNSET] + choices
        opts = dict(base)
        if has_default and str(s["default"]) in choices:
            opts["default"] = str(s["default"])
        return FieldPlan(name, "enum", required, choices, opts, enum_values=values)

    t = s.get("type")
    if t == "boolean":
        if optional_unset:
            return FieldPlan(name, "enum", required, [UNSET, "true", "false"], dict(base), enum_values=[True, False])
        return FieldPlan(name, "bool", required, "BOOLEAN", {**base, "default": bool(s.get("default", False))})

    if t in ("integer", "number"):
        is_int = t == "integer"
        opts = {**base, **_number_opts(s, is_int)}
        if name == "seed" and is_int:
            # Randomize picks anywhere in [min, max]; most models reject seeds past int32.
            hi = s.get("maximum", SEED_MAX)
            opts.update(min=max(opts.get("min", 0), 0), max=min(int(hi), INT_MAX), control_after_generate="fixed")  # randomize would make every queue a paid run
            opts["default"] = min(max(opts["default"], opts["min"]), opts["max"])
            return FieldPlan(name, "int", required, "INT", opts)
        if optional_unset:
            opts["forceInput"] = True  # a socket: left unconnected, the field is omitted
        return FieldPlan(name, "int" if is_int else "float", required, "INT" if is_int else "FLOAT", opts)

    if _is_uri(s):
        kind = _media_kind(name, input_modalities)
        if kind == "image":
            return FieldPlan(name, "image", required, "IMAGE", base)
        if kind == "audio":
            return FieldPlan(name, "audio", required, "AUDIO", base)
        if kind == "video":
            tip = f"{tooltip} (or put a video URL under this name in extra_json)".strip()
            return FieldPlan(name, "video", required, "VIDEO", {"tooltip": tip})
        return FieldPlan(name, "url", required, "STRING", {**base, "default": ""})

    if t == "array" and _is_uri(normalize(s.get("items", {}))):
        kind = _media_kind(name, input_modalities)
        if kind == "image":
            return FieldPlan(name, "images", required, "IMAGE", base)
        return FieldPlan(name, "urls", required, "STRING", {**base, "default": "", "multiline": True})

    if t == "array" and normalize(s.get("items", {})).get("type") in ("integer", "number", "string"):
        item_type = normalize(s["items"])["type"]
        default = s.get("default") if isinstance(s.get("default"), list) else []
        sep = "\n" if item_type == "string" else ", "
        hint = "one per line" if item_type == "string" else "comma-separated"
        tip = f"{tooltip} ({hint})".strip()
        return FieldPlan(name, f"list:{item_type}", required, "STRING",
                         {"tooltip": tip, "default": sep.join(str(v) for v in default), "multiline": item_type == "string"})

    if t == "string":
        multiline = bool(MULTILINE_HINT.search(name)) or (s.get("maxLength") or 0) > 200
        default = s.get("default") if isinstance(s.get("default"), str) else ""
        return FieldPlan(name, "string", required, "STRING", {**base, "default": default, "multiline": multiline})

    if t == "object" and depth == 0 and s.get("properties"):
        req = set(s.get("required", []))
        kids = [plan_field(k, v, k in req, input_modalities, depth + 1) for k, v in s["properties"].items()]
        if all(k.kind in ("enum", "bool", "int", "float", "string") for k in kids):
            return FieldPlan(name, "object", required, None, children=kids)

    return FieldPlan(name, "object", required, "JSON")


def plan_fields(entry: dict) -> list[FieldPlan]:
    props = entry["input"]["properties"]
    required = set(entry["input"].get("required", []))
    order = sorted(props, key=lambda k: (props[k].get("x-order", 10_000), list(props).index(k)))
    offset = entry.get("advancedInputsOffset")
    plans = []
    for i, name in enumerate(order):
        p = plan_field(name, props[name], name in required, entry.get("inputModalities", []))
        p.advanced = offset is not None and i >= offset and not p.required
        plans.append(p)
    return plans


def node_id(endpoint: str) -> str:
    # LiteGraph reads "/" in a type name as a category path, so it is replaced.
    return "ModelRunner_" + endpoint.replace("/", "__").replace(".", "_")


def output_kind(entry: dict) -> str | None:
    """Which typed node a catalog row gets, or None to leave it to the generic node."""
    if (entry.get("category") or "").endswith("-to-3d"):
        return "3d"
    for kind in ("video", "image", "audio"):
        if kind in (entry.get("outputModalities") or []):
            return kind
    return None


def retired_message(entry: dict, now: datetime | None = None) -> str:
    """Error text for a retired row, or "" while the model is live."""
    retires = entry.get("retiresAt")
    now = now or datetime.now(timezone.utc)
    gone = bool(entry.get("retired"))
    if retires and not gone:
        try:
            gone = datetime.fromisoformat(retires.replace("Z", "+00:00")) <= now
        except ValueError:
            gone = False
    if not gone:
        return ""
    successor = entry.get("successorModelEndpoint")
    hint = f" Replace this node with the {successor} node." if successor else ""
    return f"{entry['endpoint']} has been retired.{hint}"


def widget_name(parent: str, child: str) -> str:
    return f"{parent}__{child}"


def input_types(plans: list[FieldPlan]) -> dict:
    """ComfyUI INPUT_TYPES for a planned node (extra_json/force_rerun added by the node)."""
    req: dict = {}
    opt: dict = {}

    def put(p: FieldPlan, wname: str):
        # Sockets that may stay unconnected, and fields the catalog marks advanced,
        # go in "optional"; ComfyUI then passes nothing for an unconnected socket.
        # A VIDEO socket is always optional so a required video URL can come from
        # extra_json instead; the node checks one of the two is set before it runs.
        socket = p.comfy_type in ("IMAGE", "AUDIO", "VIDEO") or p.options.get("forceInput")
        target = req if (p.required and p.kind != "video") or not (socket or p.advanced) else opt
        target[wname] = (p.comfy_type, p.options) if p.options else (p.comfy_type,)

    for p in plans:
        if p.kind == "object" and p.children:
            for c in p.children:
                put(c, widget_name(p.name, c.name))
        elif p.comfy_type != "JSON":
            put(p, p.name)
    return {"required": req, "optional": opt}


def to_api_value(p: FieldPlan, value: Any) -> tuple[bool, Any]:
    """(include?, value) for one plain widget value."""
    if value is None:
        return False, None
    if p.kind == "enum":
        if value == UNSET:
            return False, None
        for v in p.enum_values:
            if str(v) == str(value) or (isinstance(v, bool) and str(v).lower() == str(value)):
                return True, v
        return True, value
    if p.kind == "string":
        return (value != "" or p.required), value
    if p.kind == "url":
        return value.strip() != "", value.strip()
    if p.kind == "urls":
        urls = [u for u in re.split(r"[\s,]+", value or "") if u]
        return bool(urls), urls
    if p.kind.startswith("list:"):
        item_type = p.kind.split(":", 1)[1]
        if item_type == "string":
            items = [v.strip() for v in (value or "").splitlines() if v.strip()]
        else:
            cast = int if item_type == "integer" else float
            items = [cast(v) for v in re.split(r"[\s,]+", value or "") if v]
        return bool(items), items
    return True, value


def fallback_report(catalog: list[dict]) -> dict:
    """Fields that only `extra_json` can set, across the given entries."""
    fields, models = [], set()
    for e in catalog:
        for p in plan_fields(e):
            if p.comfy_type == "JSON":
                fields.append((e["endpoint"], p.name, "required" if p.required else ("advanced" if p.advanced else "basic")))
                models.add(e["endpoint"])
    return {"fields": fields, "models": sorted(models)}


def parse_extra_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("extra_json must be a JSON object of input fields")
    return data
