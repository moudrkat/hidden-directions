"""Thin brainscope HTTP client — the GPU work happens on the server."""
import json
import os
import urllib.request

BASE = os.environ.get("BRAINSCOPE_BASE", "http://localhost:8010")


def forced_diff(messages, spec, *, kl=False, heads_layer=None,
                attribute_layer=None, max_tokens=48, timeout=3600):
    """Teacher-forced clean-vs-steered diff. spec is a hotwire steering dict."""
    body = {"messages": messages, "steering": spec, "forced": True,
            "max_tokens": max_tokens, "kl": bool(kl)}
    if heads_layer is not None:
        body["attribute_heads_layer"] = heads_layer
    if attribute_layer is not None:
        body["attribute_layer"] = attribute_layer
    req = urllib.request.Request(BASE + "/replay", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def unembed(direction_id, layer, top=60, timeout=120):
    with urllib.request.urlopen(
            f"{BASE}/directions/{direction_id}/unembed?layer={layer}&top={top}",
            timeout=timeout) as r:
        return json.loads(r.read())


def chat(messages, spec=None, *, tools=None, tool_choice=None, max_tokens=300,
         timeout=600):
    """Real generation via /v1/chat/completions, with optional per-request
    steering + tools + forced tool_choice. Returns the raw OpenAI-shaped
    response (message may carry content and/or tool_calls)."""
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if spec is not None:
        body["steering"] = spec
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
