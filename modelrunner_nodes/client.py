"""ModelRunner REST client, standard library only.

One place builds the API headers (key, app tag, User-Agent). The presigned
upload PUT deliberately does not use them: it carries only the headers the
upload URL was signed for.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

VERSION = "0.2.1"
APP_TAG = "modelrunner-comfyui"
# Fixed on purpose: the API key goes only to these hosts, so no setting may redirect it.
API_BASE = "https://modelrunner.run"
QUEUE_BASE = "https://queue.modelrunner.run"
KEYED_HOSTS = {"modelrunner.run", "queue.modelrunner.run"}
KEY_FILE = Path(__file__).resolve().parent.parent / "modelrunner_api_key.txt"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
UPLOAD_CACHE_TTL = 3600  # seconds; uploads expire under the account's retention setting


class ModelRunnerError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("MODELRUNNER_API_KEY", "").strip()
    if not key and KEY_FILE.is_file():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise ModelRunnerError(
            "No ModelRunner API key. Set the MODELRUNNER_API_KEY environment variable before starting "
            f"ComfyUI, or put the key in {KEY_FILE.name} inside the pack folder. Create a key at "
            "https://modelrunner.ai/settings/api-keys"
        )
    return key


def _headers(json_body: bool = True) -> dict:
    h = {
        "Authorization": f"Key {api_key()}",
        "x-modelrunner-app": APP_TAG,
        "User-Agent": f"{APP_TAG}/{VERSION}",
        "Accept": "application/json",
    }
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _is_keyed_host(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme == "https" and parts.hostname in KEYED_HOSTS


class _KeyStaysOnApiHosts(urllib.request.HTTPRedirectHandler):
    """urllib copies every header, the key included, onto a redirect. Drop the key
    when a redirect leaves the API hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not _is_keyed_host(newurl):
            new.remove_header("Authorization")
        return new


_opener = urllib.request.build_opener(_KeyStaysOnApiHosts)


def _request(method: str, url: str, body: Any = None, headers: Optional[dict] = None, timeout: int = 60) -> Any:
    if headers is None and not _is_keyed_host(url):
        # Status and result URLs come from the API response; never follow one elsewhere with the key.
        raise ModelRunnerError(f"Refusing to send the API key to {url}: not a ModelRunner API host.")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers if headers is not None else _headers())
    try:
        with _opener.open(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        try:
            parsed = json.loads(detail)
            err = parsed.get("error", parsed)
            detail = err.get("message", detail) if isinstance(err, dict) else str(err)
        except ValueError:
            pass
        raise ModelRunnerError(f"{method} {url} -> HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise ModelRunnerError(f"{method} {url} failed: {e.reason}") from None
    return json.loads(raw) if raw else None


def submit(endpoint: str, payload: dict) -> dict:
    """Queue a run. Returns {request_id, status_url, response_url, ...}."""
    return _request("POST", f"{QUEUE_BASE}/{endpoint.strip('/')}", payload)


def run(
    endpoint: str,
    payload: dict,
    check_interrupt: Callable[[], None] = lambda: None,
    on_progress: Callable[[str, float], None] = lambda status, elapsed: None,
    timeout: int = 1800,
) -> dict:
    """Submit, poll until terminal, return the result payload.

    `check_interrupt` raises to abandon the wait. That stops polling only: the
    queued job keeps running and is billed if it completes.
    """
    queued = submit(endpoint, payload)
    request_id = queued["request_id"]
    status_url, response_url = queued["status_url"], queued["response_url"]
    logging.info(f"[ModelRunner] {endpoint}: request {request_id} queued")
    start, delay, status = time.monotonic(), 1.0, "IN_QUEUE"
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise ModelRunnerError(f"Request {request_id} still {status} after {timeout}s; it may still finish.")
        # Sleep in short slices so an interrupt is noticed within a quarter second.
        until = time.monotonic() + delay
        while time.monotonic() < until:
            check_interrupt()
            time.sleep(0.25)
        status = (_request("GET", status_url) or {}).get("status", status)
        on_progress(status, elapsed)
        if status in TERMINAL:
            break
        delay = min(delay * 1.5, 5.0)

    result = _request("GET", response_url) or {}
    if status != "COMPLETED":
        raise ModelRunnerError(f"Request {request_id} {status}: {result.get('error') or 'no error message'}")
    # Just after COMPLETED the output can still be an inline value that is being
    # moved to a hosted URL; re-read briefly until a URL appears.
    for _ in range(5):
        if output_urls(result.get("output")):
            break
        check_interrupt()
        time.sleep(2)
        result = _request("GET", response_url) or result
    result.setdefault("request_id", request_id)
    return result


_chat_cache: dict[str, bool] = {}


def is_chat(endpoint: str) -> bool:
    """Whether a model is served by the chat-completions route instead of the
    queue (language models are). Read from the public model row; a failed read
    means False, and the queue's own error then names the right route."""
    if endpoint not in _chat_cache:
        try:
            row = _request("GET", f"{API_BASE}/models/{endpoint}",
                           headers={"User-Agent": f"{APP_TAG}/{VERSION}", "Accept": "application/json"}, timeout=15)
            _chat_cache[endpoint] = ((row or {}).get("providerRequestConfig") or {}).get("apiType") == "chat_completions"
        except ModelRunnerError:
            _chat_cache[endpoint] = False
    return _chat_cache[endpoint]


def chat(endpoint: str, body: dict, timeout: int = 600) -> dict:
    """One synchronous chat completion. `body` is an OpenAI-style request
    ({"messages": [...], ...}); returns {request_id, output: {content, ...}}."""
    completion = _request("POST", f"{QUEUE_BASE}/{endpoint.strip('/')}/chat/completions",
                          {**body, "stream": False}, timeout=timeout) or {}
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = {"content": message.get("content"), "finish_reason": choice.get("finish_reason"),
              "usage": completion.get("usage")}
    if message.get("tool_calls"):
        output["tool_calls"] = message["tool_calls"]
    return {"request_id": (completion.get("id") or "").removeprefix("chatcmpl-"), "status": "COMPLETED", "output": output}


def output_urls(output: Any) -> list[str]:
    """Every http(s) URL in an output value, in document order."""
    found: list[str] = []

    def walk(v: Any):
        if isinstance(v, str):
            if v.startswith(("http://", "https://")):
                found.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(output)
    return found


_upload_cache: dict[str, tuple[float, str]] = {}


def upload_bytes(data: bytes, file_name: str, content_type: str) -> str:
    """Upload bytes and return the hosted file URL. Identical bytes within the
    cache window reuse the earlier URL instead of uploading again."""
    digest = hashlib.sha256(data).hexdigest() + content_type
    hit = _upload_cache.get(digest)
    if hit and time.time() - hit[0] < UPLOAD_CACHE_TTL:
        return hit[1]
    init = _request("POST", f"{API_BASE}/storage/upload/initiate", {"content_type": content_type, "file_name": file_name})
    put_headers = {"Content-Type": content_type}
    if content_type == "image/svg+xml":
        put_headers["Content-Disposition"] = "attachment"
    req = urllib.request.Request(init["upload_url"], data=data, method="PUT", headers=put_headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        raise ModelRunnerError(f"Upload of {file_name} rejected: HTTP {e.code} {e.read()[:200]!r}") from None
    _upload_cache[digest] = (time.time(), init["file_url"])
    return init["file_url"]


def download(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": f"{APP_TAG}/{VERSION}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


_estimate_cache: dict[str, Optional[str]] = {}


def estimate(endpoint: str) -> Optional[str]:
    """Average cost of one run in USD as a string, or None when unknown.
    Public endpoint, no key needed; failures are never fatal."""
    if endpoint in _estimate_cache:
        return _estimate_cache[endpoint]
    value = None
    try:
        body = {"items": [{"endpoint": endpoint, "callQuantity": 1}]}
        res = _request("POST", f"{API_BASE}/models/pricing/estimate", body,
                       headers={"Content-Type": "application/json", "User-Agent": f"{APP_TAG}/{VERSION}"}, timeout=10)
        if res and res.get("estimates"):
            value = res["estimates"][0]["estimatedCost"]
    except Exception:
        value = None
    _estimate_cache[endpoint] = value
    return value
