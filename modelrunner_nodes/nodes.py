"""ComfyUI nodes: one typed node per catalog model, plus the generic and helper nodes.

Typed nodes are built at import time from the committed catalog.json, so ComfyUI
starts offline and no per-model code is committed.
"""
from __future__ import annotations

import io
import json
import logging
import re
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import client, mapper

try:  # present in every ComfyUI install; guarded so the mapper can be checked without one
    import comfy.model_management as model_management
    import comfy.utils as comfy_utils
    import folder_paths
except ImportError:  # pragma: no cover
    model_management = comfy_utils = folder_paths = None

# ComfyUI's typed VIDEO and 3D file objects, from its versioned public API. Each is
# looked up on its own: on a ComfyUI without one, that kind keeps STRING outputs, so
# every node still registers.
try:
    from comfy_api.latest import InputImpl as _ComfyInputs, Types as _ComfyTypes
except ImportError:  # pragma: no cover
    _ComfyInputs = _ComfyTypes = None
VideoFromFile = getattr(_ComfyInputs, "VideoFromFile", None)
VideoCodec = getattr(_ComfyTypes, "VideoCodec", None)
VideoContainer = getattr(_ComfyTypes, "VideoContainer", None)
File3D = getattr(_ComfyTypes, "File3D", None)

CATALOG = Path(__file__).resolve().parent.parent / "catalog.json"
MODEL_PAGE = "https://modelrunner.ai/models/{endpoint}"


# ---------------------------------------------------------------- media helpers

def _check_interrupt():
    if model_management is not None:
        model_management.throw_exception_if_processing_interrupted()


def image_to_png(image: torch.Tensor, index: int = 0) -> bytes:
    """One frame of a ComfyUI IMAGE batch ([B,H,W,C] floats 0..1) as PNG bytes."""
    arr = (image[index].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def upload_image(image: torch.Tensor, index: int = 0, name: str = "image") -> str:
    return client.upload_bytes(image_to_png(image, index), f"{name}.png", "image/png")


def bytes_to_image(data: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode to (IMAGE [1,H,W,3], MASK [1,H,W]). The mask follows LoadImage:
    1 - alpha, so a cut-out's background is 1; all zeros when there is no alpha."""
    img = Image.open(io.BytesIO(data))
    rgba = np.asarray(img.convert("RGBA")).astype(np.float32) / 255.0
    image = torch.from_numpy(rgba[..., :3])[None, ...]
    if "A" in img.getbands():
        mask = 1.0 - torch.from_numpy(rgba[..., 3])[None, ...]
    else:
        mask = torch.zeros(image.shape[:3])
    return image, mask


def urls_to_images(urls: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    frames = [bytes_to_image(client.download(u)) for u in urls]
    if not frames:
        raise client.ModelRunnerError("The run completed but returned no image URL.")
    # A batch needs one size; outputs of another size than the first are dropped.
    first = frames[0][0].shape[1:3]
    kept = [f for f in frames if f[0].shape[1:3] == first]
    if len(kept) < len(frames):
        logging.warning(f"[ModelRunner] {len(frames) - len(kept)} output image(s) differ in size from the first; "
              "they are in the urls output but not in the IMAGE batch.")
    return torch.cat([k[0] for k in kept], dim=0), torch.cat([k[1] for k in kept], dim=0)


def audio_to_wav(audio: dict) -> bytes:
    """ComfyUI AUDIO ({"waveform": [B,C,T], "sample_rate": int}) as 16-bit WAV bytes."""
    wav = audio["waveform"][0].clamp(-1, 1).cpu().numpy()  # [C, T]
    pcm = (wav.T * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(audio["sample_rate"]))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def video_to_mp4(video) -> bytes:
    """ComfyUI VIDEO as MP4 (H.264) bytes. Re-encoding, as ComfyUI's own API nodes do,
    keeps any trim or crop on the input and makes webm/mov sources a real MP4."""
    if VideoContainer is None:
        raise client.ModelRunnerError("This ComfyUI cannot encode VIDEO inputs; update ComfyUI, "
                                      "or pass the video URL in extra_json.")
    buf = io.BytesIO()
    video.save_to(buf, format=VideoContainer.MP4, codec=VideoCodec.H264)
    return buf.getvalue()


def load_audio(path: str) -> dict:
    """Decode an audio file to ComfyUI AUDIO with ComfyUI's own loader (PyAV),
    falling back to torchaudio on installs that predate it."""
    try:
        from comfy_extras.nodes_audio import load
    except ImportError:
        import torchaudio
        waveform, rate = torchaudio.load(path)
    else:
        waveform, rate = load(path)
    return {"waveform": waveform[None, ...], "sample_rate": rate}


def _ext(url: str, fallback: str) -> str:
    m = re.search(r"\.([a-zA-Z0-9]{2,5})(?:\?|$)", url)
    return m.group(1).lower() if m else fallback


def save_output(url: str, request_id: str, fallback_ext: str, index: int = 0) -> str:
    base = Path(folder_paths.get_output_directory()) if folder_paths else Path.cwd() / "output"
    folder = base / "modelrunner"
    folder.mkdir(parents=True, exist_ok=True)
    suffix = f"-{index}" if index else ""
    path = folder / f"{request_id}{suffix}.{_ext(url, fallback_ext)}"
    path.write_bytes(client.download(url))
    return str(path)


def _run(endpoint: str, payload: dict) -> dict:
    cost = client.estimate(endpoint)
    if cost:
        logging.info(f"[ModelRunner] {endpoint}: average cost per run ${float(cost):.4f}")
    bar = comfy_utils.ProgressBar(100) if comfy_utils else None

    def progress(status: str, elapsed: float):
        if bar is not None:
            # Queue progress has no percentage; creep towards 95 so the bar moves.
            bar.update_absolute(min(95, int(elapsed / (elapsed + 30) * 100)), 100)

    result = client.run(endpoint, payload, check_interrupt=_check_interrupt, on_progress=progress)
    if bar is not None:
        bar.update_absolute(100, 100)
    return result


# ---------------------------------------------------------------- typed nodes

# Multi-output runs (several images, several stems) return every URL, one per line.
# Video, 3D and audio nodes return the first result as a typed object for core nodes
# (Save Video, Save 3D Model, Save Audio), plus every saved file path and URL.
_FILES = (("STRING", "STRING", "STRING"), ("file_paths", "urls", "request_id"))


def _typed_first(comfy_type: str, name: str, available: bool) -> tuple:
    return ((comfy_type, *_FILES[0]), (name, *_FILES[1])) if available else _FILES


OUTPUTS = {
    "image": (("IMAGE", "MASK", "STRING", "STRING"), ("images", "mask", "urls", "request_id")),
    "video": _typed_first("VIDEO", "video", VideoFromFile is not None),
    "3d": _typed_first("FILE_3D", "model_3d", File3D is not None),
    "audio": _typed_first("AUDIO", "audio", True),
}
FALLBACK_EXT = {"video": "mp4", "3d": "glb", "audio": "mp3"}


class _ModelNode:
    """Base for generated nodes. Subclasses set ENDPOINT, PLANS, KIND, RETIRED_MESSAGE."""
    ENDPOINT = ""
    PLANS: list = []
    KIND = "image"
    RETIRED_MESSAGE = ""
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        types = mapper.input_types(cls.PLANS)
        types["optional"]["extra_json"] = ("STRING", {
            "default": "{}", "multiline": True,
            "tooltip": "Any other input fields as a JSON object; merged last, so it overrides widgets.",
        })
        types["optional"]["force_rerun"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "Run again even when no input changed. Every run is a paid request.",
        })
        return types

    @classmethod
    def IS_CHANGED(cls, force_rerun=False, **kwargs):
        # NaN never equals itself, so ComfyUI treats the node as changed every time.
        return float("nan") if force_rerun else ""

    def _payload(self, kwargs: dict) -> dict:
        extra = mapper.parse_extra_json(kwargs.get("extra_json", ""))
        # Fail before any upload when a required text field or video is still missing.
        # A VIDEO socket is always optional, so its URL can come from extra_json instead.
        missing = [p.name for p in self.PLANS if p.required and p.name not in extra and (
            (p.comfy_type == "STRING" and not str(kwargs.get(p.name) or "").strip())
            or (p.kind == "video" and kwargs.get(p.name) is None))]
        if missing:
            raise client.ModelRunnerError(f"{self.ENDPOINT}: fill in {', '.join(missing)} (required).")
        payload: dict = {}
        for p in self.PLANS:
            if p.kind == "object" and p.children:
                obj = {}
                for c in p.children:
                    ok, v = mapper.to_api_value(c, kwargs.get(mapper.widget_name(p.name, c.name)))
                    if ok:
                        obj[c.name] = v
                if obj or p.required:
                    payload[p.name] = obj
                continue
            if p.comfy_type == "JSON":
                continue
            value = kwargs.get(p.name)
            if value is None:
                continue
            if p.kind == "image":
                payload[p.name] = upload_image(value, 0, p.name)
            elif p.kind == "images":
                payload[p.name] = [upload_image(value, i, f"{p.name}-{i}") for i in range(value.shape[0])]
            elif p.kind == "audio":
                payload[p.name] = client.upload_bytes(audio_to_wav(value), f"{p.name}.wav", "audio/wav")
            elif p.kind == "video":
                if p.name not in extra:  # a URL in extra_json wins; skip the encode and upload
                    payload[p.name] = client.upload_bytes(video_to_mp4(value), f"{p.name}.mp4", "video/mp4")
            else:
                ok, v = mapper.to_api_value(p, value)
                if ok:
                    payload[p.name] = v
        payload.update(extra)
        return payload

    def run(self, **kwargs):
        if self.RETIRED_MESSAGE:
            raise client.ModelRunnerError(self.RETIRED_MESSAGE)
        result = _run(self.ENDPOINT, self._payload(kwargs))
        request_id = result.get("request_id") or result.get("id") or ""
        urls = client.output_urls(result.get("output"))
        if not urls:
            raise client.ModelRunnerError(
                f"Request {request_id} completed without an output URL: {json.dumps(result.get('output'))[:300]}")
        joined = "\n".join(urls)
        if self.KIND == "image":
            images, mask = urls_to_images(urls)
            return (images, mask, joined, request_id)
        paths = [save_output(u, request_id, FALLBACK_EXT[self.KIND], i) for i, u in enumerate(urls)]
        files = ("\n".join(paths), joined, request_id)
        # The typed first output exists only when OUTPUTS gave this kind one (see _typed_first).
        if self.KIND == "audio":
            return (load_audio(paths[0]), *files)
        if self.KIND == "video" and VideoFromFile is not None:
            return (VideoFromFile(paths[0]), *files)
        if self.KIND == "3d" and File3D is not None:
            return (File3D(paths[0]), *files)
        return files


def _description(entry: dict) -> str:
    lines = [entry["endpoint"], MODEL_PAGE.format(endpoint=entry["endpoint"])]
    if entry.get("averageCost"):
        lines.append(f"Average cost per run when this pack was released: ${float(entry['averageCost']):.4f}. "
                     "Each run is a paid request.")
    return "\n".join(lines)


def build_typed_node(entry: dict) -> tuple[str, type, str] | None:
    kind = mapper.output_kind(entry)
    if kind is None:
        return None
    nid = mapper.node_id(entry["endpoint"])
    retired = mapper.retired_message(entry)
    out_types, out_names = OUTPUTS[kind]
    cls = type(nid, (_ModelNode,), {
            "ENDPOINT": entry["endpoint"],
            "PLANS": mapper.plan_fields(entry),
            "KIND": kind,
            "RETIRED_MESSAGE": retired,
            "RETURN_TYPES": out_types,
            "RETURN_NAMES": out_names,
            "CATEGORY": "ModelRunner/Retired" if retired else f"ModelRunner/{'3D' if kind == '3d' else kind.title()}",
            "DESCRIPTION": _description(entry),
            # Video, 3D and audio nodes save their files under output/ themselves, so a graph that
            # ends in one must still run. Image nodes feed SaveImage / PreviewImage like any other.
            "OUTPUT_NODE": kind != "image",
        })
    label = entry["name"] + (f" ({entry['variantName']})" if entry.get("variantName") else "")
    return nid, cls, f"{label} [retired]" if retired else label


# ---------------------------------------------------------------- generic + helper nodes

class ModelRunnerRun:
    """Run any model by endpoint with a JSON input."""
    CATEGORY = "ModelRunner"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_json", "urls", "request_id")
    DESCRIPTION = ("Run any ModelRunner model by its owner/alias endpoint. Input fields are a JSON object; "
                   "file inputs must be URLs (use Upload Image). Language models take an OpenAI-style body, "
                   '{"messages": [{"role": "user", "content": "..."}]}, and output_json holds the reply. '
                   "Browse models at https://modelrunner.ai/models")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint": ("STRING", {"default": "", "tooltip": "owner/alias, e.g. from a modelrunner.ai model URL"}),
                "input_json": ("STRING", {"default": "{\n  \"prompt\": \"\"\n}", "multiline": True}),
            },
            "optional": {"force_rerun": ("BOOLEAN", {"default": False})},
        }

    @classmethod
    def IS_CHANGED(cls, force_rerun=False, **kwargs):
        return float("nan") if force_rerun else ""

    def run(self, endpoint, input_json, force_rerun=False):
        endpoint = endpoint.strip().strip("/")
        if not endpoint:
            raise client.ModelRunnerError("Set the endpoint, e.g. owner/alias.")
        payload = mapper.parse_extra_json(input_json)
        if client.is_chat(endpoint):
            _check_interrupt()
            result = client.chat(endpoint, payload)
        else:
            result = _run(endpoint, payload)
        output = result.get("output")
        return (json.dumps(output, indent=2), "\n".join(client.output_urls(output)),
                result.get("request_id") or result.get("id") or "")


class ModelRunnerUploadImage:
    """Upload an IMAGE and return its URL, for the generic node or any URL input."""
    CATEGORY = "ModelRunner"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("urls",)
    DESCRIPTION = "Uploads each image in the batch as a lossless PNG and returns the URLs, one per line."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def run(self, image):
        return ("\n".join(upload_image(image, i, f"image-{i}") for i in range(image.shape[0])),)


class ModelRunnerDownloadImage:
    """Fetch image URLs into an IMAGE batch."""
    CATEGORY = "ModelRunner"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"urls": ("STRING", {"default": "", "multiline": True})}}

    def run(self, urls):
        found = [u for u in re.split(r"\s+", urls or "") if u.startswith(("http://", "https://"))]
        return urls_to_images(found)


def build_all() -> tuple[dict, dict]:
    """Register the generic and helper nodes first, then one node per catalog row.
    A row that fails to build is skipped with one log line, so one bad row never
    takes the rest of the pack down."""
    classes, names = {}, {}
    missing = [n for n, t in (("VIDEO", VideoFromFile), ("3D file", File3D)) if t is None]
    if missing:
        logging.warning(f"[ModelRunner] this ComfyUI has no {' or '.join(missing)} type; those nodes output "
                        "file paths and URLs only. Update ComfyUI to get typed outputs.")
    for cls, label in ((ModelRunnerRun, "ModelRunner Run (any model)"),
                       (ModelRunnerUploadImage, "ModelRunner Upload Image"),
                       (ModelRunnerDownloadImage, "ModelRunner Download Image")):
        classes[cls.__name__] = cls
        names[cls.__name__] = label
    try:
        catalog = json.loads(CATALOG.read_text())["models"]
    except Exception as e:
        logging.warning(f"[ModelRunner] catalog.json unreadable, only the generic node is available: {e}")
        return classes, names
    skipped = 0
    for entry in catalog:
        try:
            built = build_typed_node(entry)
            if built is None:
                continue
            nid, cls, label = built
            if nid in classes:
                raise ValueError(f"duplicate node id {nid}")
            classes[nid], names[nid] = cls, label
        except Exception as e:
            skipped += 1
            logging.warning(f"[ModelRunner] skipped {entry.get('endpoint', '?')}: {e}")
    logging.info(f"[ModelRunner] loaded {len(classes)} nodes" + (f", skipped {skipped}" if skipped else ""))
    return classes, names
