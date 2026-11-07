"""ModelRunner nodes for ComfyUI: hundreds of image, video, audio and 3D models behind one API key."""
from .modelrunner_nodes.nodes import build_all

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = build_all()
WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
