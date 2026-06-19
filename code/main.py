import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import urllib.error as urlerror
import urllib.request as urlrequest
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

try:
    import anthropic as _anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

# HARDCODED VALUES

ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ALLOWED_ISSUE_TYPE = {"dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part", "torn_packaging",
                      "crushed_packaging", "water_damage", "stain", "none", "unknown"}

ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}

ALLOWED_FLAGS = {"none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle", "wrong_object",
                 "wrong_object_part", "damage_not_visible", "claim_mismatch", "possible_maniplation", "non_original_image",
                 "text_instruction_present", "user_history_risk", "manual_review_required"}

OBJECT_PARTS = {
    "car": {"front_bumper", "rear_bumper", "door", "windshield", "side_mirror", "headlight", "taillight", "fender",
            "quarter_panel", "body", "unknown"},
    "laptop": {"screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body", "unknown"},
    "package": {"box", "package_corner", "package_side", "seal", "label", "contents", "item", "unknown"}
}

# CACHE FUNCTIONALITY

class Cache:
    """JSON cache, keyed with SHA256 hashes of the input data to the model, to store and retrieve model responses."""

    def __init__(self, cache_path: Path):
        self.path = cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                self._data: dict = json.load(f)
        else:
            self._data = {}

    def get(self, key: str) -> Optional[dict]:
        """Retrieve a cached response by key."""
        return self._data.get(key)
    
    def set(self, key: str, value: dict):
        """Store a response in the cache under the given key."""
        self._data[key] = value
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def __len__(self) -> int:
        return len(self._data)
    

def make_cache_key(provider: str, model : str, user_id: str, image_paths_str: str, user_claim: str, claim_object: str) -> str:
    """Create a SHA256 hash key for caching based on the input parameters."""
    raw = f"{provider}|{model}|{user_id}|{image_paths_str}|{user_claim}|{claim_object}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# MODEL INTERACTION

def query_model(prompt: str, images_b64: str, media_types: list, model: str) -> str:
    """Query the model with the given prompt and images, returning the response text."""
    if not _HAS_ANTHROPIC:
        raise ImportError("Anthropic library is required to query the model. Please install it with 'pip install anthropic'.")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set." \
        "Please set it to your Anthropic API key using a .env file, or through PowerShell/Bash.")
    client = _anthropic.Client(api_key=api_key)
    content: list = []
    for b64, mime in zip(images_b64, media_types):
        content.append({
            "type": "input_image",
            "source": {"type": "base64", "media_type": mime, "data": b64}
        })
    content.append({"type": "text", "text": prompt})
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages = [{"role": "user", "content": content}]
    )
    return response.content[0].text
    

# CSV PROCESSING

def load_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
    
def build_user_history_map(rows: list) -> dict:
    return {r["user_id"]: r for r in rows}

# IMAGE PROCESSING

_EXT_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".png": "image/png",
}


def _detect_mime(data: bytes) -> Optional[str]:
    """Detect the MIME type of the given image from magic bytes, ignoring file extension."""
    if data[:4] == b"RIFF" and data [8:12] == b"WEBP":
        return "image/webp"
    elif data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return None


def load_images(images_paths_str: str, dataset_root: Path) -> tuple:
    """Returns (b64_list, image_ids, missing_ids, media_types)."""
    raw_paths = [p.strip() for p in images_paths_str.split(",") if p.strip()]
    b64_list = []
    ids = []
    missing = []
    media_types = []
    for p in raw_paths:
        img_id = Path(p).stem
        ids.append(img_id)
        full = dataset_root / p
        if full.exists():
            with open(full, "rb") as f:
                data = f.read()
            mime = _detect_mime(data) or _EXT_TO_MIME.get(full.suffix.lower())
            if mime:
                b64_list.append(base64.b64encode(data).decode("utf-8"))
                media_types.append(mime)
            else:
                missing.append(img_id)
        else:
            missing.append(img_id)
    return b64_list, ids, missing, media_types
        