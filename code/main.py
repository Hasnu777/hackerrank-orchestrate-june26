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

