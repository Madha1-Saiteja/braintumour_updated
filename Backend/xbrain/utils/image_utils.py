"""
image_utils.py
Shared image helpers — encoding, decoding, numpy ↔ base64.
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image


def read_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded bytes → RGB numpy array."""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image data")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def ndarray_to_base64(img: np.ndarray, fmt: str = "PNG") -> str:
    """Convert uint8 numpy image → base64 string for JSON transport."""
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_ndarray(b64: str) -> np.ndarray:
    """Convert base64 string → RGB numpy array."""
    data = base64.b64decode(b64)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid base64 image")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_to_base64(mask: np.ndarray) -> str:
    """Convert float binary mask [0,1] → base64 PNG."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    pil = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")