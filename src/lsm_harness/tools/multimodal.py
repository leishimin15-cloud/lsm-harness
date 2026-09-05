"""Multimodal input helpers — image loading and base64 encoding.

Supports the ``@path/to/image.png`` syntax in user messages, converting
local image files to data URLs that the model's vision API can consume.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


# Supported image formats
_SUPPORTED_MIMES: frozenset[str] = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})

_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


def load_image_as_data_url(path: str) -> str | None:
    """Load an image file and return a ``data:`` URL.

    Returns None if the file doesn't exist, is too large, or has an
    unsupported format.
    """
    filepath = Path(path).expanduser().resolve()

    if not filepath.exists():
        return None
    if not filepath.is_file():
        return None
    if filepath.stat().st_size > _MAX_IMAGE_BYTES:
        return None

    mime_type, _ = mimetypes.guess_type(str(filepath))
    if mime_type not in _SUPPORTED_MIMES:
        # Try to detect from extension
        suffix = filepath.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_map.get(suffix)
        if not mime_type:
            return None

    raw = filepath.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parse_multimodal_message(
    text: str,
) -> dict | None:
    """Parse a user message for ``@path`` image references.

    If the message starts with ``@`` followed by a valid image path,
    returns a ``content`` array suitable for vision API.  Otherwise
    returns None (use plain text).

    Examples::

        "@screenshot.png What's wrong?" → content array
        "Hello"                          → None (plain text)
    """
    text = text.strip()
    if not text.startswith("@"):
        return None

    # Split into image path + text question
    parts = text.split(maxsplit=1)
    path_candidate = parts[0][1:]  # strip the @
    question = parts[1] if len(parts) > 1 else "描述这张图片的内容。"

    data_url = load_image_as_data_url(path_candidate)
    if data_url is None:
        return None

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
        ],
    }


def has_image(content: str | list | None) -> bool:
    """Check if a message content contains an image."""
    if isinstance(content, list):
        return any(
            item.get("type") == "image_url"
            for item in content
            if isinstance(item, dict)
        )
    return False


def extract_text(content: str | list | None) -> str:
    """Extract the text portion of a content array."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(parts)
