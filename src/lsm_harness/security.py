"""Small, deterministic redaction helpers for durable operational records."""

from __future__ import annotations

import re
from typing import Any


_TOKEN_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[opsu]_[A-Za-z0-9]{20,})\b"
)


def redact_text(text: str, secrets: tuple[str, ...] = ()) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return _TOKEN_PATTERN.sub("[REDACTED]", redacted)


def redact_data(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, dict):
        return {key: redact_data(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_data(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item, secrets) for item in value)
    return value
