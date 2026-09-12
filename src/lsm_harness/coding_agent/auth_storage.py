"""Local API-key credential storage for product frontends.

The AI/provider layer knows how to *use* a credential.  This module owns the
coding-agent product concern of persisting credentials entered through
``/login``.  The on-disk shape intentionally mirrors Pi's small typed envelope
instead of mixing secrets into ``Settings`` or session JSONL files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class AuthStorageError(RuntimeError):
    """The local credential file could not be read or written safely."""


def auth_path(home: Path) -> Path:
    """Return the credential file owned by one LSM home."""
    return home.expanduser() / "auth.json"


def _read(home: Path) -> dict[str, dict[str, str]]:
    path = auth_path(home)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthStorageError(f"Could not read credentials from {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise AuthStorageError(f"Credential file must contain an object: {path}")
    credentials: dict[str, dict[str, str]] = {}
    for provider, value in raw.items():
        if (
            isinstance(provider, str)
            and isinstance(value, dict)
            and value.get("type") == "api_key"
            and isinstance(value.get("key"), str)
        ):
            credentials[provider] = {
                "type": "api_key",
                "key": value["key"],
            }
    return credentials


def read_api_key(home: Path, provider: str) -> str:
    """Read one stored key without exposing it through provider metadata."""
    value = _read(home).get(provider, {})
    return value.get("key", "").strip()


def save_api_key(home: Path, provider: str, api_key: str) -> Path:
    """Atomically store one API key in a user-only file."""
    key = api_key.strip()
    if not key or key in {"your-key-here", "replace-me"}:
        raise AuthStorageError("API key cannot be empty or a placeholder")
    try:
        key.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise AuthStorageError(
            "API key contains non-ASCII characters; paste the original key again"
        ) from exc

    path = auth_path(home)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        credentials = _read(home)
        credentials[provider] = {"type": "api_key", "key": key}
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(credentials, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)
    except OSError as exc:
        raise AuthStorageError(f"Could not save credentials to {path}: {exc}") from exc
    return path


__all__ = ["AuthStorageError", "auth_path", "read_api_key", "save_api_key"]
