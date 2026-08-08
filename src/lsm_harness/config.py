"""Configuration with LSM-first and Waku-compatible environment names."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


DOTENV_PATH = find_dotenv(usecwd=True)
if DOTENV_PATH:
    load_dotenv(DOTENV_PATH)


def _value(name: str, default: str = "") -> str:
    return os.getenv(f"LSM_{name}") or os.getenv(f"WAKU_{name}") or default


def _integer(name: str, default: int) -> int:
    return int(_value(name, str(default)))


def _api_key() -> str:
    value = (
        os.getenv("LSM_API_KEY")
        or os.getenv("WAKU_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or ""
    ).strip()
    return "" if value in {"your-key-here", "replace-me"} else value


@dataclass
class Settings:
    provider: str = field(default_factory=lambda: _value("PROVIDER", ""))
    api_key: str = field(default_factory=_api_key)
    base_url: str = field(default_factory=lambda: _value("BASE_URL", ""))
    model: str = field(default_factory=lambda: _value("MODEL", ""))
    small_model: str = field(default_factory=lambda: _value("SMALL_MODEL", ""))
    home: Path = field(default_factory=lambda: Path(_value("HOME", ".lsm")))
    thinking: str = field(default_factory=lambda: _value("THINKING", "disabled"))
    max_iterations: int = field(default_factory=lambda: _integer("MAX_ITERATIONS", 10))
    max_tokens: int = field(default_factory=lambda: _integer("MAX_TOKENS", 8192))
    history_turns: int = field(default_factory=lambda: _integer("HISTORY_TURNS", 12))
    consolidate_every: int = field(default_factory=lambda: _integer("CONSOLIDATE_EVERY", 6))
    retrieval_top_k: int = field(default_factory=lambda: _integer("RETRIEVAL_TOP_K", 4))
    context_budget_tokens: int = field(
        default_factory=lambda: _integer("CONTEXT_BUDGET_TOKENS", 24000)
    )
    context_compression_tokens: int = field(
        default_factory=lambda: _integer("CONTEXT_COMPRESSION_TOKENS", 18000)
    )
    context_recent_turns: int = field(
        default_factory=lambda: _integer("CONTEXT_RECENT_TURNS", 6)
    )
    summary_max_tokens: int = field(
        default_factory=lambda: _integer("SUMMARY_MAX_TOKENS", 1200)
    )

    def ensure_home(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True)
        for name in ("traces", "skills", "outbox"):
            (self.home / name).mkdir(exist_ok=True)
        return self.home
