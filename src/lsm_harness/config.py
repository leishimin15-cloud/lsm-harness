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


def _cache_retention() -> str:
    value = _value("CACHE_RETENTION", "short").lower()
    if value not in {"none", "short", "long"}:
        raise ValueError(
            "LSM_CACHE_RETENTION must be one of: none, short, long"
        )
    return value


def _api_key() -> str:
    generic = (os.getenv("LSM_API_KEY") or os.getenv("WAKU_API_KEY") or "").strip()
    if generic:
        return "" if generic in {"your-key-here", "replace-me"} else generic

    provider_keys = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "xai": "XAI_API_KEY",
        "kimi": "MOONSHOT_API_KEY",
        "glm": "ZHIPU_API_KEY",
        "minimax": "MINIMAX_API_KEY",
    }
    selected = _value("PROVIDER", "").lower()
    if selected:
        value = os.getenv(provider_keys.get(selected, ""), "").strip()
    else:
        value = next(
            (
                os.getenv(env_name, "").strip()
                for env_name in provider_keys.values()
                if os.getenv(env_name, "").strip()
            ),
            "",
        )
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
    context_budget_tokens: int = field(
        default_factory=lambda: _integer("CONTEXT_BUDGET_TOKENS", 24000)
    )
    context_compression_tokens: int = field(
        default_factory=lambda: _integer("CONTEXT_COMPRESSION_TOKENS", 18000)
    )
    context_recent_turns: int = field(
        default_factory=lambda: _integer("CONTEXT_RECENT_TURNS", 6)
    )
    context_keep_recent_tokens: int = field(
        default_factory=lambda: _integer("CONTEXT_KEEP_RECENT_TOKENS", 6000)
    )
    summary_max_tokens: int = field(
        default_factory=lambda: _integer("SUMMARY_MAX_TOKENS", 1200)
    )
    # ── shell tool ──────────────────────────────────────────
    shell_timeout: int = field(
        default_factory=lambda: _integer("SHELL_TIMEOUT", 60)
    )
    shell_allow: str = field(
        default_factory=lambda: _value("SHELL_ALLOW", "")
    )
    shell_deny: str = field(
        default_factory=lambda: _value("SHELL_DENY", "")
    )
    # ── web tools ───────────────────────────────────────────
    web_search_provider: str = field(
        default_factory=lambda: _value("WEB_SEARCH_PROVIDER", "duckduckgo")
    )
    web_fetch_max_chars: int = field(
        default_factory=lambda: _integer("WEB_FETCH_MAX_CHARS", 8000)
    )
    # ── agent runner ────────────────────────────────────────
    max_empty_retries: int = field(
        default_factory=lambda: _integer("MAX_EMPTY_RETRIES", 2)
    )
    max_length_recoveries: int = field(
        default_factory=lambda: _integer("MAX_LENGTH_RECOVERIES", 3)
    )
    max_model_retries: int = field(
        default_factory=lambda: _integer("MAX_MODEL_RETRIES", 2)
    )
    cache_retention: str = field(
        default_factory=_cache_retention
    )
    # ── subagent ───────────────────────────────────────────
    subagent_max_concurrent: int = field(
        default_factory=lambda: _integer("SUBAGENT_MAX_CONCURRENT", 3)
    )
    # ── context governance ─────────────────────────────────
    governance_max_result_chars: int = field(
        default_factory=lambda: _integer("GOVERNANCE_MAX_RESULT_CHARS", 4000)
    )
    governance_offload_threshold: int = field(
        default_factory=lambda: _integer("GOVERNANCE_OFFLOAD_THRESHOLD", 12000)
    )
    # ── sandbox ───────────────────────────────────────────
    sandbox_enabled: bool = field(
        default_factory=lambda: _value("SANDBOX_ENABLED", "").lower() in ("1", "true", "yes")
    )
    sandbox_project_dir: str = field(
        default_factory=lambda: _value("SANDBOX_PROJECT_DIR", "")
    )

    def ensure_home(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True)
        for name in ("traces", "skills", "outbox"):
            (self.home / name).mkdir(exist_ok=True)
        return self.home
