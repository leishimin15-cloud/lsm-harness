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


def _optional_positive_integer(name: str) -> int | None:
    """Read an optional positive limit; blank or 0 means unlimited."""
    raw = _value(name, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(f"LSM_{name} must be 0 or a positive integer")
    return value or None


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
        "google": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "xai": "XAI_API_KEY",
        "kimi-coding": "KIMI_API_KEY",
        "moonshotai": "MOONSHOT_API_KEY",
        "moonshotai-cn": "MOONSHOT_API_KEY",
        "glm": "ZHIPU_API_KEY",
        "zai": "ZAI_API_KEY",
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
    # Pi 七档:off/minimal/low/medium/high/xhigh/max(旧三档
    # disabled/auto/enabled 在 CodingSession 读取处转换,不在这里)。
    thinking: str = field(default_factory=lambda: _value("THINKING", "off"))
    system_prompt: str = field(default_factory=lambda: _value("SYSTEM_PROMPT", ""))
    # Pi parity: interactive runs have no fixed Turn cap.  Products such
    # as eval may still opt into a positive limit; 0 also means unlimited.
    max_iterations: int | None = field(
        default_factory=lambda: _optional_positive_integer("MAX_ITERATIONS")
    )
    max_tokens: int = field(default_factory=lambda: _integer("MAX_TOKENS", 8192))
    history_turns: int = field(default_factory=lambda: _integer("HISTORY_TURNS", 12))
    # 0 = 跟随当前模型的 context_window(Pi 行为);>0 = 用户显式
    # override(对小窗模型压预算)。压缩红线 = effective - reserve。
    context_budget_tokens: int = field(
        default_factory=lambda: _integer("CONTEXT_BUDGET_TOKENS", 0)
    )
    # 红线后的保留区(Pi reserveTokens)。0 = 自动:min(16384,
    # budget//4)——大窗模型(1M)拿满 16384,小预算不退化(24k 兜底
    # 时红线 = 18000,与旧默认一致);>0 = 显式指定(至多留 1 的红线)。
    context_reserve_tokens: int = field(
        default_factory=lambda: _integer("CONTEXT_RESERVE_TOKENS", 0)
    )
    context_recent_turns: int = field(
        default_factory=lambda: _integer("CONTEXT_RECENT_TURNS", 6)
    )
    context_keep_recent_tokens: int = field(
        default_factory=lambda: _integer("CONTEXT_KEEP_RECENT_TOKENS", 20000)
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
    # ── 执行审批 ──────────────────────────────────────────
    # off(默认,无审批门,保持既有行为)/ policy(headless 自动策略:
    # 工作区写自动放行,外部副作用默认拒)/ prompt(CLI 交互 y/n)。
    approval: str = field(default_factory=lambda: _value("APPROVAL", "off"))

    def ensure_home(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True)
        for name in ("traces", "skills", "outbox"):
            (self.home / name).mkdir(exist_ok=True)
        return self.home
