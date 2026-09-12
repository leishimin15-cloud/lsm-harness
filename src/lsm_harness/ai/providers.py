"""Service catalog plus the legacy ``ModelClient`` construction facade.

The Agent Loop consumes the canonical ``StreamFunction`` protocol. Existing
Memory, RAG, tests, and external callers can keep using ``get_client()`` while
they migrate from ``complete`` / ``stream_complete``.

Two wire formats:
  - **openai-completions** – OpenAI, DeepSeek, Gemini, OpenRouter, xAI,
    Moonshot AI
  - **anthropic-messages** – Anthropic, Kimi Coding, GLM/Z.ai, MiniMax

Pick with LSM_PROVIDER=... and set that provider's API key.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any, Iterator, Literal

from lsm_harness.ai.types import (
    AIContext,
    Model,
    ModelResponse,
    StreamOptions,
    StreamDelta,
    ToolCall,
    Usage,
    normalize_stop_reason,
)
from lsm_harness.ai.api.anthropic_messages import (
    build_anthropic_request,
    stream_anthropic_client,
    stream_anthropic_messages,
)
from lsm_harness.ai.api.openai_compat import (
    stream_openai_client,
    stream_openai_compat,
)
from lsm_harness.ai.registry import ApiProvider, register_api_provider


# ── provider registry ─────────────────────────────────────────────


class ProviderConfigurationError(RuntimeError):
    """Provider selection/authentication is invalid but recoverable by a UI."""


class ProviderAuthError(ProviderConfigurationError):
    """The selected provider has no usable API credential."""


@dataclass(frozen=True)
class Provider:
    """Metadata for one LLM provider."""

    api: str  # API translator key
    key_env: str  # env var for the API key
    base_url: str | None  # override endpoint; None = SDK default
    model: str  # default main model
    small_model: str  # default cheap model (gate + summary)
    name: str = ""
    api_key_name: str = "API key"
    # Where to get a key (shown in error messages)
    key_url: str = ""
    # Model limits; resolve_max_tokens clamps against these. Conservative
    # defaults so unverified providers stay safe — verify against each
    # provider's docs before tightening.
    context_window: int = 128_000
    max_output_tokens: int = 8_192
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    cache_read_cost_per_million: float = 0.0
    cache_write_cost_per_million: float = 0.0
    # Selectable main-model catalog. Empty means only ``model`` is exposed.
    models: tuple[str, ...] = ()
    # API-key login uses the SDK's native key field. OAuth-derived credentials
    # may instead select Bearer when that separate flow is implemented.
    auth_mode: Literal["api_key", "bearer"] = "api_key"
    headers: dict[str, str] = field(default_factory=dict)
    model_context_windows: dict[str, int] = field(default_factory=dict)
    model_max_output_tokens: dict[str, int] = field(default_factory=dict)
    model_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


_MOONSHOT_MODELS = (
    "kimi-k2-0711-preview",
    "kimi-k2-0905-preview",
    "kimi-k2-thinking",
    "kimi-k2-thinking-turbo",
    "kimi-k2-turbo-preview",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "kimi-k2.7-code-highspeed",
    "kimi-k3",
)


_KIMI_CODING_METADATA: dict[str, dict[str, Any]] = {
    "k3": {
        "name": "Kimi K3",
        "reasoning": True,
        "input_modalities": ("text", "image"),
        "context_window": 1_048_576,
        "max_tokens": 131_072,
        "cost": (3.0, 15.0, 0.3, 0.0),
        "thinking_level_map": {
            "off": None,
            "minimal": None,
            "low": "low",
            "medium": None,
            "high": "high",
            "xhigh": None,
            "max": "max",
        },
        "allow_empty_thinking_signature": True,
        "force_adaptive_thinking": True,
    },
    "k3-256k": {
        "name": "Kimi K3-256K",
        "reasoning": True,
        "input_modalities": ("text", "image"),
        "context_window": 262_144,
        "max_tokens": 131_072,
        "cost": (0.0, 0.0, 0.0, 0.0),
        "thinking_level_map": {
            "off": None,
            "minimal": None,
            "low": "low",
            "medium": None,
            "high": "high",
            "xhigh": None,
            "max": "max",
        },
        "force_adaptive_thinking": True,
    },
    "kimi-for-coding": {
        "name": "Kimi K2.7 Code",
        "reasoning": True,
        "input_modalities": ("text", "image"),
        "context_window": 262_144,
        "max_tokens": 32_768,
        "cost": (0.95, 4.0, 0.19, 0.0),
        "force_adaptive_thinking": True,
        "allow_empty_thinking_signature": True,
    },
    "kimi-for-coding-highspeed": {
        "name": "Kimi For Coding HighSpeed",
        "reasoning": True,
        "input_modalities": ("text", "image"),
        "context_window": 262_144,
        "max_tokens": 32_768,
        "cost": (1.9, 8.0, 0.38, 0.0),
        "force_adaptive_thinking": True,
    },
}


def _moonshot_model(
    name: str,
    *,
    reasoning: bool,
    cost: tuple[float, float, float, float],
    context_window: int = 262_144,
    max_tokens: int = 262_144,
    image: bool = False,
    thinking_level_map: dict[str, str | None] | None = None,
    thinking_format: str = "deepseek",
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "name": name,
        "reasoning": reasoning,
        "input_modalities": ("text", "image") if image else ("text",),
        "context_window": context_window,
        "max_tokens": max_tokens,
        "cost": cost,
        "thinking_format": thinking_format,
    }
    if thinking_level_map is not None:
        value["thinking_level_map"] = thinking_level_map
    return value


_MOONSHOT_METADATA: dict[str, dict[str, Any]] = {
    "kimi-k2-0711-preview": _moonshot_model(
        "Kimi K2 0711", reasoning=False, cost=(0.6, 2.5, 0.15, 0.0),
        context_window=131_072, max_tokens=16_384,
    ),
    "kimi-k2-0905-preview": _moonshot_model(
        "Kimi K2 0905", reasoning=False, cost=(0.6, 2.5, 0.15, 0.0),
    ),
    "kimi-k2-thinking": _moonshot_model(
        "Kimi K2 Thinking", reasoning=True, cost=(0.6, 2.5, 0.15, 0.0),
    ),
    "kimi-k2-thinking-turbo": _moonshot_model(
        "Kimi K2 Thinking Turbo", reasoning=True,
        cost=(1.15, 8.0, 0.15, 0.0),
    ),
    "kimi-k2-turbo-preview": _moonshot_model(
        "Kimi K2 Turbo", reasoning=False, cost=(2.4, 10.0, 0.6, 0.0),
    ),
    "kimi-k2.5": _moonshot_model(
        "Kimi K2.5", reasoning=True, cost=(0.6, 3.0, 0.1, 0.0),
        image=True,
    ),
    "kimi-k2.6": _moonshot_model(
        "Kimi K2.6", reasoning=True, cost=(0.95, 4.0, 0.16, 0.0),
        image=True,
    ),
    "kimi-k2.7-code": _moonshot_model(
        "Kimi K2.7 Code", reasoning=True, cost=(0.95, 4.0, 0.19, 0.0),
        image=True, thinking_level_map={"off": None},
    ),
    "kimi-k2.7-code-highspeed": _moonshot_model(
        "Kimi K2.7 Code HighSpeed", reasoning=True,
        cost=(1.9, 8.0, 0.38, 0.0), image=True,
        thinking_level_map={"off": None},
    ),
    "kimi-k3": _moonshot_model(
        "Kimi K3", reasoning=True, cost=(3.0, 15.0, 0.3, 0.0),
        context_window=1_048_576, max_tokens=131_072, image=True,
        thinking_level_map={
            "off": None,
            "minimal": None,
            "low": "low",
            "medium": None,
            "high": "high",
            "xhigh": None,
            "max": "max",
        },
        thinking_format="reasoning_effort",
    ),
}

_PROVIDER_ALIASES = {
    "gemini": "google",
    "glm": "zai",
}


PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider(
        "openai-completions", "DEEPSEEK_API_KEY", "https://api.deepseek.com",
        "deepseek-v4-pro", "deepseek-v4-flash",
        name="DeepSeek", api_key_name="DeepSeek API key",
        key_url="https://platform.deepseek.com/api_keys",
    ),
    "openai": Provider(
        "openai-completions", "OPENAI_API_KEY", None,
        "gpt-4o", "gpt-4o-mini",
        name="OpenAI", api_key_name="OpenAI API key",
        key_url="https://platform.openai.com/api-keys",
        context_window=128_000, max_output_tokens=16_384,
    ),
    "anthropic": Provider(
        "anthropic-messages", "ANTHROPIC_API_KEY", None,
        "claude-sonnet-4-20250514", "claude-haiku-3-5-20241022",
        name="Anthropic", api_key_name="Anthropic API key",
        key_url="https://console.anthropic.com/settings/keys",
        context_window=200_000, max_output_tokens=64_000,
    ),
    "google": Provider(
        "openai-completions", "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.5-flash", "gemini-2.5-flash",
        name="Google", api_key_name="Gemini API key",
        key_url="https://aistudio.google.com/apikey",
        context_window=1_000_000, max_output_tokens=65_536,
    ),
    "openrouter": Provider(
        "openai-completions", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
        "anthropic/claude-sonnet-4", "google/gemini-2.5-flash",
        name="OpenRouter", api_key_name="OpenRouter API key",
        key_url="https://openrouter.ai/keys",
        auth_mode="bearer",
    ),
    "xai": Provider(
        "openai-completions", "XAI_API_KEY", "https://api.x.ai/v1",
        "grok-4-latest", "grok-4-fast-latest",
        name="xAI", api_key_name="xAI API key",
        key_url="https://console.x.ai",
    ),
    "kimi-coding": Provider(
        "anthropic-messages", "KIMI_API_KEY", "https://api.kimi.com/coding",
        "kimi-for-coding", "kimi-for-coding",
        name="Kimi For Coding", api_key_name="Kimi API key",
        key_url="https://www.kimi.com/code",
        context_window=262_144, max_output_tokens=32_768,
        models=(
            "k3",
            "k3-256k",
            "kimi-for-coding",
            "kimi-for-coding-highspeed",
        ),
        model_metadata=_KIMI_CODING_METADATA,
    ),
    "moonshotai": Provider(
        "openai-completions", "MOONSHOT_API_KEY", "https://api.moonshot.ai/v1",
        "kimi-k2.6", "kimi-k2.6",
        name="Moonshot AI", api_key_name="Moonshot AI API key",
        key_url="https://platform.moonshot.ai/console/api-keys",
        context_window=262_144, max_output_tokens=262_144,
        models=_MOONSHOT_MODELS,
        model_metadata=_MOONSHOT_METADATA,
    ),
    "moonshotai-cn": Provider(
        "openai-completions", "MOONSHOT_API_KEY", "https://api.moonshot.cn/v1",
        "kimi-k2.6", "kimi-k2.6",
        name="Moonshot AI CN", api_key_name="Moonshot AI API key",
        key_url="https://platform.moonshot.cn/console/api-keys",
        context_window=262_144, max_output_tokens=262_144,
        models=_MOONSHOT_MODELS,
        model_metadata=_MOONSHOT_METADATA,
    ),
    "zai": Provider(
        "openai-completions", "ZAI_API_KEY",
        "https://api.z.ai/api/coding/paas/v4",
        "glm-5.2", "glm-5-turbo",
        name="Z.AI", api_key_name="Z.AI API key",
        key_url="https://z.ai/manage-apikey/apikey-list",
    ),
    "minimax": Provider(
        "anthropic-messages", "MINIMAX_API_KEY", "https://api.minimax.io/anthropic",
        "MiniMax-M3", "MiniMax-M2",
        name="MiniMax", api_key_name="MiniMax API key",
        key_url="https://platform.minimaxi.com/user-center/basic-information",
    ),
}


_ADDITIONAL_PROVIDER_SPECS: dict[
    str, tuple[str, str, str, Literal["api_key", "bearer"]]
] = {
    "amazon-bedrock": (
        "Amazon Bedrock", "AWS_BEARER_TOKEN_BEDROCK",
        "AWS credentials or bearer token", "bearer",
    ),
    "ant-ling": ("Ant Ling", "ANT_LING_API_KEY", "Ant Ling API key", "api_key"),
    "azure-openai-responses": (
        "Azure OpenAI", "AZURE_OPENAI_API_KEY", "Azure OpenAI API key", "api_key",
    ),
    "baseten": ("Baseten", "BASETEN_API_KEY", "Baseten API key", "api_key"),
    "cerebras": ("Cerebras", "CEREBRAS_API_KEY", "Cerebras API key", "api_key"),
    "cloudflare-ai-gateway": (
        "Cloudflare AI Gateway", "CLOUDFLARE_API_KEY", "Cloudflare API token", "bearer",
    ),
    "cloudflare-workers-ai": (
        "Cloudflare Workers AI", "CLOUDFLARE_API_KEY", "Cloudflare API token", "bearer",
    ),
    "fireworks": ("Fireworks", "FIREWORKS_API_KEY", "Fireworks API key", "bearer"),
    "github-copilot": (
        "GitHub Copilot", "COPILOT_GITHUB_TOKEN", "GitHub Copilot token", "bearer",
    ),
    "google-vertex": (
        "Google Vertex AI", "GOOGLE_CLOUD_API_KEY", "Google Cloud API key", "api_key",
    ),
    "groq": ("Groq", "GROQ_API_KEY", "Groq API key", "api_key"),
    "huggingface": ("Hugging Face", "HF_TOKEN", "Hugging Face token", "bearer"),
    "minimax-cn": (
        "MiniMax CN", "MINIMAX_CN_API_KEY", "MiniMax CN API key", "api_key",
    ),
    "mistral": ("Mistral", "MISTRAL_API_KEY", "Mistral API key", "api_key"),
    "nvidia": ("NVIDIA", "NVIDIA_API_KEY", "NVIDIA API key", "api_key"),
    "openai-codex": ("OpenAI Codex", "", "OpenAI account", "bearer"),
    "opencode": ("OpenCode Zen", "OPENCODE_API_KEY", "OpenCode API key", "bearer"),
    "opencode-go": ("OpenCode Go", "OPENCODE_API_KEY", "OpenCode API key", "bearer"),
    "qwen-token-plan": (
        "Qwen Token Plan", "QWEN_TOKEN_PLAN_API_KEY", "Qwen Token Plan API key", "api_key",
    ),
    "qwen-token-plan-cn": (
        "Qwen Token Plan CN", "QWEN_TOKEN_PLAN_CN_API_KEY",
        "Qwen Token Plan CN API key", "api_key",
    ),
    "together": ("Together", "TOGETHER_API_KEY", "Together API key", "api_key"),
    "vercel-ai-gateway": (
        "Vercel AI Gateway", "AI_GATEWAY_API_KEY", "Vercel AI Gateway API key", "bearer",
    ),
    "xiaomi": ("Xiaomi", "XIAOMI_API_KEY", "Xiaomi API key", "api_key"),
    "xiaomi-token-plan-ams": (
        "Xiaomi Token Plan AMS", "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
        "Xiaomi Token Plan AMS API key", "api_key",
    ),
    "xiaomi-token-plan-cn": (
        "Xiaomi Token Plan CN", "XIAOMI_TOKEN_PLAN_CN_API_KEY",
        "Xiaomi Token Plan CN API key", "api_key",
    ),
    "xiaomi-token-plan-sgp": (
        "Xiaomi Token Plan SGP", "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
        "Xiaomi Token Plan SGP API key", "api_key",
    ),
    "zai-coding-cn": (
        "Z.AI Coding CN", "ZAI_CODING_CN_API_KEY", "Z.AI Coding CN API key", "api_key",
    ),
}


for _provider_id, (
    _provider_name,
    _provider_env,
    _provider_key_name,
    _provider_auth_mode,
) in _ADDITIONAL_PROVIDER_SPECS.items():
    PROVIDERS.setdefault(
        _provider_id,
        Provider(
            api="",
            key_env=_provider_env,
            base_url=None,
            model="",
            small_model="",
            name=_provider_name,
            api_key_name=_provider_key_name,
            auth_mode=_provider_auth_mode,
        ),
    )


def _load_pi_model_metadata() -> dict[str, dict[str, dict[str, Any]]]:
    """Load the vendored snapshot generated from Pi's provider catalog."""
    path = files("lsm_harness.ai").joinpath("data/pi_models.json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _normalize_pi_model(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate Pi's generated JSON field names to ``Model`` metadata."""
    compat = raw.get("compat") if isinstance(raw.get("compat"), dict) else {}
    cost = raw.get("cost") if isinstance(raw.get("cost"), dict) else {}
    source_api = str(raw.get("api") or "")
    metadata: dict[str, Any] = {
        "name": str(raw.get("name") or raw.get("id") or ""),
        "source_api": source_api,
        "base_url": raw.get("baseUrl"),
        "reasoning": bool(raw.get("reasoning", False)),
        "input_modalities": tuple(raw.get("input") or ("text",)),
        "context_window": int(raw.get("contextWindow") or 128_000),
        "max_tokens": int(raw.get("maxTokens") or 16_384),
        "cost": (
            float(cost.get("input", 0.0)),
            float(cost.get("output", 0.0)),
            float(cost.get("cacheRead", 0.0)),
            float(cost.get("cacheWrite", 0.0)),
        ),
        "thinking_level_map": dict(raw.get("thinkingLevelMap") or {}),
        "thinking_format": compat.get("thinkingFormat", "none"),
        "allow_empty_thinking_signature": bool(
            compat.get("allowEmptySignature", False)
        ),
        "force_adaptive_thinking": bool(
            compat.get("forceAdaptiveThinking", False)
        ),
        "headers": dict(raw.get("headers") or {}),
    }
    return metadata


def _install_pi_model_catalog() -> None:
    """Replace hand-written samples with Pi's complete generated lists."""
    catalog = _load_pi_model_metadata()
    for provider_id, raw_models in catalog.items():
        provider = PROVIDERS.get(provider_id)
        if provider is None or not isinstance(raw_models, dict):
            continue
        normalized = {
            model_id: _normalize_pi_model(raw)
            for model_id, raw in raw_models.items()
            if isinstance(model_id, str) and isinstance(raw, dict)
        }
        if not normalized:
            continue
        first_metadata = next(iter(normalized.values()))
        source_api = str(first_metadata.get("source_api") or provider.api)
        base_url = first_metadata.get("base_url") or provider.base_url
        first_model = next(iter(normalized))
        PROVIDERS[provider_id] = Provider(
            **{
                **provider.__dict__,
                "api": provider.api or source_api,
                "base_url": base_url,
                "model": provider.model or first_model,
                "small_model": provider.small_model or first_model,
                "models": tuple(normalized),
                "model_metadata": normalized,
            }
        )

    # Pi's Radius provider is dynamic and intentionally has no generated
    # model catalog. It still appears in /login as a built-in provider.
    PROVIDERS.setdefault(
        "radius",
        Provider(
            api="openai-completions",
            key_env="RADIUS_API_KEY",
            base_url=None,
            model="",
            small_model="",
            name="Radius",
            api_key_name="Radius API key",
            auth_mode="bearer",
        ),
    )


_install_pi_model_catalog()


def get_model(
    provider_name: str,
    model_id: str = "",
    *,
    small: bool = False,
    base_url: str | None = None,
    catalog: dict[str, Provider] | None = None,
) -> Model:
    """Resolve service metadata into the canonical model contract."""
    provider_name = canonical_provider_name(provider_name)
    provider = _get_provider(provider_name, catalog=catalog)
    resolved_id = model_id or (
        provider.small_model if small else provider.model
    )
    metadata = provider.model_metadata.get(resolved_id, {})
    thinking_level_map = metadata.get(
        "thinking_level_map",
        {} if metadata.get("reasoning") else {"off": None},
    )
    thinking_format = metadata.get("thinking_format", "none")
    cache_control_format = "none"
    supports_long_cache_retention = False
    if provider_name == "deepseek":
        thinking_level_map = {
            "off": "disabled",
            "minimal": "enabled",
            "low": "enabled",
            "medium": "enabled",
            "high": "enabled",
            "xhigh": "enabled",
        }
        thinking_format = "deepseek"
    elif provider_name in {"anthropic", "kimi-coding"}:
        if provider_name == "anthropic":
            thinking_level_map = {
                "off": None,
                "minimal": "1024",
                "low": "2048",
                "medium": "8192",
                "high": "16384",
                "xhigh": "32768",
            }
        elif "thinking_level_map" not in metadata:
            thinking_level_map = {}
        thinking_format = "anthropic"
        if provider_name == "anthropic":
            cache_control_format = "anthropic"
            supports_long_cache_retention = True
    elif provider_name == "openai" and resolved_id.startswith(
        ("o1", "o3", "o4", "gpt-5")
    ):
        thinking_level_map = {
            "off": None,
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
        }
        thinking_format = "reasoning_effort"
    costs = metadata.get("cost", (
        provider.input_cost_per_million,
        provider.output_cost_per_million,
        provider.cache_read_cost_per_million,
        provider.cache_write_cost_per_million,
    ))
    return Model(
        id=resolved_id,
        # Preserve the exact Pi API dialect when this Python port implements
        # it. OpenAI Responses and Google native models currently use the
        # product's compatible endpoints while retaining ``source_api`` in
        # metadata for the next translator migration.
        api=(
            metadata.get("source_api")
            if metadata.get("source_api") in {
                "openai-completions", "anthropic-messages"
            }
            else provider.api
        ),
        provider=provider_name,
        name=metadata.get("name", resolved_id),
        base_url=base_url or metadata.get("base_url") or provider.base_url,
        reasoning=metadata.get("reasoning", any(
            value is not None
            for level, value in thinking_level_map.items()
            if level != "off"
        )),
        input_modalities=metadata.get(
            "input_modalities", ("text", "image")
        ),
        context_window=metadata.get(
            "context_window",
            provider.model_context_windows.get(resolved_id, provider.context_window),
        ),
        max_tokens=metadata.get(
            "max_tokens",
            provider.model_max_output_tokens.get(
                resolved_id, provider.max_output_tokens
            ),
        ),
        thinking_level_map=thinking_level_map,
        thinking_format=thinking_format,
        cache_control_format=cache_control_format,
        supports_long_cache_retention=supports_long_cache_retention,
        input_cost_per_million=costs[0],
        output_cost_per_million=costs[1],
        cache_read_cost_per_million=costs[2],
        cache_write_cost_per_million=costs[3],
        auth_mode=provider.auth_mode,
        headers={**provider.headers, **metadata.get("headers", {})},
        allow_empty_thinking_signature=metadata.get(
            "allow_empty_thinking_signature", False
        ),
        force_adaptive_thinking=metadata.get(
            "force_adaptive_thinking", False
        ),
    )


def _detect_provider() -> str:
    """Auto-detect provider from whichever API key is set."""
    for name, provider in PROVIDERS.items():
        if os.getenv(provider.key_env, "").strip():
            return name
    return "deepseek"  # fallback


def _get_provider(
    name: str,
    *,
    catalog: dict[str, Provider] | None = None,
) -> Provider:
    """Resolve provider by name or raise with a helpful message."""
    providers = catalog or PROVIDERS
    if name in providers:
        return providers[name]
    raise ProviderConfigurationError(
        f"Unknown LSM_PROVIDER '{name}'.\n"
        f"Pick one of: {', '.join(providers)}\n"
        f"Set LSM_PROVIDER=<name> and that provider's API key."
    )


def canonical_provider_name(name: str) -> str:
    """Resolve a short-lived legacy id to Pi's unambiguous provider id."""
    return _PROVIDER_ALIASES.get(name, name)


def resolve_api_key(
    provider_name: str,
    api_key: str = "",
    *,
    catalog: dict[str, Provider] | None = None,
) -> str:
    """Resolve explicit, generic-env, then provider-env credentials."""
    provider_name = canonical_provider_name(provider_name)
    provider = _get_provider(provider_name, catalog=catalog)
    key = (
        api_key
        or os.getenv("LSM_API_KEY", "")
        or os.getenv("WAKU_API_KEY", "")
        or os.getenv(provider.key_env, "")
    ).strip()
    return key


def provider_has_credentials(
    provider_name: str,
    api_key: str = "",
    *,
    catalog: dict[str, Provider] | None = None,
) -> bool:
    """Return whether explicit or environment authentication is usable."""
    key = resolve_api_key(provider_name, api_key, catalog=catalog)
    return bool(key and key not in {"your-key-here", "replace-me"})


def provider_model_ids(
    provider_name: str,
    *,
    catalog: dict[str, Provider] | None = None,
) -> tuple[str, ...]:
    """Return selectable main models in catalog order."""
    provider = _get_provider(
        canonical_provider_name(provider_name), catalog=catalog
    )
    return provider.models or ((provider.model,) if provider.model else ())


def available_models(
    *,
    providers: tuple[str, ...] | None = None,
    catalog: dict[str, Provider] | None = None,
) -> list[Model]:
    """Return concrete provider/model entries for a provider scope."""
    provider_catalog = catalog or PROVIDERS
    names = providers if providers is not None else tuple(provider_catalog)
    return [
        get_model(
            canonical_provider_name(name), model_id, catalog=provider_catalog
        )
        for name in names
        for model_id in provider_model_ids(name, catalog=provider_catalog)
    ]


MODEL_CATALOG: dict[str, Model] = {
    model.id: model
    for model in available_models()
}


register_api_provider(ApiProvider("openai-completions", stream_openai_compat))
register_api_provider(ApiProvider("anthropic-messages", stream_anthropic_messages))


def get_client(
    provider_name: str = "",
    api_key: str = "",
    base_url: str | None = None,
    model: str = "",
    small_model: str = "",
    thinking: str = "disabled",
    timeout: float = 120.0,
    catalog: dict[str, Provider] | None = None,
) -> Any:
    """Build a ModelClient for the given provider.

    Returns an object with .complete(...) and .stream_complete(...).
    """
    resolved_provider_name = canonical_provider_name(
        provider_name or _detect_provider()
    )
    provider = _get_provider(resolved_provider_name, catalog=catalog)

    # Resolve API key: explicit > env var
    key = resolve_api_key(
        resolved_provider_name, api_key, catalog=catalog
    )
    if not provider_has_credentials(
        resolved_provider_name, key, catalog=catalog
    ):
        raise ProviderAuthError(
            f"No API key for provider '{resolved_provider_name}'.\n"
            f"  1. Get a key: {provider.key_url}\n"
            f"  2. Set {provider.key_env}=your-key in .env\n"
        )
    try:
        key.encode("latin-1")
    except UnicodeEncodeError:
        raise ProviderAuthError(
            f"{provider.key_env} contains non-ASCII characters. "
            f"Re-paste the key with no spaces or line breaks."
        )

    endpoint = base_url or provider.base_url
    if provider.api == "anthropic-messages":
        client = AnthropicClient(
            key,
            endpoint,
            timeout,
            provider_name=resolved_provider_name,
            auth_mode=provider.auth_mode,
            headers=provider.headers,
        )
    else:
        provider_thinking = thinking if resolved_provider_name == "deepseek" else None
        client = OpenAICompatClient(key, endpoint, provider_thinking, timeout)
    # Attach resolved model names so Harness can fill Settings defaults
    client._resolved_model = model or provider.model  # type: ignore[attr-defined]
    client._resolved_small_model = small_model or provider.small_model  # type: ignore[attr-defined]
    client.model = get_model(
        resolved_provider_name,
        model or provider.model,
        base_url=endpoint,
        catalog=catalog,
    )
    client.small_model = get_model(
        resolved_provider_name,
        small_model or provider.small_model,
        small=True,
        base_url=endpoint,
        catalog=catalog,
    )
    return client


# ── OpenAI-compatible client ──────────────────────────────────────


class OpenAICompatClient:
    """Speaks the ModelClient protocol over chat.completions."""

    def __init__(self, api_key: str, base_url: str | None, thinking: str | None = None, timeout: float = 120.0):
        from openai import OpenAI
        kwargs: dict = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": 0,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._thinking: ContextVar[str | None] = ContextVar(
            "lsm_thinking", default=thinking
        )

    @contextmanager
    def thinking_context(self, mode: str):
        """Apply one turn's DeepSeek thinking mode without cross-thread leakage."""
        if self._thinking.get() is None:
            yield
            return
        token = self._thinking.set(mode)
        try:
            yield
        finally:
            self._thinking.reset(token)

    def complete(self, *, model, system, messages, tools, max_tokens) -> ModelResponse:
        kwargs = self._build_kwargs(model, system, messages, tools, max_tokens)
        response = self._client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    def stream_complete(self, *, model, system, messages, tools, max_tokens) -> Iterator[StreamDelta]:
        kwargs = self._build_kwargs(model, system, messages, tools, max_tokens)
        kwargs["stream"] = True
        stream = self._client.chat.completions.create(**kwargs)

        pending: dict[int, dict[str, Any]] = {}
        seen: set[int] = set()
        final_stop = "stop"
        final_usage: Usage | None = None

        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = choice.delta

            if delta.content:
                yield StreamDelta(kind="text_delta", text=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in seen:
                        seen.add(idx)
                        pending[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                        yield StreamDelta(
                            kind="tool_call_start", tool_index=idx,
                            tool_id=tc.id or "", tool_name="",
                        )
                    if tc.function:
                        if tc.function.name:
                            pending[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            pending[idx]["arguments"] += tc.function.arguments
                            yield StreamDelta(
                                kind="tool_call_delta", tool_index=idx,
                                tool_id=pending[idx]["id"],
                                tool_name=pending[idx]["name"],
                                arguments_delta=tc.function.arguments,
                            )
            if choice.finish_reason:
                final_stop = choice.finish_reason
            if chunk.usage:
                final_usage = Usage(
                    input_tokens=getattr(chunk.usage, "prompt_tokens", 0),
                    output_tokens=getattr(chunk.usage, "completion_tokens", 0),
                )
        yield StreamDelta(
            kind="done",
            stop_reason=normalize_stop_reason(final_stop),
            usage=final_usage,
        )

    def stream(self, model: Model, context: AIContext, options: StreamOptions):
        return stream_openai_client(self._client, model, context, options)

    def _build_kwargs(self, model, system, messages, tools, max_tokens) -> dict:
        # Normalize messages: convert content arrays to wire format
        wire = ([{"role": "system", "content": system}] if system else [])
        for msg in messages:
            wire.append(_normalize_message(msg))
        kwargs: dict = {
            "model": model, "messages": wire,
            "max_tokens": max_tokens,
        }
        thinking = self._thinking.get()
        if thinking is not None:
            kwargs["extra_body"] = {"thinking": {"type": thinking}}
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {
                    "name": t["name"], "description": t["description"],
                    "parameters": t["input_schema"],
                }}
                for t in tools
            ]
        return kwargs

    def _parse_response(self, response) -> ModelResponse:
        choice = response.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for raw in msg.tool_calls or []:
            try:
                args = json.loads(raw.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must be an object")
            except (json.JSONDecodeError, ValueError) as exc:
                args = {"__parse_error__": str(exc), "__raw__": raw.function.arguments}
            calls.append(ToolCall(raw.id, raw.function.name, args))
        usage = response.usage
        return ModelResponse(
            text=msg.content or "", tool_calls=calls,
            stop_reason=normalize_stop_reason(choice.finish_reason or "stop"),
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            ),
        )


# ── Anthropic-native client ───────────────────────────────────────


class AnthropicClient:
    """Speaks the ModelClient protocol over Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None,
        timeout: float = 120.0,
        *,
        provider_name: str = "anthropic",
        auth_mode: Literal["api_key", "bearer"] = "api_key",
        headers: dict[str, str] | None = None,
    ):
        import anthropic
        kwargs: dict = {
            "timeout": timeout,
            "max_retries": 0,
        }
        kwargs["auth_token" if auth_mode == "bearer" else "api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if headers:
            kwargs["default_headers"] = dict(headers)
        self._client = anthropic.Anthropic(**kwargs)
        self._provider_name = provider_name

    def complete(self, *, model, system, messages, tools, max_tokens) -> ModelResponse:
        kwargs = self._build_kwargs(model, system, messages, tools, max_tokens)
        response = self._client.messages.create(**kwargs)
        return self._parse_response(response)

    def stream_complete(self, *, model, system, messages, tools, max_tokens) -> Iterator[StreamDelta]:
        resolved = Model(
            id=model,
            api="anthropic-messages",
            provider=self._provider_name,
        )
        context = AIContext(system, messages, tools)
        options = StreamOptions(max_tokens=max_tokens)
        for event in stream_anthropic_client(
            self._client,
            resolved,
            context,
            options,
        ):
            if event.kind == "text_delta":
                yield StreamDelta(kind="text_delta", text=event.text_delta)
            elif event.kind == "toolcall_start":
                yield StreamDelta(
                    kind="tool_call_start",
                    tool_index=event.tool_index,
                    tool_id=event.tool_id,
                    tool_name=event.tool_name,
                )
            elif event.kind == "toolcall_delta":
                yield StreamDelta(
                    kind="tool_call_delta",
                    tool_index=event.tool_index,
                    tool_id=event.tool_id,
                    tool_name=event.tool_name,
                    arguments_delta=event.arguments_delta,
                )
            elif event.kind in {"done", "error"}:
                yield StreamDelta(
                    kind="done",
                    stop_reason=event.partial.stop_reason,
                    usage=event.partial.usage,
                )

    def stream(self, model: Model, context: AIContext, options: StreamOptions):
        return stream_anthropic_client(self._client, model, context, options)

    def _build_kwargs(self, model, system, messages, tools, max_tokens) -> dict:
        return build_anthropic_request(
            Model(
                id=model,
                api="anthropic-messages",
                provider=self._provider_name,
            ),
            AIContext(system, messages, tools),
            StreamOptions(max_tokens=max_tokens),
        )

    def _parse_response(self, response) -> ModelResponse:
        text = ""
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, block.input))
        return ModelResponse(
            text=text, tool_calls=calls,
            stop_reason=normalize_stop_reason(response.stop_reason or "end_turn"),
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )


def _normalize_message(msg: dict) -> dict:
    """Ensure message content is in the right format for the API.

    Content arrays (for multimodal messages) pass through as-is.
    String content also passes through as-is.
    """
    return msg
