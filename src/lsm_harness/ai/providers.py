"""Service catalog plus the legacy ``ModelClient`` construction facade.

The Agent Loop consumes the canonical ``StreamFunction`` protocol. Existing
Memory, RAG, tests, and external callers can keep using ``get_client()`` while
they migrate from ``complete`` / ``stream_complete``.

Two wire formats:
  - **openai-completions** – OpenAI, DeepSeek, Gemini, OpenRouter, xAI
  - **anthropic-messages** – Anthropic, Kimi/Moonshot, GLM/Z.ai, MiniMax

Pick with LSM_PROVIDER=... and set that provider's API key.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

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


@dataclass(frozen=True)
class Provider:
    """Metadata for one LLM provider."""

    api: str  # API translator key
    key_env: str  # env var for the API key
    base_url: str | None  # override endpoint; None = SDK default
    model: str  # default main model
    small_model: str  # default cheap model (gate + summary)
    # Where to get a key (shown in error messages)
    key_url: str = ""
    # Model limits; resolve_max_tokens clamps against these. Conservative
    # defaults so unverified providers stay safe — verify against each
    # provider's docs before tightening.
    context_window: int = 128_000
    max_output_tokens: int = 8_192


PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider(
        "openai-completions", "DEEPSEEK_API_KEY", "https://api.deepseek.com",
        "deepseek-v4-pro", "deepseek-v4-flash",
        key_url="https://platform.deepseek.com/api_keys",
    ),
    "openai": Provider(
        "openai-completions", "OPENAI_API_KEY", None,
        "gpt-4o", "gpt-4o-mini",
        key_url="https://platform.openai.com/api-keys",
        context_window=128_000, max_output_tokens=16_384,
    ),
    "anthropic": Provider(
        "anthropic-messages", "ANTHROPIC_API_KEY", None,
        "claude-sonnet-4-20250514", "claude-haiku-3-5-20241022",
        key_url="https://console.anthropic.com/settings/keys",
        context_window=200_000, max_output_tokens=64_000,
    ),
    "gemini": Provider(
        "openai-completions", "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.5-flash", "gemini-2.5-flash",
        key_url="https://aistudio.google.com/apikey",
        context_window=1_000_000, max_output_tokens=65_536,
    ),
    "openrouter": Provider(
        "openai-completions", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
        "anthropic/claude-sonnet-4", "google/gemini-2.5-flash",
        key_url="https://openrouter.ai/keys",
    ),
    "xai": Provider(
        "openai-completions", "XAI_API_KEY", "https://api.x.ai/v1",
        "grok-4-latest", "grok-4-fast-latest",
        key_url="https://console.x.ai",
    ),
    "kimi": Provider(
        "anthropic-messages", "MOONSHOT_API_KEY", "https://api.moonshot.ai/anthropic",
        "kimi-k3", "kimi-k2.6",
        key_url="https://platform.moonshot.ai/console/api-keys",
        context_window=256_000, max_output_tokens=8_192,
    ),
    "glm": Provider(
        "anthropic-messages", "ZHIPU_API_KEY", "https://api.z.ai/api/anthropic",
        "glm-5.2", "glm-5-turbo",
        key_url="https://z.ai/manage-apikey/apikey-list",
    ),
    "minimax": Provider(
        "anthropic-messages", "MINIMAX_API_KEY", "https://api.minimaxi.com/anthropic",
        "MiniMax-M3", "MiniMax-M2",
        key_url="https://platform.minimaxi.com/user-center/basic-information",
    ),
}


def get_model(
    provider_name: str,
    model_id: str = "",
    *,
    small: bool = False,
    base_url: str | None = None,
) -> Model:
    """Resolve service metadata into the canonical model contract."""
    provider = _get_provider(provider_name)
    resolved_id = model_id or (
        provider.small_model if small else provider.model
    )
    thinking_level_map = {"off": None}
    thinking_format = "none"
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
    elif provider_name == "anthropic":
        thinking_level_map = {
            "off": None,
            "minimal": "1024",
            "low": "2048",
            "medium": "8192",
            "high": "16384",
            "xhigh": "32768",
        }
        thinking_format = "anthropic"
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
    return Model(
        id=resolved_id,
        api=provider.api,
        provider=provider_name,
        base_url=base_url or provider.base_url,
        context_window=provider.context_window,
        max_tokens=provider.max_output_tokens,
        thinking_level_map=thinking_level_map,
        thinking_format=thinking_format,
        cache_control_format=cache_control_format,
        supports_long_cache_retention=supports_long_cache_retention,
    )


def _detect_provider() -> str:
    """Auto-detect provider from whichever API key is set."""
    for name, provider in PROVIDERS.items():
        if os.getenv(provider.key_env, "").strip():
            return name
    return "deepseek"  # fallback


def _get_provider(name: str) -> Provider:
    """Resolve provider by name or raise with a helpful message."""
    if name in PROVIDERS:
        return PROVIDERS[name]
    raise SystemExit(
        f"Unknown LSM_PROVIDER '{name}'.\n"
        f"Pick one of: {', '.join(PROVIDERS)}\n"
        f"Set LSM_PROVIDER=<name> and that provider's API key."
    )


MODEL_CATALOG: dict[str, Model] = {
    model.id: model
    for name, provider in PROVIDERS.items()
    for model in (
        get_model(name, provider.model),
        get_model(name, provider.small_model, small=True),
    )
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
) -> Any:
    """Build a ModelClient for the given provider.

    Returns an object with .complete(...) and .stream_complete(...).
    """
    resolved_provider_name = provider_name or _detect_provider()
    provider = _get_provider(resolved_provider_name)

    # Resolve API key: explicit > env var
    key = (api_key or os.getenv(provider.key_env, "")).strip()
    if not key or key in {"your-key-here", "replace-me"}:
        raise SystemExit(
            f"No API key for provider '{resolved_provider_name}'.\n"
            f"  1. Get a key: {provider.key_url}\n"
            f"  2. Set {provider.key_env}=your-key in .env\n"
        )
    try:
        key.encode("latin-1")
    except UnicodeEncodeError:
        raise SystemExit(
            f"{provider.key_env} contains non-ASCII characters. "
            f"Re-paste the key with no spaces or line breaks."
        )

    endpoint = base_url or provider.base_url

    if provider.api == "anthropic-messages":
        client = AnthropicClient(key, endpoint, timeout)
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
    )
    client.small_model = get_model(
        resolved_provider_name,
        small_model or provider.small_model,
        small=True,
        base_url=endpoint,
    )
    return client


# ── OpenAI-compatible client ──────────────────────────────────────


class OpenAICompatClient:
    """Speaks the ModelClient protocol over chat.completions."""

    def __init__(self, api_key: str, base_url: str | None, thinking: str | None = None, timeout: float = 120.0):
        from openai import OpenAI
        kwargs: dict = {"api_key": api_key, "timeout": timeout}
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

    def __init__(self, api_key: str, base_url: str | None, timeout: float = 120.0):
        import anthropic
        kwargs: dict = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)

    def complete(self, *, model, system, messages, tools, max_tokens) -> ModelResponse:
        kwargs = self._build_kwargs(model, system, messages, tools, max_tokens)
        response = self._client.messages.create(**kwargs)
        return self._parse_response(response)

    def stream_complete(self, *, model, system, messages, tools, max_tokens) -> Iterator[StreamDelta]:
        resolved = Model(
            id=model,
            api="anthropic-messages",
            provider="anthropic",
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
                provider="anthropic",
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
