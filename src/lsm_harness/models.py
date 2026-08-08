"""Multi-provider model access — one loop, zero framework lock-in.

The loop speaks one dialect: the ``ModelClient`` protocol (complete + stream_complete).
Providers plug in behind that interface.

Two wire formats:
  - **openai** – chat.completions (OpenAI, DeepSeek, Google Gemini, OpenRouter, xAI, ...)
  - **anthropic** – messages (Anthropic, Kimi/Moonshot, GLM/Z.ai, MiniMax)

Pick with LSM_PROVIDER=... and set that provider's API key.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator

from lsm_harness.types import ModelResponse, StreamDelta, ToolCall, Usage


# ── provider registry ─────────────────────────────────────────────


@dataclass(frozen=True)
class Provider:
    """Metadata for one LLM provider."""

    kind: str  # 'openai' or 'anthropic' — the wire format
    key_env: str  # env var for the API key
    base_url: str | None  # override endpoint; None = SDK default
    model: str  # default main model
    small_model: str  # default cheap model (gate + summary)
    # Where to get a key (shown in error messages)
    key_url: str = ""


PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider(
        "openai", "DEEPSEEK_API_KEY", "https://api.deepseek.com",
        "deepseek-v4-pro", "deepseek-v4-flash",
        key_url="https://platform.deepseek.com/api_keys",
    ),
    "openai": Provider(
        "openai", "OPENAI_API_KEY", None,
        "gpt-4o", "gpt-4o-mini",
        key_url="https://platform.openai.com/api-keys",
    ),
    "anthropic": Provider(
        "anthropic", "ANTHROPIC_API_KEY", None,
        "claude-sonnet-4-20250514", "claude-haiku-3-5-20241022",
        key_url="https://console.anthropic.com/settings/keys",
    ),
    "gemini": Provider(
        "openai", "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.5-flash", "gemini-2.5-flash",
        key_url="https://aistudio.google.com/apikey",
    ),
    "openrouter": Provider(
        "openai", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
        "anthropic/claude-sonnet-4", "google/gemini-2.5-flash",
        key_url="https://openrouter.ai/keys",
    ),
    "xai": Provider(
        "openai", "XAI_API_KEY", "https://api.x.ai/v1",
        "grok-4-latest", "grok-4-fast-latest",
        key_url="https://console.x.ai",
    ),
    "kimi": Provider(
        "anthropic", "MOONSHOT_API_KEY", "https://api.moonshot.ai/anthropic",
        "kimi-k3", "kimi-k2.6",
        key_url="https://platform.moonshot.ai/console/api-keys",
    ),
    "glm": Provider(
        "anthropic", "ZHIPU_API_KEY", "https://api.z.ai/api/anthropic",
        "glm-5.2", "glm-5-turbo",
        key_url="https://z.ai/manage-apikey/apikey-list",
    ),
    "minimax": Provider(
        "anthropic", "MINIMAX_API_KEY", "https://api.minimaxi.com/anthropic",
        "MiniMax-M3", "MiniMax-M2",
        key_url="https://platform.minimaxi.com/user-center/basic-information",
    ),
}


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
    provider = _get_provider(provider_name or _detect_provider())

    # Resolve API key: explicit > env var
    key = (api_key or os.getenv(provider.key_env, "")).strip()
    if not key or key in {"your-key-here", "replace-me"}:
        raise SystemExit(
            f"No API key for provider '{provider_name or _detect_provider()}'.\n"
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

    if provider.kind == "anthropic":
        client = AnthropicClient(key, endpoint, timeout)
    else:
        client = OpenAICompatClient(key, endpoint, thinking, timeout)
    # Attach resolved model names so Harness can fill Settings defaults
    client._resolved_model = model or provider.model  # type: ignore[attr-defined]
    client._resolved_small_model = small_model or provider.small_model  # type: ignore[attr-defined]
    return client


# ── OpenAI-compatible client ──────────────────────────────────────


class OpenAICompatClient:
    """Speaks the ModelClient protocol over chat.completions."""

    def __init__(self, api_key: str, base_url: str | None, thinking: str = "disabled", timeout: float = 120.0):
        from openai import OpenAI
        kwargs: dict = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._thinking = thinking

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
        yield StreamDelta(kind="done", stop_reason=final_stop, usage=final_usage)

    def _build_kwargs(self, model, system, messages, tools, max_tokens) -> dict:
        wire = ([{"role": "system", "content": system}] if system else []) + messages
        kwargs: dict = {
            "model": model, "messages": wire,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": self._thinking}},
        }
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
            stop_reason=choice.finish_reason or "stop",
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
        kwargs = self._build_kwargs(model, system, messages, tools, max_tokens)
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield StreamDelta(kind="text_delta", text=text)
            final = stream.get_final_message()
            usage = Usage(
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
            )
            yield StreamDelta(
                kind="done",
                stop_reason=final.stop_reason or "end_turn",
                usage=usage,
            )

    def _build_kwargs(self, model, system, messages, tools, max_tokens) -> dict:
        kwargs: dict = {
            "model": model, "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t["description"],
                 "input_schema": t["input_schema"]}
                for t in tools
            ]
        return kwargs

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
            stop_reason=response.stop_reason or "end_turn",
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )
