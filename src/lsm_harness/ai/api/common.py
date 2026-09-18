"""Shared helpers for provider event translators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from lsm_harness.ai.types import ModelResponse, StopReason, ToolCall, Usage


@dataclass
class PendingToolCall:
    id: str
    name: str = ""
    arguments: str = ""


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be an object")
        return parsed
    except (json.JSONDecodeError, ValueError) as exc:
        return {"__parse_error__": str(exc), "__raw__": raw}


def snapshot(
    *,
    text: str,
    thinking: str,
    pending: dict[int, PendingToolCall],
    stop_reason: StopReason = "stop",
    usage: Usage | None = None,
    error_message: str = "",
    thinking_signature: str = "",
) -> ModelResponse:
    return ModelResponse(
        text=text,
        thinking=thinking,
        thinking_signature=thinking_signature,
        tool_calls=[
            ToolCall(
                id=item.id,
                name=item.name,
                arguments=parse_tool_arguments(item.arguments),
            )
            for _, item in sorted(pending.items())
        ],
        stop_reason=stop_reason,
        usage=usage or Usage(),
        error_message=error_message,
    )


def is_aborted(interrupt: Any) -> bool:
    return bool(interrupt is not None and interrupt.is_set())


# ── 代理可达性兜底(2026-09-12 实机问题)─────────────────────────
# openai/anthropic SDK 的 httpx 默认 trust_env=True,macOS 下会拾起
# **系统代理**(urllib getproxies 读 scutil)——用户代理软件没开时
# 所有模型调用 Connection refused 全挂,而 curl(不走系统代理)正常。
# 策略:进程启动时探测一次代理端口,不可达则改用 trust_env=False 的
# 直连 client;代理可达则完全保持默认行为。

_PROXY_REACHABLE: bool | None = None


def proxy_reachable() -> bool:
    """配置的 HTTP(S) 代理是否可连(无代理配置视为 True = 用默认)。

    结果进程级缓存——每次模型调用不重复探测;代理后启动的进程
    需要重启才恢复走代理。
    """
    global _PROXY_REACHABLE
    if _PROXY_REACHABLE is not None:
        return _PROXY_REACHABLE
    import socket
    import urllib.request
    from urllib.parse import urlparse

    proxies = urllib.request.getproxies()
    url = proxies.get("https") or proxies.get("http") or ""
    if not url:
        _PROXY_REACHABLE = True
        return True
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if not parsed.hostname:
        _PROXY_REACHABLE = True
        return True
    try:
        with socket.create_connection(
            (parsed.hostname, parsed.port or 8080), timeout=1.0
        ):
            _PROXY_REACHABLE = True
    except OSError:
        _PROXY_REACHABLE = False
    return _PROXY_REACHABLE


def sdk_http_client() -> Any:
    """代理不可达时返回直连 httpx.Client(trust_env=False);可达或无
    代理配置时返回 None(SDK 用其默认 client,尊重代理设置)。"""
    if proxy_reachable():
        return None
    import httpx

    return httpx.Client(trust_env=False, follow_redirects=True)
