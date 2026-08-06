"""DeepSeek adapter for the provider-neutral harness model contract."""

from __future__ import annotations

import json
from typing import Any

from lsm_harness.types import ModelResponse, ToolCall, Usage


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, thinking: str = "disabled"):
        if not api_key:
            raise ValueError(
                "缺少 DeepSeek API Key。复制 .env.example 为 .env，填写 DEEPSEEK_API_KEY。"
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
        self._thinking = thinking

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ModelResponse:
        wire_messages = ([{"role": "system", "content": system}] if system else []) + messages
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": wire_messages,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": self._thinking}},
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in tools
            ]
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        for raw in message.tool_calls or []:
            try:
                arguments = json.loads(raw.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be an object")
            except (json.JSONDecodeError, ValueError) as exc:
                arguments = {"__parse_error__": str(exc), "__raw__": raw.function.arguments}
            calls.append(ToolCall(raw.id, raw.function.name, arguments))

        usage = response.usage
        return ModelResponse(
            text=message.content or "",
            tool_calls=calls,
            stop_reason=choice.finish_reason or "stop",
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            ),
        )

