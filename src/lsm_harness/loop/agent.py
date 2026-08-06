"""Provider-neutral reason → act → observe loop."""

from __future__ import annotations

import json
from typing import Any, Callable

from lsm_harness.tools.registry import ToolRegistry
from lsm_harness.types import ModelClient, TurnResult


Emit = Callable[[str, dict], None]


def run_loop(
    *,
    client: ModelClient,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    max_iterations: int,
    max_tokens: int,
    emit: Emit,
) -> TurnResult:
    result = TurnResult(reply="")
    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        response = client.complete(
            model=model,
            system=system,
            messages=messages,
            tools=tools.schemas(),
            max_tokens=max_tokens,
        )
        emit(
            "llm.completed",
            {
                "role": "main",
                "model": model,
                "iteration": iteration,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            },
        )
        assistant: dict[str, Any] = {"role": "assistant", "content": response.text or None}
        if response.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in response.tool_calls
            ]
        messages.append(assistant)

        if not response.tool_calls:
            result.reply = response.text
            return result

        for call in response.tool_calls:
            emit("tool.requested", {"tool": call.name, "args": call.arguments})
            output = tools.execute(call.name, call.arguments)
            status = "error" if output.lower().startswith("error") else "ok"
            record = {"tool": call.name, "args": call.arguments, "output": output}
            result.tool_calls.append(record)
            emit("tool.completed", {**record, "status": status})
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": output}
            )

    result.reply = "达到最大迭代次数，任务尚未完成。请缩小请求范围后重试。"
    emit("loop.limit_reached", {"max_iterations": max_iterations})
    return result

