"""LLM retrieval gate: skip irrelevant memory, fail open on uncertainty."""

from __future__ import annotations

import json

from lsm_harness.types import ModelClient


PROMPT = """你是个人 Agent 的长期记忆检索门。
判断回答用户消息是否需要读取用户的事实、偏好、人物、项目或过往经历。
只返回 JSON：
{{"retrieve": true/false, "query": "检索关键词或空字符串", "reason": "简短理由"}}
常识、数学、寒暄和自包含问题返回 false；涉及用户生活、项目、计划或历史返回 true。

用户消息：{message}"""


def _json_object(text: str) -> dict:
    return json.loads(text[text.index("{") : text.rindex("}") + 1])


def should_retrieve(
    client: ModelClient, model: str, message: str, emit=None
) -> tuple[bool, str, str]:
    try:
        response = client.complete(
            model=model,
            system="",
            messages=[{"role": "user", "content": PROMPT.format(message=message)}],
            tools=[],
            max_tokens=600,
        )
        if emit:
            emit(
                "llm.completed",
                {
                    "role": "retrieval_gate",
                    "model": model,
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                },
            )
        decision = _json_object(response.text)
        retrieve = bool(decision.get("retrieve"))
        query = str(decision.get("query") or (message if retrieve else ""))
        return retrieve, query, str(decision.get("reason") or "")
    except Exception as exc:
        if emit:
            emit("llm.failed", {"role": "retrieval_gate", "model": model, "error": type(exc).__name__})
        return True, message, f"gate failed open ({type(exc).__name__})"
