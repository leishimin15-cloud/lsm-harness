"""Pi parity for optional product-owned Agent Loop limits."""

from __future__ import annotations

from lsm_harness.agent.tools import ToolRegistry
from lsm_harness.agent.types import AgentLoopConfig
from lsm_harness.ai.types import ModelResponse, ToolCall
from lsm_harness.config import Settings

from helpers import QueueClient, Tool, run_test_loop


def _registry() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(Tool(
        "echo",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        fn=lambda value: f"ok:{value}",
    ))
    return tools


def test_agent_loop_has_no_default_turn_limit() -> None:
    assert AgentLoopConfig().max_iterations is None


def test_unbounded_loop_can_continue_past_legacy_ten_turn_cap() -> None:
    responses = [
        ModelResponse(
            tool_calls=[ToolCall(str(index), "echo", {"value": str(index)})]
        )
        for index in range(1, 12)
    ]
    responses.append(ModelResponse(text="finished"))
    client = QueueClient(*responses)

    result = run_test_loop(
        client=client,
        model="scripted",
        messages=[{"role": "user", "content": "long task"}],
        tools=_registry(),
        max_iterations=None,
    )

    assert result.status == "completed"
    assert result.reply == "finished"
    assert result.iterations == 12
    assert len(result.tool_calls) == 11


def test_settings_zero_or_missing_means_unlimited(monkeypatch) -> None:
    monkeypatch.delenv("LSM_MAX_ITERATIONS", raising=False)
    monkeypatch.delenv("WAKU_MAX_ITERATIONS", raising=False)
    assert Settings().max_iterations is None

    monkeypatch.setenv("LSM_MAX_ITERATIONS", "0")
    assert Settings().max_iterations is None


def test_settings_positive_value_keeps_product_owned_limit(monkeypatch) -> None:
    monkeypatch.setenv("LSM_MAX_ITERATIONS", "25")
    assert Settings().max_iterations == 25
