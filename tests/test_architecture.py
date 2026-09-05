"""Regression tests for the Pi-shaped three-layer package boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

from lsm_harness.agent import Agent, AgentContext, AgentLoopConfig
from lsm_harness.agent.messages import user_message
from lsm_harness.agent.agent_loop import run_agent_loop
from lsm_harness.agent.tools import ToolRegistry
from lsm_harness.agent.types import TraceResult
from lsm_harness.ai.types import (
    AssistantMessageEvent,
    Model,
    ModelResponse,
)
from lsm_harness.coding_agent.app import Harness
from lsm_harness.coding_agent.cli import _queue_running_input


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "lsm_harness"


def _internal_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return {name for name in imports if name.startswith("lsm_harness")}


def test_canonical_package_dependency_direction():
    for path in (SOURCE_ROOT / "ai").rglob("*.py"):
        imports = _internal_imports(path)
        assert not any(name.startswith("lsm_harness.agent") for name in imports)
        assert not any(name.startswith("lsm_harness.coding_agent") for name in imports)

    for path in (SOURCE_ROOT / "agent").glob("*.py"):
        imports = _internal_imports(path)
        assert not any(name.startswith("lsm_harness.coding_agent") for name in imports)


def test_agent_context_and_config_form_the_low_level_boundary():
    context = AgentContext("system", [], ToolRegistry())
    config = AgentLoopConfig("model", max_iterations=3, max_tokens=100)

    assert context.system_prompt == "system"
    assert config.model == "model"
    assert config.before_tool_call is None
    assert config.after_tool_call is None
    assert config.tool_execution == "parallel"
    assert callable(run_agent_loop)


def test_agent_loop_consumes_stream_function_and_recovers_overflow():
    calls = []

    def stream_fn(model, context, options):
        calls.append((model, context, options))
        yield AssistantMessageEvent("start", ModelResponse())
        if len(calls) == 1:
            yield AssistantMessageEvent(
                "error",
                ModelResponse(
                    stop_reason="error",
                    error_message="maximum context length exceeded",
                ),
                error_category="permanent",
            )
            return
        yield AssistantMessageEvent("text_start", ModelResponse())
        yield AssistantMessageEvent(
            "text_delta",
            ModelResponse(text="recovered"),
            text_delta="recovered",
        )
        yield AssistantMessageEvent("text_end", ModelResponse(text="recovered"))
        yield AssistantMessageEvent("done", ModelResponse(text="recovered"))

    model = Model(id="model", api="test", provider="test")
    context = AgentContext(
        "system",
        [user_message("hello")],
        ToolRegistry(),
    )
    config = AgentLoopConfig(
        model,
        max_iterations=3,
        max_tokens=100,
        thinking="high",
        cache_retention="long",
        on_truncation=lambda: (
            "compact system",
            [user_message("compact request")],
        ),
    )

    result = run_agent_loop(
        context=context,
        config=config,
        stream_fn=stream_fn,
        emit=lambda _event, _data: None,
    )

    assert result.reply == "recovered"
    assert result.turn_count == 2
    assert calls[0][0] is model
    assert calls[0][2].reasoning == "high"
    assert calls[0][2].cache_retention == "long"
    assert calls[1][1].system_prompt == "compact system"


def test_stateful_agent_owns_active_run_and_both_message_queues():
    agent = Agent()
    assert not agent.steer("idle")
    assert not agent.follow_up("idle")

    active = agent.begin()
    assert _queue_running_input(agent, "urgent", follow_up=False)
    assert _queue_running_input(agent, "later", follow_up=True)
    assert agent.has_queued_messages()
    assert agent.pending_messages() == {
        "steering": ["urgent"],
        "follow_up": ["later"],
    }

    agent.finish(active)
    assert not agent.is_running
    assert agent.clear_all_queues() == {
        "steering": ["urgent"],
        "follow_up": ["later"],
    }


def test_trace_result_remains_an_ai_layer_contract():
    result = TraceResult(reply="done")
    assert result.turn_count == 0
