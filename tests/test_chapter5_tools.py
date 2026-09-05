"""Chapter 5 contracts: tool layers, pipeline, scheduling, and Operations."""

from __future__ import annotations

import threading
import time

from lsm_harness.agent.tools import (
    AgentTool,
    ExecutionContext,
    ToolRegistry,
    ToolResult,
    ToolResultMessage,
)
from lsm_harness.ai.api.anthropic_messages import translate_anthropic_messages
from lsm_harness.ai.api.openai_compat import build_openai_request
from lsm_harness.ai.types import (
    AIContext,
    AssistantMessageEvent,
    Model,
    ModelResponse,
    StreamOptions,
    Tool as AITool,
    ToolCall,
)
from lsm_harness.coding_agent.app import Harness
from lsm_harness.coding_agent.operations import (
    MockFileOperations,
    MockShellOperations,
    ShellResult,
)
from lsm_harness.coding_agent.tools import ToolDefinition, wrap_tool_definition
from lsm_harness.tools.filesystem import make_tools as make_file_tools
from lsm_harness.tools.shell import make_tool as make_shell_tool

from helpers import Tool, run_test_loop


EMPTY_SCHEMA = {"type": "object", "properties": {}}


def test_three_tool_layers_keep_product_metadata_out_of_ai_descriptor():
    product_context = {"prefix": "ctx"}

    def execute(value: str, _product_context=None) -> str:
        return f"{_product_context['prefix']}:{value}"

    definition = ToolDefinition(
        name="echo",
        label="Echo value",
        description="echo",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        execute=execute,
        product_context=product_context,
        prompt_snippet="Use echo for exact repetition.",
        execution_mode="parallel",
    )
    agent_tool = wrap_tool_definition(definition)
    descriptor = agent_tool.descriptor()

    assert isinstance(agent_tool, AgentTool)
    assert agent_tool.label == "Echo value"
    assert agent_tool.execute(value="x") == "ctx:x"
    assert descriptor == AITool("echo", "echo", definition.parameters)
    assert not hasattr(descriptor, "execute")
    assert not hasattr(descriptor, "prompt_snippet")


def test_product_prompt_snippets_are_assembled_only_by_harness():
    harness = object.__new__(Harness)
    harness.tool_prompt_snippets = ["Read in chunks.", "Write inside workspace."]

    prompt = harness._with_tool_prompt_snippets("base")

    assert prompt.startswith("base")
    assert "工具使用指南" in prompt
    assert "Read in chunks." in prompt


def test_prepare_runs_before_draft_2020_12_validation():
    registry = ToolRegistry()
    registry.register(AgentTool(
        name="nested",
        description="nested validation",
        parameters={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["safe"]},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["mode", "items"],
            "additionalProperties": False,
        },
        prepare_arguments=lambda args: {
            **args,
            "items": [{"count": int(args["items"][0]["count"])}],
        },
        execute=lambda mode, items: f"{mode}:{items[0]['count']}",
    ))

    assert registry.execute(
        "nested", {"mode": "safe", "items": [{"count": "2"}]}
    ).output == "safe:2"
    invalid = registry.execute(
        "nested", {"mode": "unsafe", "items": [{"count": "2"}]}
    )
    assert invalid.is_error
    assert "invalid arguments for 'nested'" in invalid.output
    assert "$.mode" in invalid.output
    extra = registry.execute(
        "nested", {"mode": "safe", "items": [{"count": "2", "extra": 1}]}
    )
    # prepare_arguments intentionally normalizes the nested object, proving
    # validation happens on prepared rather than raw input.
    assert not extra.is_error


def test_json_schema_paths_cover_nested_arrays_and_default_extra_fields():
    registry = ToolRegistry()
    registry.register(Tool(
        "check",
        "check",
        {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                    },
                },
            },
            "required": ["rows"],
        },
        lambda rows, **kwargs: str((rows, kwargs)),
    ))

    wrong = registry.execute("check", {"rows": [{"count": "bad"}]})
    assert wrong.is_error
    assert "$.rows[0].count" in wrong.output
    assert "is not of type 'integer'" in wrong.output
    # additionalProperties defaults to allowed when omitted by the schema.
    assert not registry.execute("check", {"rows": [{"count": 1}], "extra": 2}).is_error

    strict = ToolRegistry()
    strict.register(Tool(
        "strict",
        "strict",
        {"type": "object", "properties": {}, "additionalProperties": False},
        lambda: "ok",
    ))
    strict_result = strict.execute("strict", {"extra": 2})
    assert strict_result.is_error
    assert "Additional properties are not allowed" in strict_result.output


def test_every_pipeline_failure_becomes_an_error_message():
    def raises(message: str):
        def fail(*_args, **_kwargs):
            raise RuntimeError(message)
        return fail

    scenarios: list[tuple[str, ToolRegistry, dict, dict]] = []

    scenarios.append(("unknown", ToolRegistry(), {}, {}))

    prepare_registry = ToolRegistry()
    prepare_registry.register(Tool(
        "prepare", "", EMPTY_SCHEMA, lambda: "never", prepare_args=raises("prepare")
    ))
    scenarios.append(("prepare", prepare_registry, {}, {}))

    validate_registry = ToolRegistry()
    validate_registry.register(Tool(
        "validate",
        "",
        {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        lambda n: str(n),
    ))
    scenarios.append(("validate", validate_registry, {"n": "bad"}, {}))

    before_registry = ToolRegistry()
    before_registry.register(Tool("before", "", EMPTY_SCHEMA, lambda: "never"))
    scenarios.append((
        "before",
        before_registry,
        {},
        {"before_tool_call": raises("before")},
    ))

    execute_registry = ToolRegistry()
    execute_registry.register(Tool("execute", "", EMPTY_SCHEMA, raises("execute")))
    scenarios.append(("execute", execute_registry, {}, {}))

    after_registry = ToolRegistry()
    after_registry.register(Tool(
        "after", "", EMPTY_SCHEMA, lambda: "ok", after_hook=raises("after")
    ))
    scenarios.append(("after", after_registry, {}, {}))

    config_after_registry = ToolRegistry()
    config_after_registry.register(Tool("config_after", "", EMPTY_SCHEMA, lambda: "ok"))
    scenarios.append((
        "config_after",
        config_after_registry,
        {},
        {"after_tool_call": raises("config after")},
    ))

    for name, registry, arguments, options in scenarios:
        result = registry.execute(name, arguments, **options)
        assert result.is_error, name
        assert result.tool_name == name


def test_approval_exception_is_an_error_message():
    class BrokenApproval:
        def request(self, **_kwargs):
            raise RuntimeError("approval offline")

    registry = ToolRegistry()
    registry.register(Tool(
        "write", "", EMPTY_SCHEMA, lambda: "never", effect="local_write"
    ))

    result = registry.execute(
        "write", {}, ExecutionContext(approval_broker=BrokenApproval())
    )

    assert result.is_error
    assert "approval" in result.output
    assert "approval offline" in result.output


def test_progress_has_identity_and_late_updates_are_dropped():
    release = threading.Event()
    background_threads: list[threading.Thread] = []
    events: list[tuple[str, dict]] = []

    def execute(_on_update=None):
        _on_update("now")

        def late_update():
            release.wait()
            _on_update("late")

        worker = threading.Thread(target=late_update)
        worker.start()
        background_threads.append(worker)
        return "done"

    registry = ToolRegistry()
    registry.register(AgentTool(
        name="progress",
        label="Progress tool",
        description="",
        parameters=EMPTY_SCHEMA,
        execute=execute,
    ))
    results = registry.execute_batch(
        [("call-7", "progress", {})],
        ctx=ExecutionContext(emit=lambda kind, data: events.append((kind, data))),
    )
    release.set()
    for worker in background_threads:
        worker.join(timeout=1)

    progress = [data for kind, data in events if kind == "tool.progress"]
    assert [event["delta"] for event in progress] == ["now"]
    assert progress[0]["tool"] == "progress"
    assert progress[0]["label"] == "Progress tool"
    assert progress[0]["tool_call_id"] == "call-7"
    assert results[0][2].tool_call_id == "call-7"


def test_parallel_batch_has_sequential_preflight_completion_events_and_ordered_results():
    events: list[tuple[str, dict]] = []
    activity: list[str] = []
    both_started = threading.Event()
    started: set[str] = set()
    lock = threading.Lock()

    def prepare(name: str):
        def inner(arguments):
            activity.append(f"prepare:{name}")
            return arguments
        return inner

    def execute(name: str, delay: float):
        def inner():
            with lock:
                started.add(name)
                if len(started) == 2:
                    both_started.set()
            assert both_started.wait(timeout=1)
            time.sleep(delay)
            return name
        return inner

    registry = ToolRegistry()
    registry.register(AgentTool(
        "slow", "", EMPTY_SCHEMA, execute("slow", 0.04),
        prepare_arguments=prepare("slow"), execution_mode="parallel",
    ))
    registry.register(AgentTool(
        "fast", "", EMPTY_SCHEMA, execute("fast", 0.0),
        prepare_arguments=prepare("fast"), execution_mode="parallel",
    ))

    results = registry.execute_batch(
        [("1", "slow", {}), ("2", "fast", {})],
        ctx=ExecutionContext(emit=lambda kind, data: events.append((kind, data))),
    )

    assert activity == ["prepare:slow", "prepare:fast"]
    execution_end = [
        data["tool_call_id"] for kind, data in events if kind == "tool.execution_end"
    ]
    assert execution_end == ["2", "1"]
    assert [result.output for _, _, result in results] == ["slow", "fast"]


def test_one_sequential_tool_and_config_mode_each_force_the_whole_batch_sequential():
    def run(modes: tuple[str, str], batch_mode: str) -> int:
        active = 0
        maximum = 0
        lock = threading.Lock()

        def execute() -> str:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return "ok"

        registry = ToolRegistry()
        for index, mode in enumerate(modes):
            registry.register(AgentTool(
                f"tool-{index}", "", EMPTY_SCHEMA, execute,
                execution_mode=mode,
            ))
        registry.execute_batch(
            [("1", "tool-0", {}), ("2", "tool-1", {})],
            mode=batch_mode,
        )
        return maximum

    assert run(("parallel", "sequential"), "parallel") == 1
    assert run(("parallel", "parallel"), "sequential") == 1


def test_provider_boundary_maps_errors_and_strips_internal_fields():
    # Agent extras (details / terminate) are stripped at the Agent→AI
    # boundary; providers only ever see the strict standard messages.
    from lsm_harness.agent.messages import (
        AgentToolResultMessage,
        default_convert_to_llm,
    )

    agent_message = AgentToolResultMessage(
        tool_call_id="call-1",
        tool_name="echo",
        content="bad",
        is_error=True,
        details={"secret": "trace-only"},
        terminate=False,
    )
    messages = default_convert_to_llm([agent_message])
    context = AIContext(
        "system",
        messages,
        [AITool("echo", "echo", EMPTY_SCHEMA)],
    )
    request = build_openai_request(
        Model("m", "openai-completions", "test"),
        context,
        StreamOptions(max_tokens=100),
    )
    translated = translate_anthropic_messages(messages)

    assert request["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "bad",
    }
    assert translated[0]["content"][0]["is_error"] is True


def test_mock_file_and_shell_operations_do_not_require_real_system(tmp_path):
    file_ops = MockFileOperations({"src/app.py": "one\ntwo"})
    registry = ToolRegistry(allowed_effects={"read", "local_write", "external_write"})
    for definition in make_file_tools(tmp_path, operations=file_ops):
        registry.register(wrap_tool_definition(definition))
    assert "two" in registry.execute("read_file", {"path": "src/app.py"}).output
    assert not registry.execute(
        "write_file", {"path": "src/new.py", "content": "new"}
    ).is_error
    assert file_ops.files["/workspace/src/new.py"] == "new"

    shell_ops = MockShellOperations(ShellResult(0, "mock-output", ""))
    registry.register(wrap_tool_definition(make_shell_tool(
        tmp_path,
        operations=shell_ops,
    )))
    shell_result = registry.execute("exec", {"command": "pwd", "cwd": "src"})
    assert shell_result.output == "mock-output"
    assert shell_ops.calls == [(["pwd"], "src", 60)]


def test_scripted_model_corrects_bad_tool_arguments_then_finishes():
    registry = ToolRegistry()
    registry.register(Tool(
        "double",
        "double",
        {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        lambda value: str(value * 2),
    ))
    calls = 0

    def stream_fn(_model, context, _options):
        nonlocal calls
        calls += 1
        if calls == 1:
            response = ModelResponse(
                tool_calls=[ToolCall("bad", "double", {"value": "2"})],
                stop_reason="tool_calls",
            )
        elif calls == 2:
            assert context.messages[-1].is_error is True
            response = ModelResponse(
                tool_calls=[ToolCall("good", "double", {"value": 2})],
                stop_reason="tool_calls",
            )
        else:
            assert context.messages[-1].content == "4"
            response = ModelResponse(text="fixed", stop_reason="stop")
        yield AssistantMessageEvent("done", response)

    result = run_test_loop(
        stream_fn=stream_fn,
        model=Model("scripted", "test", "test"),
        system="system",
        messages=[{"role": "user", "content": "double two"}],
        tools=registry,
        max_iterations=4,
        max_tokens=100,
        emit=lambda _kind, _data: None,
    )

    assert result.reply == "fixed"
    assert result.iterations == 3
    assert [item["is_error"] for item in result.tool_calls] == [True, False]


# ── batch F-2: per-tool independent abort (refactor plan §9.2) ────


def test_tool_timeout_does_not_pollute_batch_siblings():
    """One tool's timeout aborts only itself; the sibling finishes."""
    from lsm_harness.agent.tools import CombinedAbortHandle

    started = threading.Event()
    seen: dict[str, bool] = {}

    def slow(_abort=None):
        started.set()
        time.sleep(5)
        return "slow done"

    def fast(_abort=None):
        assert started.wait(timeout=2)
        # Wait until the sibling has definitely timed out.
        time.sleep(0.3)
        seen["fast_aborted"] = _abort.aborted
        return "fast done"

    registry = ToolRegistry()
    registry.register(AgentTool(
        "slow", "", EMPTY_SCHEMA, slow, timeout=0.1, execution_mode="parallel",
    ))
    registry.register(AgentTool(
        "fast", "", EMPTY_SCHEMA, fast, execution_mode="parallel",
    ))

    results = registry.execute_batch(
        [("c1", "slow", {}), ("c2", "fast", {})],
        ctx=ExecutionContext(),
    )

    by_id = {call_id: result for _i, call_id, result in results}
    assert by_id["c1"].is_error is True
    assert "timed out" in by_id["c1"].output
    assert by_id["c2"].is_error is False
    assert by_id["c2"].output == "fast done"
    # The sibling's handle never saw the timeout abort.
    assert seen["fast_aborted"] is False
    # Each call ran on its own CombinedAbortHandle.
    assert CombinedAbortHandle is not None


def test_global_interrupt_still_aborts_every_tool():
    """The parent handle (global user interrupt) aborts the whole batch."""
    interrupt = threading.Event()
    seen: dict[str, bool] = {}

    def make_tool(name):
        def run(_abort=None):
            interrupt.set()
            time.sleep(0.05)
            seen[name] = _abort.aborted
            return name
        return run

    registry = ToolRegistry()
    registry.register(AgentTool(
        "t1", "", EMPTY_SCHEMA, make_tool("t1"), execution_mode="parallel",
    ))
    registry.register(AgentTool(
        "t2", "", EMPTY_SCHEMA, make_tool("t2"), execution_mode="parallel",
    ))

    from lsm_harness.agent.tools import AbortHandle

    ctx = ExecutionContext(abort=AbortHandle(external=interrupt))
    results = registry.execute_batch(
        [("c1", "t1", {}), ("c2", "t2", {})],
        ctx=ctx,
    )

    assert all(seen.values())
    assert all(result.output for _i, _cid, result in results)


def test_combined_abort_handle_layers_local_and_parent():
    from lsm_harness.agent.tools import AbortHandle, CombinedAbortHandle

    parent = AbortHandle()
    child = CombinedAbortHandle(parent)
    assert not child.aborted

    child.abort()
    assert child.aborted
    assert not parent.aborted  # local abort does not climb

    other = CombinedAbortHandle(parent)
    parent.abort()
    assert other.aborted  # parent abort propagates down

