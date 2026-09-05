from lsm_harness.agent.hooks import LoopHooks, NextTurnUpdate
from lsm_harness.agent.pending import PendingMessageQueue
from lsm_harness.agent.messages import (
    custom_message,
    default_convert_to_llm,
    user_message,
)
from lsm_harness.agent.types import (
    AfterToolCallResult,
    BeforeToolCallResult,
)
from lsm_harness.agent.tools import AbortHandle, ExecutionContext, ToolRegistry, ToolResult
from lsm_harness.ai.types import (
    ModelResponse,
    StreamDelta,
    ToolCall,
    Usage,
    normalize_stop_reason,
)

from helpers import QueueClient, Tool, run_test_loop


def _drain_queue(sq):
    """Adapt a legacy steering queue.Queue to a get_steering_messages getter."""
    import queue as _queue

    def getter():
        out = []
        while True:
            try:
                out.append(sq.get_nowait())
            except _queue.Empty:
                break
        return out

    return getter


def registry(handler=lambda value="": f"ok:{value}"):
    tools = ToolRegistry()
    tools.register(
        Tool(
            "echo",
            "echo a value",
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            handler,
            "local_write",
        )
    )
    return tools


def execute(client, tools=None, maximum=3, hooks=None, **loop_options):
    events = []
    result = run_test_loop(
        client=client,
        model="scripted",
        system="system",
        messages=loop_options.pop("messages", [{"role": "user", "content": "hello"}]),
        tools=tools or registry(),
        max_iterations=maximum,
        max_tokens=100,
        emit=lambda kind, data: events.append((kind, data)),
        hooks=hooks,
        **loop_options,
    )
    return result, events


# ── existing tests (unchanged) ───────────────────────────────────


def test_plain_text_natural_stop():
    result, _ = execute(QueueClient(ModelResponse(text="done")))
    assert result.reply == "done"
    assert result.iterations == 1
    assert result.status == "completed"
    assert result.stop_reason == "stop"


def test_stop_reason_aliases_are_normalized():
    assert normalize_stop_reason("toolUse") == "tool_calls"
    assert normalize_stop_reason("tool_use") == "tool_calls"
    assert normalize_stop_reason("end_turn") == "stop"
    assert normalize_stop_reason("max_output_tokens") == "length"
    assert normalize_stop_reason("cancelled") == "aborted"
    assert normalize_stop_reason("provider_specific_failure") == "error"


def test_tool_round_trip_then_reply():
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="finished"),
    )
    result, events = execute(client)
    assert result.reply == "finished"
    assert result.tool_calls[0]["output"] == "ok:x"
    assert [kind for kind, _ in events].count("llm.completed") == 2
    assert client.calls[1]["messages"][-1]["role"] == "tool"


def test_turn_hooks_match_model_call_boundaries():
    seen = []
    hooks = LoopHooks(
        on_turn_end=lambda turn, _emit: seen.append((turn.turn_index, turn.status)),
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="finished"),
    )

    execute(client, hooks=hooks)

    assert seen == [(1, "tool_use"), (2, "completed")]


def test_unknown_tool_is_returned_to_model():
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "missing", {})]),
        ModelResponse(text="recovered"),
    )
    result, _ = execute(client)
    assert result.reply == "recovered"
    assert result.tool_calls[0]["output"].startswith("Error: unknown tool")


def test_tool_exception_is_surface_error():
    def broken(value):
        raise RuntimeError("boom")

    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="handled"),
    )
    result, _ = execute(client, registry(broken))
    assert "RuntimeError" not in result.tool_calls[0]["output"]
    assert "boom" in result.tool_calls[0]["output"]


def test_multi_tool_call_in_one_iteration():
    client = QueueClient(
        ModelResponse(
            tool_calls=[
                ToolCall("1", "echo", {"value": "a"}),
                ToolCall("2", "echo", {"value": "b"}),
            ]
        ),
        ModelResponse(text="both"),
    )
    result, _ = execute(client)
    assert [call["output"] for call in result.tool_calls] == ["ok:a", "ok:b"]


def test_iteration_limit_is_hard_stop():
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "a"})]),
        ModelResponse(tool_calls=[ToolCall("2", "echo", {"value": "b"})]),
    )
    result, events = execute(client, maximum=2)
    assert "最大迭代次数" in result.reply
    assert result.status == "failed"
    assert result.stop_reason == "error"
    assert events[-1][0] == "loop.limit_reached"


def test_provider_tool_reason_without_calls_is_failure():
    result, events = execute(
        QueueClient(ModelResponse(text="missing call", stop_reason="tool_calls"))
    )

    assert result.status == "failed"
    assert result.stop_reason == "error"
    assert "loop.stop_reason_mismatch" in [kind for kind, _ in events]


def test_permanent_model_error_returns_structured_failure():
    result, events = execute(QueueClient(RuntimeError("boom")))

    assert result.status == "failed"
    assert result.stop_reason == "error"
    assert result.aborted is False
    assert "boom" in result.error
    assert result.iterations == 1
    assert [data["status"] for kind, data in events if kind == "turn.completed"] == [
        "error"
    ]


def test_transient_model_error_retries_then_completes(monkeypatch):
    class Timeout(Exception):
        pass

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = QueueClient(Timeout("temporary"), ModelResponse(text="recovered"))

    result, _ = execute(client)

    assert len(client.calls) == 2
    assert result.status == "completed"
    assert result.stop_reason == "stop"
    assert result.reply == "recovered"


def test_empty_response_retries_then_fails():
    client = QueueClient(
        ModelResponse(),
        ModelResponse(),
        ModelResponse(),
    )

    result, events = execute(client)

    assert len(client.calls) == 3
    assert result.iterations == 3
    assert result.status == "failed"
    assert result.stop_reason == "error"
    assert "仍未生成回复" in result.error
    assert [kind for kind, _ in events].count("loop.empty_response_retry") == 2


# ── direction 1: streaming events ─────────────────────────────────


def test_streaming_text_events():
    """Streaming responses emit text.start → text.delta → text.end."""
    client = QueueClient(ModelResponse(text="hello world"))
    _, events = execute(client)
    kinds = [kind for kind, _ in events]
    assert "llm.text.start" in kinds
    assert "llm.text.delta" in kinds
    assert "llm.text.end" in kinds
    # start comes before delta comes before end
    start_idx = kinds.index("llm.text.start")
    delta_idx = kinds.index("llm.text.delta")
    end_idx = kinds.index("llm.text.end")
    assert start_idx < delta_idx < end_idx


def test_streaming_tool_call_events():
    """Tool calls emit tool_call.start → tool_call.delta → tool_call.end."""
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("c1", "echo", {"value": "hi"})]),
        ModelResponse(text="done"),
    )
    _, events = execute(client)
    kinds = [kind for kind, _ in events]
    assert "llm.tool_call.start" in kinds
    assert "llm.tool_call.delta" in kinds
    assert "llm.tool_call.end" in kinds


def test_streaming_events_include_iteration():
    """Streaming events carry the current iteration number."""
    client = QueueClient(ModelResponse(text="first"))
    _, events = execute(client)
    for kind, data in events:
        if kind.startswith("llm.text."):
            assert data["iteration"] == 1


# ── direction 2: tool lifecycle hooks ─────────────────────────────


def test_before_hook_can_block_execution():
    """A before_hook that returns a string blocks the tool."""
    def guard(name, args):
        return "blocked for testing"

    tools = ToolRegistry()
    tools.register(
        Tool("echo", "desc", {"type": "object", "properties": {"value": {"type": "string"}}},
             lambda value="": value, before_hook=guard)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "test"})]),
        ModelResponse(text="ok"),
    )
    result, _ = execute(client, tools)
    assert "blocked for testing" in result.tool_calls[0]["output"]


def test_after_hook_can_modify_result():
    """An after_hook can rewrite the tool output."""
    def add_prefix(name, args, result):
        result.output = f"[AUDIT] {result.output}"
        return result

    tools = ToolRegistry()
    tools.register(
        Tool("echo", "desc",
             {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
             lambda value: f"got:{value}", after_hook=add_prefix)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="done"),
    )
    result, _ = execute(client, tools)
    assert result.tool_calls[0]["output"] == "[AUDIT] got:x"


def test_registry_level_hooks_apply_to_all_tools():
    """Registry-level before/after hooks run unless tool overrides."""
    def global_after(name, args, result):
        result.output = f"[global] {result.output}"
        return result

    tools = ToolRegistry(after_hook=global_after)
    tools.register(
        Tool("echo", "desc",
             {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
             lambda value: f"{value}")
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "hello"})]),
        ModelResponse(text="ok"),
    )
    result, _ = execute(client, tools)
    assert result.tool_calls[0]["output"] == "[global] hello"


def test_tool_hook_overrides_registry_hook():
    """Per-tool hooks take precedence over registry-level hooks."""
    def global_after(name, args, result):
        result.output = f"[global] {result.output}"
        return result

    def tool_after(name, args, result):
        result.output = f"[tool] {result.output}"
        return result

    tools = ToolRegistry(after_hook=global_after)
    tools.register(
        Tool("echo", "desc",
             {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
             lambda value: value, after_hook=tool_after)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="ok"),
    )
    result, _ = execute(client, tools)
    assert result.tool_calls[0]["output"] == "[tool] x"


def test_terminate_on_success_ends_loop():
    """A tool with terminate_on_success=True ends the loop after execution."""
    tools = ToolRegistry()
    tools.register(
        Tool(
            "bye",
            "ends the session",
            {"type": "object", "properties": {}},
            lambda: "再见！",
            terminate_on_success=True,
        )
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "bye", {})]),
        # This second response should never be consumed
        ModelResponse(text="should not be reached"),
    )
    result, _ = execute(client, tools)
    assert result.reply == "任务已完成。"
    assert len(result.tool_calls) == 1
    assert result.iterations == 1
    assert result.status == "completed"
    assert result.stop_reason == "tool_calls"


def test_terminate_strips_marker_from_output():
    """The TERMINATE marker is stripped before recording the tool output."""
    tools = ToolRegistry()
    tools.register(
        Tool(
            "bye",
            "ends",
            {"type": "object", "properties": {}},
            lambda: "再见！",
            terminate_on_success=True,
        )
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "bye", {})]),
        ModelResponse(text="nope"),
    )
    result, _ = execute(client, tools)
    assert "__TERMINATE" not in result.tool_calls[0]["output"]
    assert result.tool_calls[0]["output"] == "再见！"


def test_tool_batch_stops_when_any_result_terminates():
    tools = ToolRegistry()
    tools.register(
        Tool(
            "stopper",
            "requests termination",
            {"type": "object", "properties": {}},
            lambda: "stop",
            terminate_on_success=True,
        )
    )
    tools.register(
        Tool(
            "worker",
            "keeps working",
            {"type": "object", "properties": {}},
            lambda: "work",
        )
    )
    client = QueueClient(
        ModelResponse(tool_calls=[
            ToolCall("1", "stopper", {}),
            ToolCall("2", "worker", {}),
        ]),
        ModelResponse(text="continued"),
    )

    result, _ = execute(client, tools)

    assert result.reply == "任务已完成。"
    assert len(client.calls) == 1


def test_tool_batch_stops_when_every_result_terminates():
    tools = ToolRegistry()
    for name in ("first", "second"):
        tools.register(
            Tool(
                name,
                "requests termination",
                {"type": "object", "properties": {}},
                lambda: "done",
                terminate_on_success=True,
            )
        )
    client = QueueClient(
        ModelResponse(tool_calls=[
            ToolCall("1", "first", {}),
            ToolCall("2", "second", {}),
        ]),
        ModelResponse(text="should not run"),
    )

    result, _ = execute(client, tools)

    assert result.reply == "任务已完成。"
    assert len(client.calls) == 1


# ── direction 3: token truncation protection ──────────────────────


def test_truncation_rejects_all_tool_calls():
    """When stop_reason='length', ALL tool calls are rejected."""
    client = QueueClient(
        ModelResponse(
            stop_reason="length",
            tool_calls=[
                ToolCall("1", "echo", {"value": "a"}),
                ToolCall("2", "echo", {"value": "b"}),
            ],
        ),
        ModelResponse(text="recovered after truncation"),
    )
    result, events = execute(client)
    kinds = [kind for kind, _ in events]
    assert "loop.truncation_rejected" in kinds
    assert "loop.length_recovery_exhausted" in kinds
    assert result.tool_calls == []
    assert result.status == "failed"
    assert result.stop_reason == "length"
    assert len(client.calls) == 1


def test_truncation_rejected_event_has_count():
    """The truncation_rejected event records how many calls were dropped."""
    client = QueueClient(
        ModelResponse(
            stop_reason="length",
            tool_calls=[ToolCall("1", "echo", {"value": "x"})],
        ),
        ModelResponse(text="ok"),
    )
    result, events = execute(client)
    assert result.status == "failed"
    assert result.stop_reason == "length"
    for kind, data in events:
        if kind == "loop.truncation_rejected":
            assert data["rejected_tool_calls"] == 1


def test_normal_stop_does_not_trigger_truncation():
    """Normal stop_reason='stop' should NOT trigger truncation logic."""
    client = QueueClient(
        ModelResponse(
            stop_reason="stop",
            tool_calls=[ToolCall("1", "echo", {"value": "test"})],
        ),
        ModelResponse(text="done"),
    )
    result, events = execute(client)
    kinds = [kind for kind, _ in events]
    assert "loop.truncation_rejected" not in kinds
    assert result.tool_calls[0]["output"] == "ok:test"  # tool should execute


def test_truncation_retry_consumes_iteration():
    """Truncation rejection counts as one iteration."""
    client = QueueClient(
        ModelResponse(
            stop_reason="length",
            tool_calls=[ToolCall("1", "echo", {"value": "x"})],
        ),
        ModelResponse(text="final"),
    )
    result, _ = execute(
        client,
        maximum=3,
        on_truncation=lambda: (
            "compacted system",
            [{"role": "user", "content": "retry after compaction"}],
        ),
    )
    assert result.iterations == 2
    assert result.reply == "final"
    assert result.status == "completed"
    assert result.stop_reason == "stop"


def test_truncation_recovery_exception_is_structured_failure():
    def fail_recovery():
        raise RuntimeError("cannot compact")

    result, events = execute(
        QueueClient(ModelResponse(stop_reason="length")),
        on_truncation=fail_recovery,
    )

    assert result.status == "failed"
    assert result.stop_reason == "length"
    assert "cannot compact" in result.error
    assert "loop.overflow_recovery_failed" in [kind for kind, _ in events]


def test_truncation_recovery_cap_is_hard_failure():
    client = QueueClient(
        ModelResponse(stop_reason="length"),
        ModelResponse(stop_reason="length"),
    )

    result, events = execute(
        client,
        maximum=3,
        max_length_recoveries=1,
        on_truncation=lambda: (
            "compacted system",
            [{"role": "user", "content": "retry"}],
        ),
    )

    assert result.status == "failed"
    assert result.stop_reason == "length"
    assert result.iterations == 2
    assert "loop.length_recovery_exhausted" in [kind for kind, _ in events]


# ── direction 4: parallel tool execution ─────────────────────────


def test_parallel_safe_tools_run_concurrently():
    """Two read tools execute in the same batch and both complete."""
    tools = ToolRegistry()
    tools.register(
        Tool("read_a", "reads A", {"type": "object", "properties": {}},
             lambda: "A", effect="read")
    )
    tools.register(
        Tool("read_b", "reads B", {"type": "object", "properties": {}},
             lambda: "B", effect="read")
    )
    client = QueueClient(
        ModelResponse(tool_calls=[
            ToolCall("1", "read_a", {}),
            ToolCall("2", "read_b", {}),
        ]),
        ModelResponse(text="done"),
    )
    result, events = execute(client, tools)
    assert [c["output"] for c in result.tool_calls] == ["A", "B"]
    # Should emit loop.tools.batch event
    kinds = [kind for kind, _ in events]
    assert "loop.tools.batch" in kinds


def test_parallel_preserves_order():
    """Results are always returned in the original call order."""
    tools = ToolRegistry()
    for i in range(5):
        tools.register(
            Tool(f"echo_{i}", "", {"type": "object", "properties": {}},
                 lambda i=i: f"result_{i}", effect="read")
        )
    calls = [ToolCall(str(i), f"echo_{i}", {}) for i in range(5)]
    client = QueueClient(
        ModelResponse(tool_calls=calls),
        ModelResponse(text="done"),
    )
    result, _ = execute(client, tools)
    assert [c["output"] for c in result.tool_calls] == [
        "result_0", "result_1", "result_2", "result_3", "result_4"
    ]


def test_sequential_tools_run_in_order():
    """Write-effect tools with parallel_safe=False run sequentially."""
    order = []
    def make_fn(name):
        def fn():
            order.append(name)
            return name
        return fn

    tools = ToolRegistry()
    tools.register(
        Tool("step1", "", {"type": "object", "properties": {}},
             make_fn("step1"), effect="local_write", parallel_safe=False)
    )
    tools.register(
        Tool("step2", "", {"type": "object", "properties": {}},
             make_fn("step2"), effect="local_write", parallel_safe=False)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[
            ToolCall("1", "step1", {}),
            ToolCall("2", "step2", {}),
        ]),
        ModelResponse(text="done"),
    )
    execute(client, tools)
    assert order == ["step1", "step2"]  # sequential, not interleaved


def test_mixed_parallel_and_sequential():
    """Sequential tools run first, then parallel tools."""
    order = []
    def make_fn(name, delay=0):
        import time
        def fn():
            order.append(name)
            if delay:
                time.sleep(delay)
            return name
        return fn

    tools = ToolRegistry()
    tools.register(
        Tool("write", "", {"type": "object", "properties": {}},
             make_fn("write"), effect="local_write", parallel_safe=False)
    )
    tools.register(
        Tool("read1", "", {"type": "object", "properties": {}},
             make_fn("read1", 0.05), effect="read")
    )
    tools.register(
        Tool("read2", "", {"type": "object", "properties": {}},
             make_fn("read2", 0.05), effect="read")
    )
    client = QueueClient(
        ModelResponse(tool_calls=[
            ToolCall("1", "write", {}),
            ToolCall("2", "read1", {}),
            ToolCall("3", "read2", {}),
        ]),
        ModelResponse(text="done"),
    )
    execute(client, tools)
    # write must run first (sequential), then reads (parallel — order may vary)
    assert order[0] == "write"
    assert set(order[1:]) == {"read1", "read2"}


def test_tool_error_in_parallel_batch_does_not_crash_loop():
    """A failing tool in a parallel batch returns error text, not exception."""
    def fail():
        raise RuntimeError("boom")

    tools = ToolRegistry()
    tools.register(
        Tool("bad", "", {"type": "object", "properties": {}},
             fail, effect="read")
    )
    tools.register(
        Tool("good", "", {"type": "object", "properties": {}},
             lambda: "ok", effect="read")
    )
    client = QueueClient(
        ModelResponse(tool_calls=[
            ToolCall("1", "bad", {}),
            ToolCall("2", "good", {}),
        ]),
        ModelResponse(text="recovered"),
    )
    result, _ = execute(client, tools)
    assert "boom" in result.tool_calls[0]["output"]
    assert result.tool_calls[1]["output"] == "ok"


# ── direction 5: interrupt ───────────────────────────────────────


def test_interrupt_stops_before_iteration():
    """Setting interrupt before the loop starts returns immediately."""
    import threading
    interrupt = threading.Event()
    interrupt.set()  # already aborted

    events = []
    result = run_test_loop(
        client=QueueClient(ModelResponse(text="never consumed")),
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=registry(),
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        interrupt=interrupt,
    )
    assert result.aborted is True
    assert result.status == "aborted"
    assert result.stop_reason == "aborted"
    assert result.iterations == 0
    assert "loop.aborted" in [k for k, _ in events]


def test_interrupt_is_checked_between_deltas():
    """Interrupt during streaming sets aborted=True."""
    import threading
    interrupt = threading.Event()

    # Create a client that sets interrupt mid-stream
    class InterruptingClient:
        def __init__(self):
            self.calls = []

        def stream_complete(self, **kwargs):
            self.calls.append(kwargs)
            yield StreamDelta(kind="text_delta", text="hello ")
            interrupt.set()  # interrupt mid-stream
            yield StreamDelta(kind="text_delta", text="world")
            yield StreamDelta(kind="done", stop_reason="stop")

        def complete(self, **kwargs):
            raise AssertionError("should use streaming")

    events = []
    result = run_test_loop(
        client=InterruptingClient(),
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=registry(),
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        interrupt=interrupt,
    )
    assert result.aborted is True
    assert result.status == "aborted"
    assert result.stop_reason == "aborted"
    kinds = [k for k, _ in events]
    assert "loop.stream_aborted" in kinds


# ── direction 6: steering messages ───────────────────────────────


def test_steering_injects_at_iteration_boundary():
    """A queued steering message appears before the next model call."""
    import queue
    sq: queue.Queue[str] = queue.Queue()
    sq.put("修正：我说的是 X 不是 Y")

    client = QueueClient(ModelResponse(text="understood"))
    events = []
    result = run_test_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=registry(),
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        get_steering_messages=_drain_queue(sq),
    )
    assert result.reply == "understood"
    kinds = [k for k, _ in events]
    assert "loop.steered" in kinds
    # Pi semantics: steering is injected as an ordinary user message.
    injected = client.calls[0]["messages"]
    assert any("修正：我说的是 X 不是 Y" == m.get("content") for m in injected)


def test_steering_empty_queue_is_noop():
    """An empty steering queue does not affect the loop."""
    import queue
    sq: queue.Queue[str] = queue.Queue()

    client = QueueClient(ModelResponse(text="ok"))
    events = []
    result = run_test_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=registry(),
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        get_steering_messages=_drain_queue(sq),
    )
    assert result.reply == "ok"
    assert "loop.steered" not in [k for k, _ in events]


def test_steering_during_tool_loop():
    """Steering injected between tool-execution iterations works."""
    import queue
    sq: queue.Queue[str] = queue.Queue()
    sq.put("correct")  # will be injected before the second model call

    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="corrected"),
    )
    events = []
    result = run_test_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=registry(),
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        get_steering_messages=_drain_queue(sq),
    )
    assert result.reply == "corrected"
    kinds = [k for k, _ in events]
    assert "loop.steered" in kinds


def test_multiple_steering_messages():
    """Multiple queued messages are all injected."""
    import queue
    sq: queue.Queue[str] = queue.Queue()
    sq.put("msg1")
    sq.put("msg2")

    client = QueueClient(ModelResponse(text="ack"))
    events = []
    run_test_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=registry(),
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        get_steering_messages=_drain_queue(sq),
    )
    # Two steering events
    steered = [d for k, d in events if k == "loop.steered"]
    assert len(steered) == 2
    assert steered[0]["message"] == "msg1"
    assert steered[1]["message"] == "msg2"


def test_pending_message_queue_defaults_to_one_at_a_time():
    pending = PendingMessageQueue()
    pending.enqueue("first")
    pending.enqueue("second")

    assert pending.drain() == ["first"]
    assert pending.has_items()
    assert pending.drain() == ["second"]
    assert not pending.has_items()


def test_pending_message_queue_all_mode_drains_batch():
    pending = PendingMessageQueue(mode="all")
    pending.enqueue("first")
    pending.enqueue("second")

    assert pending.drain() == ["first", "second"]
    assert pending.snapshot() == []


def test_steering_is_polled_after_turn_and_injected_before_next_model_call():
    queued = False
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="steered result"),
    )

    def get_steering_messages():
        nonlocal queued
        if client.calls and not queued:
            queued = True
            return ["change direction"]
        return []

    result, events = execute(
        client,
        get_steering_messages=get_steering_messages,
    )

    assert result.reply == "steered result"
    second_messages = client.calls[1]["messages"]
    tool_index = next(i for i, msg in enumerate(second_messages) if msg["role"] == "tool")
    steer_index = next(
        i for i, msg in enumerate(second_messages)
        if msg.get("content") == "change direction"
    )
    assert tool_index < steer_index
    assert "loop.steered" in [kind for kind, _ in events]


def test_follow_up_revives_same_trace_after_inner_loop_stops():
    delivered = False
    client = QueueClient(
        ModelResponse(text="first answer"),
        ModelResponse(text="follow-up answer"),
    )

    def get_follow_up_messages():
        nonlocal delivered
        if delivered:
            return []
        delivered = True
        return ["run tests too"]

    result, events = execute(
        client,
        get_follow_up_messages=get_follow_up_messages,
    )

    assert result.reply == "follow-up answer"
    assert result.iterations == 2
    assert any(
        msg.get("content") == "run tests too"
        for msg in client.calls[1]["messages"]
    )
    assert "loop.followed_up" in [kind for kind, _ in events]


def test_prepare_next_turn_updates_next_model_request():
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="done"),
    )
    prepared = False

    def prepare_next_turn(_context):
        nonlocal prepared
        if prepared:
            return None
        prepared = True
        return NextTurnUpdate(system="second system", model="second-model")

    result, _ = execute(client, prepare_next_turn=prepare_next_turn)

    assert result.reply == "done"
    assert client.calls[1]["system"] == "second system"
    assert client.calls[1]["model"] == "second-model"


def test_context_transform_and_llm_conversion_run_at_provider_boundary():
    client = QueueClient(ModelResponse(text="done"))

    def transform(messages):
        return [
            *messages,
            custom_message("internal", "not provider-visible", exclude_from_context=True),
            user_message("request-only context"),
        ]

    def convert(messages):
        # default_convert_to_llm drops the excluded custom message
        return default_convert_to_llm(messages)

    result, _ = execute(
        client,
        transform_context=transform,
        convert_to_llm=convert,
    )

    assert result.reply == "done"
    assert client.calls[0]["messages"][-1]["content"] == "request-only context"
    assert all(
        "internal" not in str(message)
        for message in client.calls[0]["messages"]
    )


def test_context_transform_failure_becomes_structured_trace_failure():
    def broken_transform(_messages):
        raise ValueError("bad context")

    result, events = execute(
        QueueClient(ModelResponse(text="never called")),
        transform_context=broken_transform,
    )

    assert result.status == "failed"
    assert result.stop_reason == "error"
    assert "bad context" in result.error
    assert "loop.context_transform_failed" in [kind for kind, _ in events]


def test_should_stop_after_turn_skips_queue_polls():
    steering_polls = 0
    follow_up_polls = 0
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="should not run"),
    )

    def get_steering_messages():
        nonlocal steering_polls
        steering_polls += 1
        return []

    def get_follow_up_messages():
        nonlocal follow_up_polls
        follow_up_polls += 1
        return ["should stay queued"]

    result, events = execute(
        client,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
        should_stop_after_turn=lambda _context: True,
    )

    assert len(client.calls) == 1
    assert result.status == "completed"
    assert steering_polls == 1
    assert follow_up_polls == 0
    assert "loop.stopped_after_turn" in [kind for kind, _ in events]


def test_config_before_tool_call_can_block_validated_request():
    executed: list[str] = []
    seen = []
    tools = registry(lambda value: executed.append(value) or f"ok:{value}")
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="handled"),
    )

    def before_tool_call(context):
        seen.append(context)
        return BeforeToolCallResult(block=True, reason="policy denied")

    result, _ = execute(
        client,
        tools,
        before_tool_call=before_tool_call,
    )

    assert executed == []
    assert len(seen) == 1
    assert seen[0].args == {"value": "x"}
    assert seen[0].context.system_prompt == "system"
    assert seen[0].tool_call["function"]["name"] == "echo"
    assert result.tool_calls[0]["output"] == "Error: policy denied"


def test_config_after_tool_call_overrides_result_field_by_field():
    seen = []
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "echo", {"value": "x"})]),
        ModelResponse(text="handled"),
    )

    def after_tool_call(context):
        seen.append(context)
        return AfterToolCallResult(
            output="redacted",
            details={"source": "afterToolCall"},
            is_error=True,
        )

    result, events = execute(client, after_tool_call=after_tool_call)

    assert seen[0].result.output == "ok:x"
    assert seen[0].is_error is False
    assert result.tool_calls[0] == {
        "tool": "echo",
        "label": "echo",
        "tool_call_id": "1",
        "args": {"value": "x"},
        "output": "redacted",
        "is_error": True,
        "details": {"source": "afterToolCall"},
    }
    completed = [data for kind, data in events if kind == "tool.completed"]
    assert completed[0]["status"] == "error"


def test_parallel_mode_finishes_all_preflight_before_execution():
    timeline: list[str] = []
    tools = ToolRegistry()
    for name in ("read_a", "read_b"):
        tools.register(Tool(
            name,
            "",
            {"type": "object", "properties": {}},
            lambda name=name: timeline.append(f"execute:{name}") or name,
            effect="read",
        ))
    client = QueueClient(
        ModelResponse(tool_calls=[
            ToolCall("1", "read_a", {}),
            ToolCall("2", "read_b", {}),
        ]),
        ModelResponse(text="done"),
    )

    def before_tool_call(context):
        timeline.append(f"before:{context.tool_call['function']['name']}")
        return None

    execute(
        client,
        tools,
        before_tool_call=before_tool_call,
        tool_execution="parallel",
    )

    assert timeline[:2] == ["before:read_a", "before:read_b"]
    assert set(timeline[2:]) == {"execute:read_a", "execute:read_b"}


def test_config_sequential_mode_forces_read_tools_to_run_one_at_a_time():
    import threading
    import time

    active = 0
    max_active = 0
    lock = threading.Lock()

    def read_tool():
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return "done"

    tools = ToolRegistry()
    for name in ("read_a", "read_b"):
        tools.register(Tool(
            name,
            "",
            {"type": "object", "properties": {}},
            read_tool,
            effect="read",
        ))
    client = QueueClient(
        ModelResponse(tool_calls=[
            ToolCall("1", "read_a", {}),
            ToolCall("2", "read_b", {}),
        ]),
        ModelResponse(text="done"),
    )

    execute(client, tools, tool_execution="sequential")

    assert max_active == 1


# ── tool system v2: structured results ───────────────────────────


def test_tool_can_return_toolresult_directly():
    """A tool returning ToolResult is used as-is."""
    tools = ToolRegistry()
    tools.register(
        Tool("check", "returns structured result",
             {"type": "object", "properties": {}},
             lambda: ToolResult(output="all good", is_error=False, details={"score": 42}))
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "check", {})]),
        ModelResponse(text="done"),
    )
    result, events = execute(client, tools)
    assert result.tool_calls[0]["output"] == "all good"
    assert result.tool_calls[0]["details"]["score"] == 42
    # Should NOT be marked as error
    tool_events = [d for k, d in events if k == "tool.completed"]
    assert tool_events[0]["status"] == "ok"


def test_tool_can_return_error_result():
    """ToolResult with is_error=True is reported as error."""
    tools = ToolRegistry()
    tools.register(
        Tool("fail", "always fails",
             {"type": "object", "properties": {}},
             lambda: ToolResult(output="网络超时", is_error=True))
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "fail", {})]),
        ModelResponse(text="recovered"),
    )
    result, events = execute(client, tools)
    tool_events = [d for k, d in events if k == "tool.completed"]
    assert tool_events[0]["status"] == "error"


def test_tool_with_terminate_flag():
    """ToolResult with terminate=True ends the loop."""
    tools = ToolRegistry()
    tools.register(
        Tool("done", "finishes",
             {"type": "object", "properties": {}},
             lambda: ToolResult(output="再见", terminate=True))
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "done", {})]),
        ModelResponse(text="never"),
    )
    result, _ = execute(client, tools)
    assert result.reply == "任务已完成。"
    assert result.iterations == 1


# ── tool system v2: prepare_args ─────────────────────────────────


def test_prepare_args_transforms_input():
    """prepare_args hook transforms raw model arguments before validation."""
    def normalise(args: dict) -> dict:
        if "email" in args:
            args["email"] = args["email"].strip().lower()
        return args

    tools = ToolRegistry()
    tools.register(
        Tool("subscribe", "adds email",
             {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
             lambda email: f"subscribed {email}",
             prepare_args=normalise)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "subscribe", {"email": "  USER@Example.COM  "})]),
        ModelResponse(text="ok"),
    )
    result, _ = execute(client, tools)
    assert result.tool_calls[0]["output"] == "subscribed user@example.com"


def test_prepare_args_error_is_surfaced():
    """If prepare_args raises, it becomes an error result."""
    def bad_prepare(args):
        raise ValueError("invalid date format")

    tools = ToolRegistry()
    tools.register(
        Tool("create", "",
             {"type": "object", "properties": {"date": {"type": "string"}}},
             lambda date: date,
             prepare_args=bad_prepare)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "create", {"date": "nope"})]),
        ModelResponse(text="ok"),
    )
    result, _ = execute(client, tools)
    assert result.tool_calls[0]["output"].startswith("Error preparing")


# ── tool system v2: AbortSignal ──────────────────────────────────


def test_abort_signal_passed_to_tool():
    """A tool that declares _abort receives an AbortHandle."""
    received = []

    def long_task(_abort=None):
        received.append(_abort is not None)
        received.append(hasattr(_abort, 'aborted'))
        return "done"

    tools = ToolRegistry()
    tools.register(
        Tool("long", "", {"type": "object", "properties": {}}, long_task)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "long", {})]),
        ModelResponse(text="ok"),
    )
    execute(client, tools)
    assert received == [True, True]


def test_tool_can_check_abort():
    """A tool can poll _abort.aborted and return early."""

    def cancellable(_abort: AbortHandle | None = None):
        if _abort and _abort.aborted:
            return "cancelled"
        return "completed"

    tools = ToolRegistry()
    tools.register(
        Tool("task", "", {"type": "object", "properties": {}}, cancellable)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "task", {})]),
        ModelResponse(text="ok"),
    )
    result, _ = execute(client, tools)
    # Abort not set → should complete normally
    assert result.tool_calls[0]["output"] == "completed"


def test_interrupt_sets_tool_abort_handle():
    """A tool polling _abort.aborted sees a user interrupt mid-run."""
    import threading
    interrupt = threading.Event()
    observed = []

    def cancellable(_abort=None):
        # In reality the user presses Ctrl-C from another thread; here we
        # set the interrupt directly to model that mid-run signal.
        interrupt.set()
        observed.append(bool(_abort and _abort.aborted))
        return "cancelled" if (_abort and _abort.aborted) else "ran to completion"

    tools = ToolRegistry()
    tools.register(
        Tool("long", "", {"type": "object", "properties": {}}, cancellable)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "long", {})]),
        ModelResponse(text="done"),
    )
    result, _ = execute(client, tools, interrupt=interrupt)

    # The tool must see the interrupt through its _abort handle…
    assert observed == [True]
    # …and the loop aborts at the next safe point.
    assert result.aborted is True
    assert result.status == "aborted"
    assert result.stop_reason == "aborted"


# ── tool system v2: onUpdate ─────────────────────────────────────


def test_on_update_callback_passed_to_tool():
    """A tool that declares _on_update receives a callable."""
    updates = []

    def progress_tool(_on_update=None):
        if _on_update:
            _on_update("25%")
            _on_update("50%")
            _on_update("100%")
        return "done"

    tools = ToolRegistry()
    tools.register(
        Tool("progress", "", {"type": "object", "properties": {}}, progress_tool)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "progress", {})]),
        ModelResponse(text="ok"),
    )
    events = []
    run_test_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
    )
    progress_events = [d for k, d in events if k == "tool.progress"]
    assert len(progress_events) == 3
    assert progress_events[0]["delta"] == "25%"
    assert progress_events[2]["delta"] == "100%"


# ── tool system v2: timeout ──────────────────────────────────────


def test_tool_with_timeout_runs_in_thread():
    """A tool with timeout > 0 completes normally when fast enough."""
    tools = ToolRegistry()
    tools.register(
        Tool("fast", "", {"type": "object", "properties": {}},
             lambda: "ok", timeout=5.0)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "fast", {})]),
        ModelResponse(text="done"),
    )
    result, _ = execute(client, tools)
    assert result.tool_calls[0]["output"] == "ok"


def test_tool_timeout_returns_error():
    """A tool that exceeds its timeout returns a timeout error."""
    import time

    def slow():
        time.sleep(0.5)
        return "too late"

    tools = ToolRegistry()
    tools.register(
        Tool("slow", "", {"type": "object", "properties": {}},
             slow, timeout=0.1)
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "slow", {})]),
        ModelResponse(text="recovered"),
    )
    result, _ = execute(client, tools)
    assert "timed out" in result.tool_calls[0]["output"].lower()


def test_legacy_string_tool_still_works():
    """Tools returning plain strings are auto-wrapped in ToolResult."""
    tools = ToolRegistry()
    tools.register(
        Tool("legacy", "", {"type": "object", "properties": {}},
             lambda: "simple result")
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "legacy", {})]),
        ModelResponse(text="ok"),
    )
    result, _ = execute(client, tools)
    assert result.tool_calls[0]["output"] == "simple result"


def test_legacy_error_string_is_detected():
    """Strings starting with 'Error' are auto-marked as errors."""
    tools = ToolRegistry()
    tools.register(
        Tool("oops", "", {"type": "object", "properties": {}},
             lambda: "Error: something went wrong")
    )
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", "oops", {})]),
        ModelResponse(text="fixed"),
    )
    _, events = execute(client, tools)
    tool_events = [d for k, d in events if k == "tool.completed"]
    assert tool_events[0]["status"] == "error"


def test_transform_context_cannot_mutate_agent_state():
    """The loop hands transform_context a snapshot copy: mutating it must
    not corrupt the caller's message list (batch A, plan §4.7)."""
    client = QueueClient(ModelResponse(text="done"))

    def transform(messages):
        messages.clear()  # hostile transform: wipes its input
        messages.append(user_message("replacement"))
        return messages

    context_messages = [{"role": "user", "content": "hello"}]
    result, _ = execute(client, transform_context=transform, messages=context_messages)

    assert result.reply == "done"
    # the caller's list was not cleared by the transform
    assert len(context_messages) >= 2
    assert context_messages[0].role == "user"
    assert context_messages[0].content == "hello"
