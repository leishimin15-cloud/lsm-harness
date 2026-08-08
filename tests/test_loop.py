from lsm_harness.loop.agent import run_loop
from lsm_harness.tools.registry import AbortHandle, ExecutionContext, Tool, ToolRegistry, ToolResult
from lsm_harness.types import ModelResponse, StreamDelta, ToolCall, Usage

from helpers import QueueClient


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


def execute(client, tools=None, maximum=3):
    events = []
    result = run_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hello"}],
        tools=tools or registry(),
        max_iterations=maximum,
        max_tokens=100,
        emit=lambda kind, data: events.append((kind, data)),
    )
    return result, events


# ── existing tests (unchanged) ───────────────────────────────────


def test_plain_text_natural_stop():
    result, _ = execute(QueueClient(ModelResponse(text="done")))
    assert result.reply == "done"
    assert result.iterations == 1


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
    assert events[-1][0] == "loop.limit_reached"


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
    # Tool calls from truncated response should NOT appear in result
    assert result.reply == "recovered after truncation"


def test_truncation_rejected_event_has_count():
    """The truncation_rejected event records how many calls were dropped."""
    client = QueueClient(
        ModelResponse(
            stop_reason="length",
            tool_calls=[ToolCall("1", "echo", {"value": "x"})],
        ),
        ModelResponse(text="ok"),
    )
    _, events = execute(client)
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
    result, _ = execute(client, maximum=3)
    assert result.iterations == 2
    assert result.reply == "final"


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
    result = run_loop(
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
    assert result.iterations == 1
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
    result = run_loop(
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
    result = run_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=registry(),
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        steering_queue=sq,
    )
    assert result.reply == "understood"
    kinds = [k for k, _ in events]
    assert "loop.steered" in kinds
    # Verify the steering message was injected into messages
    injected = client.calls[0]["messages"]
    assert any("中途纠正" in str(m.get("content", "")) for m in injected)


def test_steering_empty_queue_is_noop():
    """An empty steering queue does not affect the loop."""
    import queue
    sq: queue.Queue[str] = queue.Queue()

    client = QueueClient(ModelResponse(text="ok"))
    events = []
    result = run_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=registry(),
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        steering_queue=sq,
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
    result = run_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=registry(),
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        steering_queue=sq,
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
    run_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=registry(),
        max_iterations=3,
        max_tokens=100,
        emit=lambda k, d: events.append((k, d)),
        steering_queue=sq,
    )
    # Two steering events
    steered = [d for k, d in events if k == "loop.steered"]
    assert len(steered) == 2
    assert steered[0]["message"] == "msg1"
    assert steered[1]["message"] == "msg2"


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
    from lsm_harness.tools.registry import AbortHandle

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
    run_loop(
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

