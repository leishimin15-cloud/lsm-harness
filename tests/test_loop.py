from lsm_harness.loop.agent import run_loop
from lsm_harness.tools.registry import Tool, ToolRegistry
from lsm_harness.types import ModelResponse, ToolCall

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

