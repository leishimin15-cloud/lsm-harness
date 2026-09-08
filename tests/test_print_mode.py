"""Print mode (``lsm -p``): stdout streaming, stderr tools, exit codes."""

from __future__ import annotations

from lsm_harness.ai.types import ModelResponse, ToolCall, Usage
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.gateway.print_mode import run_print

from helpers import QueueClient, client_stream_fn


def _harness(tmp_path, *responses: ModelResponse) -> Harness:
    client = QueueClient(*responses)
    return Harness(
        settings=Settings(api_key="scripted", home=tmp_path),
        client=client,
        stream_fn=client_stream_fn(client),
    )


def test_print_streams_answer_to_stdout(tmp_path, capsys):
    app = _harness(tmp_path, ModelResponse(text="答案是 42"))
    try:
        rc = run_print("生命的意义？", app=app)
    finally:
        app.close()

    out, err = capsys.readouterr()
    assert rc == 0
    assert out == "答案是 42\n"
    assert err == ""


def test_print_reports_tools_on_stderr(tmp_path, capsys):
    app = _harness(
        tmp_path,
        ModelResponse(
            tool_calls=[ToolCall("1", "list_dir", {})],
            stop_reason="tool_calls",
            usage=Usage(8, 8),
        ),
        ModelResponse(text="看完了"),
    )
    try:
        rc = run_print("看看目录", app=app)
    finally:
        app.close()

    out, err = capsys.readouterr()
    assert rc == 0
    assert out == "看完了\n"
    assert "⚙" in err and "✓" in err
    assert "列出目录" in err  # tool label, not the raw reply


def test_print_failure_exits_1_with_error_on_stderr(tmp_path, capsys):
    app = _harness(tmp_path, RuntimeError("model exploded"))
    try:
        rc = run_print("会失败", app=app)
    finally:
        app.close()

    out, err = capsys.readouterr()
    assert rc == 1
    assert "model exploded" in err
