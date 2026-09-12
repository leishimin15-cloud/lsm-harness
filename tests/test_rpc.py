"""RPC mode (``lsm rpc``): JSONL command protocol over stdin/stdout."""

from __future__ import annotations

import json
import threading
import time
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lsm_harness.ai.api.common import snapshot
from lsm_harness.ai.types import AssistantMessageEvent, Model, ModelResponse, Usage
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.gateway.rpc import MAX_RECORD_BYTES, RpcServer, run_rpc

from helpers import QueueClient, client_stream_fn


def _run(tmp_path, input_lines, *responses, stream_fn=None):
    """Serve the given JSONL input against a scripted Harness.

    Returns (exit_code, [parsed output lines]).
    """
    settings = Settings(api_key="scripted", home=tmp_path)
    conn = connect(tmp_path, check_same_thread=False)
    if stream_fn is None:
        client = QueueClient(*responses)
        stream_fn = client_stream_fn(client)
    else:
        client = QueueClient()
    # reasoning=True:set/cycle thinking 需要模型支持推理,否则全 clamp 成 off
    client.model = Model(
        id="rpc-main", api="legacy-client", provider="injected", reasoning=True
    )
    app = Harness(settings=settings, client=client, conn=conn, stream_fn=stream_fn)
    stdin = StringIO("".join(line + "\n" for line in input_lines))
    stdout = StringIO()
    rc = run_rpc(app=app, stdin=stdin, stdout=stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    return rc, lines


def _responses(lines):
    return [line for line in lines if line["type"] == "response"]


def _events(lines):
    return [line["event"] for line in lines if line["type"] == "event"]


def test_get_state_reports_runtime(tmp_path):
    rc, lines = _run(tmp_path, ['{"id":"1","type":"get_state"}'])
    assert rc == 0
    (answer,) = _responses(lines)
    assert answer["id"] == "1"
    assert answer["command"] == "get_state"
    assert answer["success"] is True
    assert answer["is_running"] is False
    assert answer["session_id"]


def test_prompt_streams_events_and_answers(tmp_path):
    # 末尾的 get_messages 会让主线程 join 等运行结束，再读到 EOF——
    # 否则 EOF 触发的"abort 运行中 turn"与 worker 完成存在时序竞争。
    rc, lines = _run(
        tmp_path,
        [
            '{"id":"1","type":"prompt","message":"你好"}',
            '{"id":"2","type":"get_messages"}',
        ],
        ModelResponse(text="你好呀", usage=Usage(4, 4)),
    )
    assert rc == 0
    answers = _responses(lines)
    answer = answers[0]
    assert answer["command"] == "prompt" and answer["success"] is True
    types = [event["type"] for event in _events(lines)]
    assert "llm.text.delta" in types
    assert "trace.completed" in types
    session_event_types = [
        line["event"]["type"]
        for line in lines
        if line["type"] == "session_event"
    ]
    assert "entry_appended" in session_event_types
    assert "agent_settled" in session_event_types
    deltas = [
        event["data"]["text"] for event in _events(lines)
        if event["type"] == "llm.text.delta"
    ]
    assert "".join(deltas) == "你好呀"


def test_prompt_while_running_steers(tmp_path):
    started = threading.Event()

    def blocking_stream(_model, _context, options):
        started.set()
        while not (options.interrupt and options.interrupt.is_set()):
            time.sleep(0.005)
        yield AssistantMessageEvent(
            "error",
            snapshot(text="", thinking="", pending={}, stop_reason="aborted"),
            error_category="aborted",
        )

    rc, lines = _run(
        tmp_path,
        [
            '{"id":"1","type":"prompt","message":"先跑着"}',
            '{"id":"2","type":"prompt","message":"插一句","streamingBehavior":"steer"}',
            '{"id":"3","type":"abort"}',
        ],
        stream_fn=blocking_stream,
    )
    assert rc == 0
    answers = _responses(lines)
    assert answers[0]["command"] == "prompt" and answers[0]["success"] is True
    assert answers[1]["queued_as"] == "steer"
    assert answers[1]["success"] is True
    assert answers[2]["command"] == "abort" and answers[2]["success"] is True


def test_unknown_command_and_invalid_json(tmp_path):
    rc, lines = _run(
        tmp_path,
        ['{"id":"1","type":"explode"}', "not json at all"],
    )
    assert rc == 0
    answers = _responses(lines)
    assert answers[0]["success"] is False
    assert "unknown command" in answers[0]["error"]
    assert answers[1]["success"] is False
    assert "invalid JSON" in answers[1]["error"]


def test_oversized_record_is_rejected(tmp_path):
    big = "x" * (MAX_RECORD_BYTES + 1)
    rc, lines = _run(tmp_path, [big])
    assert rc == 0
    (answer,) = _responses(lines)
    assert answer["success"] is False
    assert "16 MiB" in answer["error"]


def test_session_and_thinking_commands(tmp_path):
    rc, lines = _run(
        tmp_path,
        [
            '{"id":"1","type":"get_state"}',
            '{"id":"2","type":"cycle_thinking_level"}',
            '{"id":"3","type":"set_thinking_level","level":"enabled"}',
            '{"id":"4","type":"set_thinking_level","level":"bogus"}',
            '{"id":"5","type":"new_session"}',
            '{"id":"6","type":"switch_session","session_id":"不存在"}',
        ],
    )
    assert rc == 0
    answers = _responses(lines)
    original_session = answers[0]["session_id"]
    assert answers[1]["success"] is True and answers[1]["level"] == "minimal"  # off → 下一档
    assert answers[2]["success"] is True and answers[2]["level"] == "high"  # enabled 归一为 high
    assert answers[3]["success"] is False
    assert answers[4]["success"] is True
    assert answers[4]["session_id"] != original_session
    assert answers[5]["success"] is False


def test_get_messages_after_run(tmp_path):
    rc, lines = _run(
        tmp_path,
        [
            '{"id":"1","type":"prompt","message":"记一句"}',
            '{"id":"2","type":"get_messages"}',
        ],
        ModelResponse(text="记住了", usage=Usage(4, 4)),
    )
    assert rc == 0
    answers = _responses(lines)
    assert answers[1]["success"] is True
    contents = [m.get("content") for m in answers[1]["messages"]]
    assert "记一句" in contents
    assert "记住了" in contents


def test_compact_command(tmp_path):
    rc, lines = _run(tmp_path, ['{"id":"1","type":"compact"}'])
    assert rc == 0
    (answer,) = _responses(lines)
    # Empty session: nothing to compact, but the command round-trips.
    assert answer["command"] == "compact"
    assert answer["success"] is False


def test_eof_closes_cleanly(tmp_path):
    rc, lines = _run(tmp_path, [])
    assert rc == 0
    assert lines == []


# ── 批次 R-2: 协议结构验证 + 忙碌状态策略 ─────────────────────
# 问题原因：JSON 解析成功不代表协议合法 —— []/null 会在 .get() 处异常，
# {"type": []} 会在命令字典查找处异常；且 _busy 期间 new_session /
# switch_session / set_model 仍可执行，会让运行中的事件和持久化串会话。


@pytest.mark.parametrize(
    "record",
    ["[]", "null", '{"type": []}', '"text"', "42", '{"id":"1"}'],
)
def test_malformed_request_shapes_get_error_and_service_continues(tmp_path, record):
    rc, lines = _run(tmp_path, [record, '{"id":"ok","type":"get_state"}'])
    assert rc == 0
    answers = _responses(lines)
    assert answers[0]["success"] is False
    # 错误请求之后服务不退出：后续合法请求照常处理。
    assert answers[1]["success"] is True
    assert answers[1]["command"] == "get_state"


def test_command_param_type_errors_are_protocol_errors(tmp_path):
    rc, lines = _run(
        tmp_path,
        [
            '{"id":"1","type":"set_model","provider":["deepseek"]}',
            '{"id":"2","type":"set_model","provider":"deepseek","model":42}',
            '{"id":"3","type":"switch_session","session_id":42}',
            '{"id":"4","type":"prompt"}',
        ],
    )
    assert rc == 0
    answers = _responses(lines)
    assert all(a["success"] is False for a in answers)
    assert "provider" in answers[0]["error"]
    assert "must be a string" in answers[1]["error"]
    assert "session_id" in answers[2]["error"]
    assert "missing message" in answers[3]["error"]


@pytest.mark.parametrize("field", ["model", "small_model"])
@pytest.mark.parametrize("value", [[], {}, 0, False])
def test_falsy_non_string_model_params_are_rejected(field, value):
    """[]/{}/0/False 必须先过类型检查——不能被 ``or ""`` 洗成"用默认模型"。"""
    app = SimpleNamespace(
        is_running=False,
        switch_model=Mock(),
        settings=SimpleNamespace(model="default"),
    )
    output = StringIO()
    server = RpcServer(app, StringIO(), output)
    server._dispatch({"type": "set_model", "provider": "deepseek", field: value})
    response = json.loads(output.getvalue())
    assert response["success"] is False, f"invalid {field} accepted: {value!r}"
    app.switch_model.assert_not_called()


def _blocking_stream(entered):
    """Stream that parks inside the run until interrupted."""

    def stream(_model, _context, options):
        entered.set()
        while not (options.interrupt and options.interrupt.is_set()):
            time.sleep(0.005)
        yield AssistantMessageEvent(
            "error",
            snapshot(text="", thinking="", pending={}, stop_reason="aborted"),
            error_category="aborted",
        )

    return stream


def test_state_commands_rejected_while_busy(tmp_path):
    entered = threading.Event()
    rc, lines = _run(
        tmp_path,
        [
            '{"id":"0","type":"get_state"}',
            '{"id":"1","type":"prompt","message":"先跑着"}',
            '{"id":"2","type":"new_session"}',
            '{"id":"3","type":"switch_session","session_id":"x"}',
            '{"id":"4","type":"set_model","provider":"deepseek"}',
            '{"id":"5","type":"set_thinking_level","level":"enabled"}',
            '{"id":"6","type":"cycle_thinking_level"}',
            '{"id":"7","type":"compact"}',
            '{"id":"8","type":"abort"}',
        ],
        stream_fn=_blocking_stream(entered),
    )
    assert rc == 0
    answers = _responses(lines)
    original_session = answers[0]["session_id"]
    assert answers[1]["success"] is True
    for answer in answers[2:8]:
        assert answer["success"] is False, answer
        assert "while a run is active" in answer["error"]
    # 运行期间控制能力保留：abort 照常可用。
    assert answers[8]["command"] == "abort" and answers[8]["success"] is True
    # 所有事件仍归属运行开始时的会话。
    event_sessions = {event.get("session_id") for event in _events(lines)}
    assert event_sessions == {original_session}


def test_is_running_covers_startup_window_before_worker_runs(tmp_path):
    """begin_run 在主线程同步完成：worker 尚未被调度时 is_running 已为真，
    变更命令（new_session）必须被拒绝且不得调用变更函数——启动窗口不存在。"""
    settings = Settings(api_key="scripted", home=tmp_path)
    conn = connect(tmp_path, check_same_thread=False)
    entered = threading.Event()
    app = Harness(
        settings=settings,
        client=QueueClient(),
        conn=conn,
        stream_fn=_blocking_stream(entered),
    )
    output = StringIO()
    server = RpcServer(app, StringIO(), output)
    try:
        server._start_worker("跑着")
        # 不做任何 sleep：接受即 running，worker 是否已被调度无关。
        assert app.is_running is True

        server._dispatch({"id": "1", "type": "new_session"})

        answer = json.loads(output.getvalue())
        assert answer["success"] is False
        assert "while a run is active" in answer["error"]
    finally:
        app.abort()
        if server._worker is not None:
            server._worker.join(5)
        app.close()


def test_run_events_and_persistence_pinned_to_starting_session(tmp_path):
    """即便运行中会话被（绕过 RPC 守卫）直接换掉，本次运行的事件与
    最终 SQLite 持久化仍归属运行开始时的会话。"""
    settings = Settings(api_key="scripted", home=tmp_path)
    conn = connect(tmp_path, check_same_thread=False)
    entered = threading.Event()
    release = threading.Event()

    def parked_stream(_model, _context, _options):
        entered.set()
        release.wait(5)
        final = snapshot(text="答完了", thinking="", pending={})
        yield AssistantMessageEvent("start", snapshot(text="", thinking="", pending={}))
        yield AssistantMessageEvent("text_start", final)
        yield AssistantMessageEvent("text_delta", final, text_delta="答完了")
        yield AssistantMessageEvent("text_end", final)
        yield AssistantMessageEvent("done", final)

    app = Harness(
        settings=settings,
        client=QueueClient(),
        conn=conn,
        stream_fn=parked_stream,
    )
    events = []
    original = app.session.session_id
    worker = threading.Thread(
        target=lambda: app.respond("跑着", observer=events.append),
        daemon=True,
    )
    worker.start()
    assert entered.wait(5)
    app.session.start_new()  # 模拟绕过守卫的会话切换
    release.set()
    worker.join(10)
    try:
        assert app.session.session_id != original
        assert events
        assert {event.session_id for event in events} == {original}
        rows = conn.execute("SELECT DISTINCT session_id FROM chat_log").fetchall()
        assert {row[0] for row in rows} == {original}
        # 旧运行的问答不得追加进新会话的内存 history。
        assert app.session.history == [], (
            f"new session history polluted: {app.session.history}"
        )
    finally:
        app.close()


def test_state_commands_succeed_after_run_completes(tmp_path):
    rc, lines = _run(
        tmp_path,
        [
            '{"id":"1","type":"prompt","message":"快答"}',
            '{"id":"2","type":"get_messages"}',  # join 等待运行结束
            '{"id":"3","type":"new_session"}',
            '{"id":"4","type":"set_thinking_level","level":"enabled"}',
        ],
        ModelResponse(text="答完了", usage=Usage(4, 4)),
    )
    assert rc == 0
    answers = _responses(lines)
    assert answers[2]["command"] == "new_session" and answers[2]["success"] is True
    assert answers[3]["success"] is True and answers[3]["level"] == "high"  # enabled 归一为 high
