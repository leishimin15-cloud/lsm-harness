"""Harness 运行控制：接受即 running、idle 不提前（阶段 1 · Batch 2）。

验收 (c)：运行结束包含持久化与终态事件处理，之后 wait_for_idle 才报闲。
验收 (d) 的地基：is_running 从接受瞬间为真，任何前端共用同一 busy 事实。
"""

from __future__ import annotations

import threading

import pytest

from lsm_harness.ai.api.common import snapshot
from lsm_harness.ai.types import AssistantMessageEvent, ModelResponse, Usage
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect

from helpers import QueueClient


def _app(tmp_path, *responses, stream_fn=None):
    settings = Settings(api_key="scripted", home=tmp_path)
    conn = connect(tmp_path, check_same_thread=False)
    client = QueueClient(*responses)
    return Harness(
        settings=settings,
        client=client,
        conn=conn,
        stream_fn=stream_fn or client.as_stream_fn(),
    ), client


def _parked_stream(entered: threading.Event, release: threading.Event):
    def stream(_model, _context, _options):
        entered.set()
        release.wait(5)
        final = snapshot(text="答完了", thinking="", pending={})
        yield AssistantMessageEvent("start", snapshot(text="", thinking="", pending={}))
        yield AssistantMessageEvent("text_start", final)
        yield AssistantMessageEvent("text_delta", final, text_delta="答完了")
        yield AssistantMessageEvent("text_end", final)
        yield AssistantMessageEvent("done", final)

    return stream


def test_begin_run_makes_is_running_synchronous(tmp_path):
    """接受即 running：begin_run 返回的瞬间 is_running 已为真——
    不存在"已接受但还没 running"的启动窗口。"""
    app, _client = _app(tmp_path)
    try:
        assert app.is_running is False
        active = app.begin_run()
        assert app.is_running is True  # 无线程、无等待，同步为真
        # begin 了就必须 finish——本测试不涉及 run 主体，手动收尾。
        app.agent.finish(active)
        assert app.is_running is False
    finally:
        app.close()


def test_abort_during_acceptance_window_interrupts_run(tmp_path):
    """begin_run 之后、loop 启动之前 abort：interrupt 在首个 Turn 前已 set，
    运行以 aborted 结束且模型从未被调用。"""
    app, client = _app(tmp_path, ModelResponse(text="不应出现", usage=Usage(1, 1)))
    try:
        active = app.begin_run()
        assert app.abort() is True  # 窗口内 abort 不再丢失
        result = app.respond("问", source="test", active_run=active)
        assert result.status == "aborted"
        assert client.calls == []
        assert app.is_running is False
    finally:
        app.close()


def test_wait_for_idle_reports_only_after_terminal_events_and_persistence(tmp_path):
    """idle 报告发生在终态事件与持久化之后，不提前。"""
    entered, release = threading.Event(), threading.Event()
    app, _client = _app(tmp_path, stream_fn=_parked_stream(entered, release))
    events = []

    def work():
        app.respond("跑着", observer=events.append, source="test")

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        # 流被按住：不提前报闲。
        assert app.wait_for_idle(0.05) is False
        release.set()
        assert app.wait_for_idle(5) is True
        worker.join(5)
        # wait_for_idle 返回时：终态事件已发出……
        event_types = [e.type for e in events]
        assert "trace.completed" in event_types
        # ……且 SQLite 持久化已完成。
        rows = app.conn.execute("SELECT COUNT(*) FROM chat_log").fetchone()
        assert rows[0] >= 2  # user + assistant
    finally:
        app.close()


def test_respond_without_active_run_begins_and_finishes_internally(tmp_path):
    """不传 active_run 的既有调用路径（print_mode/eval/subagent）行为不变。"""
    app, _client = _app(tmp_path, ModelResponse(text="好", usage=Usage(2, 2)))
    try:
        result = app.respond("问", source="test")
        assert result.status == "completed"
        assert result.reply == "好"
        assert app.is_running is False
        assert app.wait_for_idle(0) is True
    finally:
        app.close()


def _break_tracer_on(app, event_type: str):
    """让 tracer.write 在指定事件类型上抛 OSError（模拟磁盘故障）。"""
    original_write = app.tracer.write

    def broken_write(event):
        if event.type == event_type:
            raise OSError("disk full")
        return original_write(event)

    app.tracer.write = broken_write  # 实例遮蔽
    return original_write


def test_startup_trace_failure_still_finishes_run(tmp_path):
    """首个启动动作 emit("trace.started") 的 tracer.write 抛错：运行已
    begin，finish 必须在 finally 里无条件执行——wait_for_idle(0) 立即
    为真，agent 不 wedge busy，且原始异常如实抛出。"""
    app, client = _app(tmp_path, ModelResponse(text="恢复后答复", usage=Usage(1, 1)))
    try:
        original_write = _break_tracer_on(app, "trace.started")
        with pytest.raises(OSError, match="disk full"):
            app.respond("问", source="test")
        assert app.wait_for_idle(0) is True
        assert app.is_running is False

        # 没有被卡死：tracer 恢复后下一轮照常运行。
        app.tracer.write = original_write
        result = app.respond("问", source="test")
        assert result.status == "completed"
        assert client.calls  # 模型真的被调用了
    finally:
        app.close()


def test_startup_failure_with_prebegun_run_releases_it(tmp_path):
    """调用方先 begin_run（RPC 形状）：启动动作最后一个 emit
    ("trace.accepted") 失败，同样必须释放这个已被接受的 run。"""
    app, _client = _app(tmp_path)
    try:
        active = app.begin_run()
        assert app.is_running is True
        _break_tracer_on(app, "trace.accepted")
        with pytest.raises(OSError, match="disk full"):
            app.respond("问", source="test", active_run=active)
        assert app.wait_for_idle(0) is True
        assert app.is_running is False
    finally:
        app.close()
