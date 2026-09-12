"""统一忙碌守卫：同一规则从任意控制入口生效（阶段 1 · Batch 3，验收 d）。

共享产品控制入口（Harness.new_session / switch_session / set_thinking /
cycle_thinking / switch_model / compact）在运行活跃时一律 RunBusyError，
不再只有 RPC 一个前端设防。
"""

from __future__ import annotations

import threading

import pytest

from lsm_harness.ai.api.common import snapshot
from lsm_harness.ai.types import AssistantMessageEvent, Model, ModelResponse, Usage
from lsm_harness.coding_agent.app import Harness, RunBusyError
from lsm_harness.config import Settings
from lsm_harness.db import connect

from helpers import QueueClient


def _app(tmp_path, *responses, stream_fn=None):
    settings = Settings(api_key="scripted", home=tmp_path)
    conn = connect(tmp_path, check_same_thread=False)
    client = QueueClient(*responses)
    # reasoning=True:七档 thinking 的 clamp/cycle 需要模型支持推理
    client.model = Model(
        id="fake-main", api="legacy-client", provider="injected", reasoning=True
    )
    return Harness(
        settings=settings,
        client=client,
        conn=conn,
        stream_fn=stream_fn or client.as_stream_fn(),
    )


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


CONTROL_CALLS = [
    ("new_session", lambda app: app.new_session()),
    ("switch_session", lambda app: app.switch_session("x")),
    ("set_thinking", lambda app: app.set_thinking("enabled")),
    ("cycle_thinking", lambda app: app.cycle_thinking()),
    ("set_model", lambda app: app.switch_model("deepseek")),
    ("compact", lambda app: app.compact()),
]


@pytest.mark.parametrize(
    "action,call", CONTROL_CALLS, ids=[name for name, _ in CONTROL_CALLS]
)
def test_control_entries_rejected_mid_run(tmp_path, action, call):
    """运行活跃期间，六个状态变更入口全部拒绝且状态不变。"""
    entered, release = threading.Event(), threading.Event()
    app = _app(tmp_path, stream_fn=_parked_stream(entered, release))
    worker = threading.Thread(
        target=lambda: app.respond("跑着", source="test"), daemon=True
    )
    worker.start()
    try:
        assert entered.wait(5)
        before_session = app.session.session_id
        before_thinking = app.settings.thinking
        before_provider = app.settings.provider

        with pytest.raises(RunBusyError, match="while a run is active"):
            call(app)

        assert app.session.session_id == before_session
        assert app.settings.thinking == before_thinking
        assert app.settings.provider == before_provider
    finally:
        release.set()
        worker.join(5)
        app.close()


def test_control_entries_succeed_when_idle(tmp_path):
    """空闲时同一批入口正常工作——守卫只拦运行中，不改变正常语义。"""
    app = _app(tmp_path, ModelResponse(text="好", usage=Usage(2, 2)))
    try:
        original = app.session.session_id

        new_id = app.new_session()
        assert new_id != original
        assert app.session.session_id == new_id

        assert app.switch_session(original) == original
        assert app.session.session_id == original

        app.set_thinking("enabled")  # 旧三档入参归一为 high
        assert app.settings.thinking == "high"

        level = app.cycle_thinking()
        assert level == "off"  # high 已是支持档最后一档,回绕到 off
        assert app.settings.thinking == "off"

        assert app.switch_session("不存在") is None
        assert app.compact() is False  # 空会话无可压缩
    finally:
        app.close()


def test_set_thinking_collapses_settings_change_and_jsonl_record(tmp_path):
    """一处调用同时完成 settings 更新与 JSONL thinking_level_change 记录——
    取代三个前端各自重复的两行。"""
    app = _app(tmp_path, ModelResponse(text="好", usage=Usage(2, 2)))
    try:
        # 先跑一轮：JSONL 树的首写是延迟的，首轮之后树才存在。
        app.respond("hi", source="test")
        app.set_thinking("enabled")  # 旧三档入参归一为 high
        assert app.settings.thinking == "high"
        tree = app.session.build_session_context()
        assert tree is not None
        assert tree.thinking_level == "high"  # 旧三档入参归一后落树
    finally:
        app.close()


def test_set_thinking_rejects_unknown_level(tmp_path):
    app = _app(tmp_path)
    try:
        with pytest.raises(ValueError, match="thinking"):
            app.set_thinking("bogus")
        assert app.settings.thinking != "bogus"
    finally:
        app.close()
