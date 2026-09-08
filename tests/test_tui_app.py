"""阶段 5 批 1:真实 TUI 的 Pilot 冒烟(textual 8.x,假模型,无付费 API)。

驱动真实 `LSMTui`(注入带 QueueClient 假模型的 Harness),覆盖:
- 提交问题 → worker 线程跑 loop → UI 线程经 call_from_thread 落状态;
- 流式文本/工具行出现在 transcript(state 为事实来源);
- 状态栏同步模型/会话/tokens;
- 退出路径 `shutdown()` 中断在跑运行并关闭 harness(幂等)。

测试不依赖 pytest 异步插件:`asyncio.run()` 包裹 `run_test()`。
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from lsm_harness.ai.api.common import snapshot
from lsm_harness.ai.registry import (
    ApiProvider,
    register_api_provider,
    unregister_api_provider,
)
from lsm_harness.ai.types import (
    AssistantMessageEvent,
    Model,
    ModelResponse,
    ToolCall,
    Usage,
)
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.gateway.tui import LSMTui
from lsm_harness.gateway.tui.widgets import QueueBar

from textual.widgets import Input

from helpers import QueueClient, client_stream_fn

_API = "test-tui-api"


def _fake_get_client(provider_name, model="", small_model="", **_kw):
    client = SimpleNamespace()
    client.model = Model(id=model or "fallback", api=_API, provider=provider_name)
    return client


def _make_app(tmp_path, monkeypatch, model_id="tui-main"):
    """注入式构造:LSMTui(harness=...),harness 用假模型 + 临时 home。"""
    monkeypatch.delenv("LSM_API_KEY", raising=False)
    monkeypatch.delenv("WAKU_API_KEY", raising=False)
    client = QueueClient()
    register_api_provider(
        ApiProvider(_API, lambda m, c, o: client_stream_fn(client)(m, c, o))
    )
    monkeypatch.setattr("lsm_harness.ai.providers.get_client", _fake_get_client)
    settings = Settings(
        api_key="k",
        provider="deepseek",
        model=model_id,
        small_model="tui-small",
        home=tmp_path,
    )
    client.model = Model(id=model_id, api=_API, provider="deepseek")
    harness = Harness(
        settings=settings,
        client=client,
        conn=connect(tmp_path, check_same_thread=False),
        stream_fn=client.as_stream_fn(),
    )
    return LSMTui(harness=harness), harness, client


async def _submit_and_wait(app, pilot, text):
    from textual.widgets import Input

    input_box = app.query_one("#input", Input)
    input_box.focus()
    input_box.value = text
    await pilot.pause()
    await pilot.press("enter")
    for _ in range(400):
        await pilot.pause(0.01)
        if not app.state.running:
            break
    assert not app.state.running, "worker 未在预期时间内结束"


def test_tui_turn_with_fake_model(tmp_path, monkeypatch):
    app, harness, client = _make_app(tmp_path, monkeypatch)
    client.responses.append(
        ModelResponse(
            tool_calls=[ToolCall("c1", "list_dir", {})],
            stop_reason="tool_calls",
            usage=Usage(2, 2),
        )
    )
    client.responses.append(ModelResponse(text="TUI 答复", usage=Usage(3, 4)))

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _submit_and_wait(app, pilot, "列个目录")

            # state 是事实来源:用户行、工具行、assistant 答复齐全
            roles = [i.role for i in app.state.items]
            assert "you" in roles and "tool" in roles and "assistant" in roles
            assert any(
                i.role == "assistant" and "TUI 答复" in i.text
                for i in app.state.items
            )
            tool = next(i for i in app.state.items if i.role == "tool")
            # 产品 renderer 的 label(中文)证明 renderer 接线生效
            assert tool.tool_status == "ok" and tool.tool_label

            # 状态栏:模型/会话 + 最后一次调用的 tokens
            from lsm_harness.gateway.tui.widgets import StatusBar
            status = app.query_one("#status", StatusBar)
            assert status.model == "tui-main"
            assert status.session == harness.session.session_id[:8]
            assert status.tokens == "↑3 ↓4"

            # 输入框恢复可用
            from textual.widgets import Input
            assert not app.query_one("#input", Input).disabled

        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)

    # shutdown:中断检查 + 关闭(conn 关闭后再用即抛)
    assert app.harness is None
    with pytest.raises(sqlite3.ProgrammingError):
        harness.conn.execute("SELECT 1")


def test_tui_shutdown_without_run_and_idempotent(tmp_path, monkeypatch):
    """未跑任何一轮直接退出:shutdown 安全且幂等。"""
    app, harness, _client = _make_app(tmp_path, monkeypatch)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert not app.state.running
        app.shutdown()
        app.shutdown()  # 幂等

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)
    assert app.harness is None


# ── 批 2:中断 vs 退出 + 运行中 steering/follow-up ──────────


def test_start_run_begins_on_ui_thread(tmp_path, monkeypatch):
    """提交瞬间 harness.is_running 即为真:begin_run 在 UI 线程执行
    (RPC _start_worker 同款),不等 worker 到达 respond——运行中的
    Escape/steer 从这一刻起就有效。worker 拿到的 respond 调用必须带
    active_run(否则它会在 worker 线程里才 begin,窗口依旧存在)。"""
    app, harness, client = _make_app(tmp_path, monkeypatch)
    respond_entered = threading.Event()
    release = threading.Event()
    seen_active_runs = []
    original_respond = harness.respond

    def gated_respond(*args, **kwargs):
        seen_active_runs.append(kwargs.get("active_run"))
        respond_entered.set()
        release.wait(5)  # 把 worker 按在 respond 入口
        return original_respond(*args, **kwargs)

    harness.respond = gated_respond
    client.responses.append(ModelResponse(text="答", usage=Usage(1, 1)))

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "任务"
            await pilot.press("enter")
            await _wait_for(pilot, respond_entered.is_set)
            # worker 还按在 respond 入口,但 run 早已被 UI 线程接受
            assert harness.is_running is True
            release.set()
            await _wait_for(pilot, lambda: not app.state.running)
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)
    assert seen_active_runs and seen_active_runs[0] is not None


def _parked_stream(entered: threading.Event):
    """停在流里直到被中断,然后以 aborted 收尾(同 test_continue)。"""

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


async def _wait_for(pilot, predicate, timeout=5.0):
    """轮询等待条件(pilot.pause 步进,让 call_from_thread 有机会执行)。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.01)
        if predicate():
            return
    raise AssertionError("等待超时")


def test_escape_aborts_run_but_keeps_app(tmp_path, monkeypatch):
    """Escape 中断当前轮但不退出应用;transcript 留下中断标记。"""
    app, harness, _client = _make_app(tmp_path, monkeypatch)
    entered = threading.Event()
    harness.stream_fn = _parked_stream(entered)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "任务"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: app.state.running and entered.is_set())

            await pilot.press("escape")
            await _wait_for(pilot, lambda: not app.state.running)

            assert any(
                i.role == "note" and "Interrupted" in i.text
                for i in app.state.items
            )
            # 应用仍在运行:输入框还在,可继续提交
            assert app.is_running
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)
    assert app.harness is None


def test_enter_during_run_steers_and_shows_queue(tmp_path, monkeypatch):
    """运行中 Enter = steer:进 harness steering 队列,QueueBar 显示。"""
    app, harness, _client = _make_app(tmp_path, monkeypatch)
    entered = threading.Event()
    harness.stream_fn = _parked_stream(entered)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "任务"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: app.state.running and entered.is_set())

            # 运行中输入框保持可用(批 2 的核心行为)
            assert not input_box.disabled
            input_box.value = "顺便改 X"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: bool(app.state.queued_steering))

            assert app.state.queued_steering == ["顺便改 X"]
            assert harness.pending_messages()["steering"] == ["顺便改 X"]
            queue = app.query_one("#queue", QueueBar)
            assert queue.steering == ("顺便改 X",)

            harness.abort()
            await _wait_for(pilot, lambda: not app.state.running)
            # 收尾后队列显示清空
            assert app.state.queued_steering == []
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_follow_command_queues_follow_up(tmp_path, monkeypatch):
    """/follow <文本> 在运行中排队 follow-up,进 harness follow_up 队列。"""
    app, harness, _client = _make_app(tmp_path, monkeypatch)
    entered = threading.Event()
    harness.stream_fn = _parked_stream(entered)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "任务"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: app.state.running and entered.is_set())

            input_box.value = "/follow 之后做这个"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: bool(app.state.queued_follow_ups))

            assert app.state.queued_follow_ups == ["之后做这个"]
            assert harness.pending_messages()["follow_up"] == ["之后做这个"]
            queue = app.query_one("#queue", QueueBar)
            assert queue.follow_ups == ("之后做这个",)

            harness.abort()
            await _wait_for(pilot, lambda: not app.state.running)
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_ctrl_c_quits_and_releases_resources(tmp_path, monkeypatch):
    """Ctrl+C 在运行中直接退出;shutdown 中断在跑的运行并关闭 harness。"""
    app, harness, _client = _make_app(tmp_path, monkeypatch)
    entered = threading.Event()
    harness.stream_fn = _parked_stream(entered)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "任务"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: app.state.running and entered.is_set())

            await pilot.press("ctrl+c")  # 退出,不是中断
            await _wait_for(pilot, lambda: not app.is_running)
        # run_test 退出后 worker 仍 parked;shutdown 负责 abort + close
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)

    assert app.harness is None
    with pytest.raises(sqlite3.ProgrammingError):
        harness.conn.execute("SELECT 1")


# ── 批 3:模型 / 会话 / 树节点选择器 ──────────────────────


def test_model_picker_switches_model(tmp_path, monkeypatch):
    """/model 打开选择器,选中即切换(走 harness.switch_model)。"""
    from lsm_harness.gateway.tui.app import PickerScreen

    app, harness, _client = _make_app(tmp_path, monkeypatch)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "/model"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))

            # Escape 取消:不改变模型
            await pilot.press("escape")
            await _wait_for(pilot, lambda: not isinstance(app.screen, PickerScreen))
            assert harness.settings.model == "tui-main"

            # 重新打开,选第二个 provider(openai → gpt-4o)
            input_box.value = "/model"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))
            await pilot.press("down")
            await pilot.press("enter")
            await _wait_for(pilot, lambda: harness.settings.model == "gpt-4o")
            assert not isinstance(app.screen, PickerScreen)

            from lsm_harness.gateway.tui.widgets import StatusBar
            assert app.query_one("#status", StatusBar).model == "gpt-4o"
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_session_picker_switches_session(tmp_path, monkeypatch):
    """/sessions 选中旧会话即切换;历史随会话恢复。"""
    from lsm_harness.gateway.tui.app import PickerScreen

    app, harness, client = _make_app(tmp_path, monkeypatch)
    client.responses.append(ModelResponse(text="答一", usage=Usage(2, 2)))

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _submit_and_wait(app, pilot, "问题一")
            sid_a = harness.session.session_id

            harness.new_session()
            assert harness.session.session_id != sid_a

            rows = harness.session.list_sessions(15)
            target = next(i for i, r in enumerate(rows) if r["id"] == sid_a)

            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "/sessions"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))
            for _ in range(target):
                await pilot.press("down")
            await pilot.press("enter")

            await _wait_for(pilot, lambda: harness.session.session_id == sid_a)
            users = [m["content"] for m in harness.session.history if m["role"] == "user"]
            assert users == ["问题一"]
            from lsm_harness.gateway.tui.widgets import StatusBar
            assert app.query_one("#status", StatusBar).session == sid_a[:8]
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_tree_picker_branches_to_node(tmp_path, monkeypatch):
    """/tree 列出当前路径条目,选中节点即 branch(leaf 移动)。"""
    from lsm_harness.agent.messages import message_preview
    from lsm_harness.gateway.tui.app import PickerScreen
    from lsm_harness.ops.session_store import read_session_entries

    app, harness, client = _make_app(tmp_path, monkeypatch)
    client.responses.append(ModelResponse(text="答一", usage=Usage(2, 2)))
    client.responses.append(ModelResponse(text="答二", usage=Usage(2, 2)))

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _submit_and_wait(app, pilot, "任务一")
            await _submit_and_wait(app, pilot, "任务二")

            entries = read_session_entries(harness.session.jsonl_path)
            u1 = next(
                e for e in entries
                if e.type == "message"
                and message_preview(e.message, limit=1_000_000) == "任务一"
            )

            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "/tree"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))

            # 找到 u1 在选项里的位置并选中
            options = app._tree_options()
            target = next(i for i, (_label, value) in enumerate(options) if value == u1.id)
            for _ in range(target):
                await pilot.press("down")
            await pilot.press("enter")

            await _wait_for(
                pilot,
                lambda: harness.session.recorder.last_entry_id == u1.id,
            )
            users = [m["content"] for m in harness.session.history if m["role"] == "user"]
            assert users == ["任务一"]  # 任务二所在侧被弃
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_tree_picker_blocked_while_running(tmp_path, monkeypatch):
    """运行中 /tree 不打开选择器(Tau 的 TREE_RUNNING_MESSAGE 对应)。"""
    from lsm_harness.gateway.tui.app import PickerScreen

    app, harness, _client = _make_app(tmp_path, monkeypatch)
    entered = threading.Event()
    harness.stream_fn = _parked_stream(entered)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "任务"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: app.state.running and entered.is_set())

            input_box.value = "/tree"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, PickerScreen)
            assert any(
                "运行中不可切换节点" in i.text
                for i in app.state.items if i.role == "note"
            )

            harness.abort()
            await _wait_for(pilot, lambda: not app.state.running)
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


# ── 批 4:工具折叠(Ctrl+O 全局重渲染) ─────────────────────


def test_ctrl_o_folds_tool_results_globally(tmp_path, monkeypatch):
    """Ctrl+O 后 RichLog 清屏重放:历史工具结果行也被折叠,再按复原。"""
    from textual.widgets import RichLog

    app, harness, client = _make_app(tmp_path, monkeypatch)
    client.responses.append(
        ModelResponse(
            tool_calls=[ToolCall("c1", "list_dir", {})],
            stop_reason="tool_calls",
            usage=Usage(2, 2),
        )
    )
    client.responses.append(ModelResponse(text="完成", usage=Usage(2, 2)))

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _submit_and_wait(app, pilot, "列目录")

            chat = app.query_one("#chat", RichLog)
            expanded_count = len(chat.lines)
            assert app.state.show_tool_results

            await pilot.press("ctrl+o")
            await pilot.pause(0.1)
            folded_count = len(chat.lines)
            assert folded_count < expanded_count  # 工具结果行被折叠掉
            assert not app.state.show_tool_results
            # 折叠后 transcript == state 全量渲染 + 提示行
            assert len(chat.lines) == len(app.state.render_lines())

            await pilot.press("ctrl+o")
            await pilot.pause(0.1)
            assert len(chat.lines) > folded_count  # 展开复原
            assert app.state.show_tool_results
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)
