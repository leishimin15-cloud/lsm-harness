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
from lsm_harness.gateway.tui.state import MessageView, ToolView
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

    transcript_size = len(app.state.transcript)
    input_box = app.query_one("#input", Input)
    input_box.focus()
    input_box.value = text
    await pilot.pause()
    await pilot.press("enter")
    for _ in range(400):
        await pilot.pause(0.01)
        # ``pilot.press`` queues the submit event.  On a slow event loop the
        # worker may not have started yet, so idle alone is not proof that
        # this submission finished.  Wait until the event was consumed
        # (the user row exists) and the accepted run has settled.
        if (
            len(app.state.transcript) > transcript_size
            and not app.state.running
        ):
            break
    assert len(app.state.transcript) > transcript_size, "提交事件未被处理"
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
            # 启动后可打印按键必须直接进入编辑器，不能让
            # Transcript/Header 抢走焦点导致“看得见输入框但打不进字”。
            assert app.query_one("#input", Input).has_focus
            await _submit_and_wait(app, pilot, "列个目录")

            # state 是事实来源:用户行、工具行、assistant 答复齐全
            def _role(item):
                return item.role if isinstance(item, MessageView) else "tool"

            roles = [_role(i) for i in app.state.transcript]
            assert "user" in roles and "tool" in roles and "assistant" in roles
            assert any(
                isinstance(i, MessageView)
                and i.role == "assistant"
                and "TUI 答复" in i.text
                for i in app.state.transcript
            )
            tool = next(
                i for i in app.state.transcript if isinstance(i, ToolView)
            )
            # 产品 renderer 的 label(中文)证明 renderer 接线生效
            assert tool.status == "ok" and tool.label

            # 单行 footer:模型/提供方在右侧,累计用量在左侧
            # (两次调用 2+3 / 2+4)
            from lsm_harness.gateway.tui.widgets import StatusBar
            status = app.query_one("#status", StatusBar)
            assert "tui-main" in status.right
            assert "(deepseek)" in status.right
            assert "↑5" in status.stats and "↓6" in status.stats

            # 输入框恢复可用
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


def test_possible_api_key_is_blocked_from_chat(tmp_path, monkeypatch):
    """A likely credential never reaches transcript history or the model."""
    app, harness, client = _make_app(tmp_path, monkeypatch)
    possible_key = "sk-kimi-example-key-that-must-not-leak"

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            input_box = app.query_one("#input", Input)
            assert input_box.placeholder == ""
            input_box.focus()
            input_box.value = possible_key
            await pilot.press("enter")
            await pilot.pause()

            assert not app.state.running
            assert not client.responses
            assert all(
                possible_key not in item.text
                for item in app.state.transcript
            )
            assert any(
                isinstance(item, MessageView)
                and item.role == "note"
                and "Blocked a possible API key" in item.text
                for item in app.state.transcript
            )
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


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
                isinstance(i, MessageView)
                and i.role == "note"
                and "Interrupted" in i.text
                for i in app.state.transcript
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
            # 收尾是两步:settled 事件落 is_running=False,_finish_run
            # (worker finally 的 UI 任务)清队列显示——两个都等到。
            await _wait_for(
                pilot,
                lambda: not app.state.running
                and app.state.queued_steering == [],
            )
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
    from lsm_harness.gateway.tui.screens import PickerScreen

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
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

            # 重新打开,从 Pi 同源完整目录中选 openai/gpt-4o。
            input_box.value = "/model"
            await pilot.press("enter")
            await _wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))
            for _ in range(8):
                await pilot.press("down")
            await pilot.press("enter")
            await _wait_for(pilot, lambda: harness.settings.model == "gpt-4o")
            assert not isinstance(app.screen, PickerScreen)

            from lsm_harness.gateway.tui.widgets import StatusBar
            await _wait_for(
                pilot,
                lambda: "gpt-4o" in app.query_one("#status", StatusBar).right,
            )
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_session_picker_switches_session(tmp_path, monkeypatch):
    """/sessions 选中旧会话即切换;历史随会话恢复。"""
    from lsm_harness.gateway.tui.screens import ResumeSessionScreen

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
            await _wait_for(
                pilot, lambda: isinstance(app.screen, ResumeSessionScreen)
            )
            for _ in range(target):
                await pilot.press("down")
            await pilot.press("enter")

            await _wait_for(pilot, lambda: harness.session.session_id == sid_a)
            users = [m["content"] for m in harness.session.history if m["role"] == "user"]
            assert users == ["问题一"]
            # 切回旧会话后,累计用量从历史现算(不依赖事件重放)
            from lsm_harness.gateway.tui.widgets import StatusBar
            stats = app.query_one("#status", StatusBar).stats
            assert "↑2" in stats and "↓2" in stats
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_tree_picker_branches_to_node(tmp_path, monkeypatch):
    """/tree 列出当前路径条目,选中节点即 branch(leaf 移动)。"""
    from lsm_harness.agent.messages import message_preview
    from lsm_harness.gateway.tui.screens import PickerScreen
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
    from lsm_harness.gateway.tui.screens import PickerScreen

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
                isinstance(i, MessageView)
                and i.role == "note"
                and "运行中不可切换节点" in i.text
                for i in app.state.transcript
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
    """Ctrl+O 折叠/展开:全局开关同步进 ToolCallWidget,原位重渲染。"""
    from lsm_harness.gateway.tui.widgets import ToolCallWidget

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

            tools = app.query(ToolCallWidget)
            assert len(tools) == 1
            tool = tools[0]
            assert app.state.show_tool_results
            assert tool.show_tool_results
            assert tool.result_line in tool.render()  # 结果行可见

            await pilot.press("ctrl+o")
            await pilot.pause(0.1)
            assert not app.state.show_tool_results
            assert not tool.show_tool_results
            folded = tool.render()
            assert tool.result_line not in folded  # 结果行被折叠
            assert tool.call_line in folded  # 调用行仍在

            await pilot.press("ctrl+o")
            await pilot.pause(0.1)
            assert app.state.show_tool_results
            assert tool.result_line in tool.render()  # 展开复原
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


# ── 批 5:真流式 / 单项折叠 / 状态栏完整显示(阶段二 ④⑤⑥) ──────


def test_streaming_updates_assistant_widget_in_place(tmp_path, monkeypatch):
    """assistant 真流式:AssistantWidget 随 text_delta 原位重渲染(批 ④)。"""
    from lsm_harness.gateway.tui.widgets import AssistantWidget

    app, harness, _client = _make_app(tmp_path, monkeypatch)
    first_seen = threading.Event()
    release = threading.Event()

    def gated_stream(_model, _context, _options):
        yield AssistantMessageEvent(
            "text_delta",
            snapshot(text="你", thinking="", pending={}),
            text_delta="你",
        )
        first_seen.set()
        release.wait(5)
        yield AssistantMessageEvent(
            "text_delta",
            snapshot(text="你好 **世界**", thinking="", pending={}),
            text_delta="好 **世界**",
        )
        yield AssistantMessageEvent(
            "done",
            snapshot(
                text="你好 **世界**", thinking="", pending={},
                stop_reason="stop", usage=Usage(2, 2),
            ),
        )

    harness.stream_fn = gated_stream

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "打个招呼"
            await pilot.press("enter")
            await _wait_for(pilot, first_seen.is_set)
            await _wait_for(
                pilot,
                lambda: any(
                    isinstance(w, AssistantWidget) and w.is_streaming
                    for w in app._entry_widgets
                ),
            )
            assistant = next(
                w for w in app._entry_widgets
                if isinstance(w, AssistantWidget) and w.is_streaming
            )
            # 流式中间态:只有第一个 token 已到,正文未落定
            assert assistant.is_streaming
            assert assistant.text == "你"
            release.set()
            await _wait_for(pilot, lambda: not app.state.running)
            assert assistant.text == "你好 **世界**"
            assert not assistant.is_streaming
            # markdown 落定后重解析(Markdown.update 异步挂载块,先等一拍)
            await _wait_for(
                pilot,
                lambda: assistant._markdown._markdown == "你好 **世界**"
                and len(assistant._markdown.children) > 0,
            )
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_tool_widget_click_folds_single_item(tmp_path, monkeypatch):
    """单击工具条目原位折叠该项;不影响全局开关(批 ⑤)。"""
    from lsm_harness.gateway.tui.widgets import ToolCallWidget

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

            tool = app.query_one(ToolCallWidget)
            assert not tool.collapsed
            assert tool.result_line in tool.render()

            await pilot.click(ToolCallWidget)
            await pilot.pause(0.1)
            assert tool.collapsed
            folded = tool.render()
            assert tool.result_line not in folded
            assert tool.call_line in folded
            # 单项折叠不改变全局开关
            assert app.state.show_tool_results
            assert tool.show_tool_results

            await pilot.click(ToolCallWidget)
            await pilot.pause(0.1)
            assert not tool.collapsed
            assert tool.result_line in tool.render()
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_status_bar_snapshot_and_transient(tmp_path, monkeypatch):
    """状态栏:快照段(身份/累计用量,footer_snapshot 现算)+ 瞬态段
    (retry/compaction,来自事件归约的 state)。

    归约逻辑(typed events → state)已由 test_tui_state 覆盖,这里钉
    ``_refresh_status`` 把快照与 state 同步进 StatusBar 的接线。
    """
    from lsm_harness.gateway.tui.widgets import StatusBar

    app, harness, client = _make_app(tmp_path, monkeypatch)
    client.responses.append(ModelResponse(text="答", usage=Usage(1000, 500)))

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _submit_and_wait(app, pilot, "问题")

            status = app.query_one("#status", StatusBar)
            # 快照段:累计用量从历史现算;身份/环境齐全
            assert "↑1.0k" in status.stats and "↓500" in status.stats
            assert "tui-main" in status.right
            assert "(deepseek)" in status.right

            # 瞬态段:retry/compaction
            app.state.is_retrying = True
            app.state.retry_attempt = 2
            app.state.retry_max = 3
            app.state.is_compacting = True
            app._refresh_status()

            assert status.retry == "2/3"
            assert status.compacting is True
            rendered = status.render()
            assert "retry 2/3" in rendered
            assert "compacting" in rendered
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


# ── 批 6:窗口化 / 补全(阶段三) ─────────────────────────


def test_transcript_windowing_hides_oldest_entries(tmp_path, monkeypatch):
    """窗口化:挂载条目数有上限,更旧的条目聚合进顶部边界占位。"""
    from lsm_harness.gateway.tui.widgets import TranscriptView

    app, harness, client = _make_app(tmp_path, monkeypatch)
    for _ in range(3):
        client.responses.append(ModelResponse(text="答", usage=Usage(1, 1)))

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", TranscriptView)
            chat.max_entries = 4  # 缩小窗口便于测试
            await _submit_and_wait(app, pilot, "问题一")
            await _submit_and_wait(app, pilot, "问题二")
            await _submit_and_wait(app, pilot, "问题三")

            # transcript 条目:欢迎 note + 每轮 (user + assistant) = 1 + 6 = 7
            assert len(app.state.transcript) == 7
            # DOM 只挂载窗口内 4 条 + 顶部边界占位
            assert chat.hidden_count == 7 - 4
            boundary = chat.query_one(".transcript-boundary")
            assert "已隐藏 3 条" in str(boundary.render())
            mounted = len(chat.children) - 1  # 减去边界
            assert mounted == 4
            # 关键:App 的同步列表也只保留窗口段(不只是 DOM 截断)——
            # 每次流式事件的同步成本是 O(窗口),不随历史增长
            assert len(app._entry_widgets) == 4
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_at_file_autocomplete_tab_accepts(tmp_path, monkeypatch):
    """@ 文件补全:输入 @ 前缀弹出 workspace 文件,Tab 接受高亮项。"""
    from lsm_harness.gateway.tui.autocomplete import SuggestionOverlay

    app, harness, _client = _make_app(tmp_path, monkeypatch)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one("#suggestions", SuggestionOverlay)
            assert not overlay.display

            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "@pyproj"  # 子序列匹配 pyproject.toml
            # 文件索引后台构建 + 20ms debounce:等结果而不是固定暂停
            await _wait_for(pilot, lambda: overlay.display)
            labels = [str(o.prompt) for o in overlay.options]
            assert any("pyproject.toml" in label for label in labels)

            await pilot.press("tab")
            await pilot.pause(0.1)
            assert input_box.value == "@pyproject.toml"
            assert not overlay.display  # 接受后不再弹出(_just_accepted)
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_slash_command_autocomplete_and_escape(tmp_path, monkeypatch):
    """/ 命令补全:Tab 接受,Esc 只关弹层不中断运行。"""
    from lsm_harness.gateway.tui.autocomplete import SuggestionOverlay

    app, harness, _client = _make_app(tmp_path, monkeypatch)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one("#suggestions", SuggestionOverlay)
            input_box = app.query_one("#input", Input)
            input_box.focus()

            input_box.value = "/sum"
            await pilot.pause(0.2)
            assert overlay.display
            labels = [str(o.prompt) for o in overlay.options]
            assert any("/summary" in label for label in labels)

            await pilot.press("tab")
            await pilot.pause(0.1)
            assert input_box.value == "/summary"

            # 再触发一次补全,Esc 关闭
            input_box.value = "/new"
            await pilot.pause(0.2)
            assert overlay.display
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert not overlay.display
            assert not app.state.running  # Esc 被弹层消费,没有触发 abort
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_slash_command_enter_accepts_and_submits_like_pi(tmp_path, monkeypatch):
    """Enter on a slash suggestion accepts it and submits immediately."""
    from lsm_harness.gateway.tui.autocomplete import SuggestionOverlay

    app, _harness, _client = _make_app(tmp_path, monkeypatch)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one("#suggestions", SuggestionOverlay)
            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "/"
            await pilot.pause(0.2)
            assert overlay.display

            await pilot.press("enter")
            await pilot.pause(0.2)

            assert input_box.value == ""
            transcript = " ".join(item.text for item in app.state.transcript)
            assert "Commands:" in transcript
            assert "Unknown: /" not in transcript
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_rebuild_applies_window_to_long_history(tmp_path, monkeypatch):
    """rebuild(启动/切会话)对超长历史直接跳过隐藏段,不挂载。"""
    from lsm_harness.gateway.tui.widgets import TranscriptView

    app, harness, client = _make_app(tmp_path, monkeypatch)
    for i in range(6):
        client.responses.append(ModelResponse(text=f"答{i}", usage=Usage(1, 1)))

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", TranscriptView)
            chat.max_entries = 3
            for i in range(6):
                await _submit_and_wait(app, pilot, f"问题{i}")

            # 1 欢迎 + 6×2 = 13 条;窗口 3
            assert len(app.state.transcript) == 13
            assert len(app._entry_widgets) == 3
            assert chat.hidden_count == 10

            # rebuild(切会话/重启路径)同样只挂窗口
            # (rebuild 从会话树重建:欢迎语 note 不是会话消息,12 条)
            app._rebuild_from_session()
            await pilot.pause(0.1)
            assert len(app.state.transcript) == 12
            assert len(app._entry_widgets) == 3
            assert chat.hidden_count == 9
            mounted = len(chat.children) - 1
            assert mounted == 3
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_scroll_to_top_loads_earlier_page(tmp_path, monkeypatch):
    """反向加载:滚到顶部把更早一页挂回 DOM 与同步列表(批 4)。"""
    from lsm_harness.gateway.tui.widgets import TranscriptView

    app, harness, client = _make_app(tmp_path, monkeypatch)
    for _ in range(4):
        client.responses.append(ModelResponse(text="答", usage=Usage(1, 1)))

    async def main():
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", TranscriptView)
            chat.max_entries = 3
            for i in range(4):
                await _submit_and_wait(app, pilot, f"问题{i}")

            # 9 条(欢迎 note + 4×2),窗口 3 → 隐藏 6
            assert len(app.state.transcript) == 9
            assert chat.hidden_count == 6
            assert len(app._entry_widgets) == 3
            assert "已隐藏 6 条" in str(
                chat.query_one(".transcript-boundary").render()
            )

            # 模拟用户滚到顶:触发反向加载(页长 50 > 6,全部载回)
            chat.watch_scroll_y(10, 0)
            await pilot.pause(0.3)

            assert chat.hidden_count == 0
            assert len(app._entry_widgets) == 9
            # 边界占位撤掉,最早条目回到首位
            assert not chat.query(".transcript-boundary")
            assert app._entry_widgets[0].sync_from is not None
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def _wheel(chat, event_cls):
    """向聊天区投递真实滚轮事件。

    textual 8.x 的 Pilot 没有 scroll 辅助方法;直接 post_message 给
    TranscriptView,走 ScrollableContainer 的 _on_mouse_scroll_* 私有
    handler——与真实鼠标滚轮同一条事件链(scroll_target_y → scroll_y
    reactive → watch_scroll_y → super() 刷新/滚动条同步)。
    """
    chat.post_message(
        event_cls(
            chat, x=1, y=1, delta_x=0, delta_y=0, button=0,
            shift=False, meta=False, ctrl=False,
        )
    )


def test_mouse_wheel_scrolls_and_syncs_scrollbar(tmp_path, monkeypatch):
    """真实滚轮事件驱动滚动:scroll_y 变化、滚动条位置同步、follow 状态
    随滚动方向切换。

    回归:watch_scroll_y 必须保留 super() 的刷新/滚动条同步——旧测试
    直接调 watcher,覆盖不到「事件 → reactive → watcher → 父类重绘」
    这条链。
    """
    from textual.events import MouseScrollDown, MouseScrollUp

    from lsm_harness.gateway.tui.widgets import TranscriptView

    app, harness, client = _make_app(tmp_path, monkeypatch)
    long_answer = "\n".join(f"行{i}" for i in range(12))
    for _ in range(3):
        client.responses.append(
            ModelResponse(text=long_answer, usage=Usage(1, 1))
        )

    async def main():
        async with app.run_test(size=(80, 16)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", TranscriptView)
            for i in range(3):
                await _submit_and_wait(app, pilot, f"问题{i}")
            await _wait_for(pilot, lambda: chat.max_scroll_y > 0)

            # follow 贴底:滚动条位置与 scroll_y 一致
            assert chat._follow
            bottom = chat.scroll_y
            assert bottom > 0
            assert chat.vertical_scrollbar.position == bottom

            # 滚轮上滚:scroll_y 真的变小,滚动条同步,follow 关闭
            _wheel(chat, MouseScrollUp)
            await pilot.pause(0.1)
            assert chat.scroll_y < bottom
            assert chat.vertical_scrollbar.position == chat.scroll_y
            assert not chat._follow

            # 滚轮下滚回底:follow 恢复
            for _ in range(40):
                if chat._follow:
                    break
                _wheel(chat, MouseScrollDown)
                await pilot.pause(0.05)
            assert chat._follow
            assert chat.scroll_y == chat.max_scroll_y
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_pageup_pagedown_scroll_with_input_focused(tmp_path, monkeypatch):
    """PageUp/PageDown 在输入框持焦点时翻聊天区(App 级绑定兜底)。

    Home/End 留给输入框光标移动;聊天区聚焦时走 ScrollableContainer
    自带的 home/end 绑定,不在 App 级重复。
    """
    from lsm_harness.gateway.tui.widgets import TranscriptView

    app, harness, client = _make_app(tmp_path, monkeypatch)
    long_answer = "\n".join(f"行{i}" for i in range(12))
    for _ in range(3):
        client.responses.append(
            ModelResponse(text=long_answer, usage=Usage(1, 1))
        )

    async def main():
        async with app.run_test(size=(80, 16)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", TranscriptView)
            for i in range(3):
                await _submit_and_wait(app, pilot, f"问题{i}")
            await _wait_for(pilot, lambda: chat.max_scroll_y > 0)

            # 焦点明确留在输入框:翻页键不被输入框消费,由 App 级绑定兜底
            app.query_one("#input", Input).focus()
            bottom = chat.scroll_y
            await pilot.press("pageup")
            await pilot.pause(0.1)
            assert chat.scroll_y < bottom
            assert not chat._follow  # 上翻自动关 follow

            for _ in range(40):
                if chat._follow:
                    break
                await pilot.press("pagedown")
                await pilot.pause(0.05)
            assert chat._follow  # 翻回底部恢复 follow
            # 翻页键没有进输入框(值仍为空)
            assert app.query_one("#input", Input).value == ""
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_mouse_wheel_to_top_loads_earlier_page(tmp_path, monkeypatch):
    """滚轮滚到顶部触发反向加载:走真实事件链,不再直接调 watcher。"""
    from textual.events import MouseScrollUp

    from lsm_harness.gateway.tui.widgets import TranscriptView

    app, harness, client = _make_app(tmp_path, monkeypatch)
    long_answer = "\n".join(f"行{i}" for i in range(10))
    for _ in range(4):
        client.responses.append(
            ModelResponse(text=long_answer, usage=Usage(1, 1))
        )

    async def main():
        async with app.run_test(size=(80, 10)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", TranscriptView)
            chat.max_entries = 3
            for i in range(4):
                await _submit_and_wait(app, pilot, f"问题{i}")
            assert chat.hidden_count == 6

            for _ in range(60):
                if chat.hidden_count == 0:
                    break
                _wheel(chat, MouseScrollUp)
                await pilot.pause(0.05)
            assert chat.hidden_count == 0
            assert len(app._entry_widgets) == 9
            assert not chat.query(".transcript-boundary")
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def _gated_stream(chunk1: str, chunk2: str, entered: threading.Event,
                  release: threading.Event):
    """流式分两段:第一段后按在门闩上,放行后补第二段并收尾。"""

    def stream(_model, _context, options):
        yield AssistantMessageEvent(
            "start", snapshot(text="", thinking="", pending={})
        )
        yield AssistantMessageEvent(
            "text_start", snapshot(text="", thinking="", pending={})
        )
        yield AssistantMessageEvent(
            "text_delta",
            snapshot(text=chunk1, thinking="", pending={}),
            text_delta=chunk1,
        )
        entered.set()
        release.wait(5)
        full = chunk1 + chunk2
        final = snapshot(
            text=full, thinking="", pending={}, usage=Usage(1, 1)
        )
        yield AssistantMessageEvent(
            "text_delta", final, text_delta=chunk2
        )
        yield AssistantMessageEvent("text_end", final)
        yield AssistantMessageEvent("done", final)

    return stream


def test_stream_does_not_yank_user_scrolled_up(tmp_path, monkeypatch):
    """follow 模式:用户上滚后,后续流式输出不把视图拽回底部;
    翻回底部后恢复 follow。"""
    from textual.events import MouseScrollUp

    from lsm_harness.gateway.tui.widgets import TranscriptView

    app, harness, _client = _make_app(tmp_path, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    # 双换行才是独立段落(单换行在 markdown 里并段,撑不出滚动高度)
    chunk1 = "\n\n".join(f"第一段行{i}" for i in range(20))
    chunk2 = "\n\n".join(f"第二段行{i}" for i in range(8))
    harness.stream_fn = _gated_stream(chunk1, chunk2, entered, release)

    async def main():
        async with app.run_test(size=(80, 16)) as pilot:
            await pilot.pause()
            chat = app.query_one("#chat", TranscriptView)
            input_box = app.query_one("#input", Input)
            input_box.focus()
            input_box.value = "任务"
            await pilot.press("enter")
            await _wait_for(pilot, entered.is_set)
            # 第一段(20 行)完整排版且 follow 贴底已生效:max 明显超出
            # 视口才说明 markdown 布局完成(scroll_end 可能延后到
            # refresh 后;不等贴底就上滚会滚不动)
            await _wait_for(
                pilot,
                lambda: chat.max_scroll_y >= 5
                and chat.scroll_y >= chat.max_scroll_y - 1,
            )

            # 用户上滚:follow 关闭
            _wheel(chat, MouseScrollUp)
            await pilot.pause(0.1)
            assert not chat._follow
            parked_y = chat.scroll_y

            # 放行第二段:内容继续追加,视口原地不动
            release.set()
            await _wait_for(pilot, lambda: not app.state.running)
            await pilot.pause(0.2)
            assert not chat._follow
            assert chat.scroll_y == parked_y
            assert chat.max_scroll_y > parked_y  # 新内容确实进来了

            # 翻回底部:follow 恢复,能看到第二段
            for _ in range(40):
                if chat._follow:
                    break
                await pilot.press("pagedown")
                await pilot.pause(0.05)
            assert chat._follow
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)



def test_startup_restore_shows_cumulative_usage(tmp_path, monkeypatch):
    """启动恢复会话:累计用量从历史 AssistantMessage.usage 现算
    (footer_snapshot),不等任何事件——修复「重启后 footer 用量全空」。"""
    from lsm_harness.gateway.tui.widgets import StatusBar

    app, harness, client = _make_app(tmp_path, monkeypatch)
    client.responses.append(ModelResponse(text="答", usage=Usage(1200, 300)))
    # 不经 TUI 直接跑一轮,把带 usage 的历史写进会话树
    harness.respond("问题", source="test", active_run=harness.begin_run())

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            status = app.query_one("#status", StatusBar)
            # on_mount → rebuild → _refresh_status:历史用量立即可见
            assert "↑1.2k" in status.stats and "↓300" in status.stats
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_ui_marshal_error_writes_debug_log(tmp_path, monkeypatch):
    """UI 编组异常不再全静默:写 <home>/logs/tui-ui.log。"""
    app, harness, _client = _make_app(tmp_path, monkeypatch)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            app._log_ui_error(RuntimeError("boom-marker"))
            log = tmp_path / "logs" / "tui-ui.log"
            assert log.exists()
            assert "boom-marker" in log.read_text(encoding="utf-8")
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


# ── 输入热路径(2026-09-12 实机卡顿修复)────────────────────────

def test_iter_workspace_files_prunes_skip_dirs(tmp_path):
    """文件索引:os.walk 直接剪枝,.git/.venv/node_modules 不进结果
    (也不会先遍历再过滤)。"""
    from pathlib import Path as P

    from lsm_harness.gateway.tui.autocomplete import _iter_workspace_files

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x", encoding="utf-8")
    for d in (".git", ".venv", "node_modules"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "junk.py").write_text("x", encoding="utf-8")
    files = _iter_workspace_files(tmp_path)
    assert files == [str(P("src") / "a.py")]


def test_plain_typing_never_touches_overlay_or_catalog(tmp_path, monkeypatch):
    """普通文本:不进补全分支、不读模型目录;/login 目录只加载一次
    (缓存命中,每键重载 1345 模型目录的回归)。"""
    from lsm_harness.gateway.tui import autocomplete
    from lsm_harness.gateway.tui.autocomplete import SuggestionOverlay

    calls = []
    real_load = autocomplete.load_model_catalog

    def counting_load(home):
        calls.append(home)
        return real_load(home)

    monkeypatch.setattr(autocomplete, "load_model_catalog", counting_load)
    app, harness, _client = _make_app(tmp_path, monkeypatch)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one("#suggestions", SuggestionOverlay)
            input_box = app.query_one("#input", Input)
            input_box.focus()

            input_box.value = "hello world"
            await pilot.pause(0.1)
            assert not overlay.display
            assert not calls  # 普通输入不碰目录

            input_box.value = "/login d"
            await pilot.pause(0.1)
            assert overlay.display
            assert len(calls) == 1  # 首次 /login 才加载

            input_box.value = "/login de"
            await pilot.pause(0.1)
            assert len(calls) == 1  # 缓存:每键不再重载

            # 登录/重激活后缓存失效,下次 /login 重读
            overlay.invalidate_providers()
            input_box.value = "/login dee"
            await pilot.pause(0.1)
            assert len(calls) == 2
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_file_suggestions_debounced_latest_wins(tmp_path, monkeypatch):
    """@ 连续键入:~20ms debounce,只应用最新 request 的结果;输入
    已离开 @ 时旧任务作废。"""
    from lsm_harness.gateway.tui.autocomplete import SuggestionOverlay

    app, harness, _client = _make_app(tmp_path, monkeypatch)

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one("#suggestions", SuggestionOverlay)
            # 直接注入索引,跳过后台构建等待
            overlay._file_cache = ["a1.py", "a2.py", "b.py"]

            overlay.update_for("@a")
            overlay.update_for("@a1")
            await pilot.pause(0.15)
            items = [str(o.prompt) for o in overlay.options]
            assert items == ["@a1.py"]  # 只应用最后一次 "@a1"

            # 输入离开 @ 后,在途的补全任务作废且弹层关闭
            overlay.update_for("hello")
            await pilot.pause(0.15)
            assert not overlay.display
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)


def test_run_tui_mode_dispatch(tmp_path, monkeypatch):
    """Textual 壳统一走稳定 fullscreen；regular 是兼容别名。"""
    from lsm_harness.gateway.tui import LSMTui, run_tui

    monkeypatch.delenv("LSM_TUI_MODE", raising=False)
    captured = []
    monkeypatch.setattr(
        LSMTui, "run", lambda self, **kw: captured.append(kw)
    )
    monkeypatch.setattr(LSMTui, "shutdown", lambda self: None)

    run_tui()
    run_tui(mode="fullscreen")
    run_tui(mode="regular")
    assert captured[0] == {}
    assert captured[1] == {}
    assert captured[2] == {}

    monkeypatch.setenv("LSM_TUI_MODE", "fullscreen")
    run_tui()
    assert captured[3] == {}


def test_shift_tab_cycles_thinking_instead_of_moving_focus(tmp_path, monkeypatch):
    """Textual 把 BackTab 命名为 shift+tab；高优先级绑定必须
    在 Input 聚焦时命中，而不是落到 Screen.focus_previous。"""
    app, harness, _client = _make_app(tmp_path, monkeypatch)
    calls: list[bool] = []

    def cycle() -> str:
        calls.append(True)
        return "minimal"

    harness.cycle_thinking = cycle

    async def main():
        async with app.run_test() as pilot:
            await pilot.pause()
            input_box = app.query_one("#input", Input)
            assert input_box.has_focus
            await pilot.press("shift+tab")
            await pilot.pause()
            assert calls == [True]
            assert input_box.has_focus
        app.shutdown()

    try:
        asyncio.run(main())
    finally:
        unregister_api_provider(_API)
