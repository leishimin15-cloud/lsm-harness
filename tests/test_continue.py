"""continue()：不重复追加用户消息的重跑（阶段 1 · Batch 4，验收 b）。

语义：respond_continue() 在不追加新 user 消息的前提下重跑 loop——
用于"被中断的问题还没答"或"还有排队的 follow-up"两种场景。
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


def _parked_stream(entered: threading.Event):
    """停在流里直到运行被中断，然后以 aborted 收尾（同 test_rpc 的阻塞流）。"""
    import time

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


def _user_contents(app) -> list[str]:
    return [
        m["content"] for m in app.session.history if m.get("role") == "user"
    ]


def test_continue_after_abort_answers_interrupted_question(tmp_path):
    """abort 后 context 以 user 结尾：continue 直接回答被中断的问题，
    不把该问题再追加一遍。"""
    entered = threading.Event()
    app, client = _app(tmp_path, ModelResponse(text="答一", usage=Usage(2, 2)))
    try:
        first = app.respond("问题一", source="test")
        assert first.status == "completed"

        # 第二轮停在流里，然后 abort——"问题二"已被 recorder 写进树，
        # 但没有得到回答。
        app.stream_fn = _parked_stream(entered)
        worker = threading.Thread(
            target=lambda: app.respond("问题二", source="test"), daemon=True
        )
        worker.start()
        assert entered.wait(5)
        app.abort()
        worker.join(5)
        assert app.is_running is False

        # continue：不追加任何 user 消息，直接重跑 loop。
        app.stream_fn = client.as_stream_fn()
        client.responses.append(ModelResponse(text="答二", usage=Usage(2, 2)))
        result = app.respond_continue(source="test")

        assert result.status == "completed"
        assert result.reply == "答二"
        # 模型收到的最后一条消息正是被中断的"问题二"（且只出现一次）。
        last_call_messages = client.calls[-1]["messages"]
        occurrences = sum(
            1 for m in last_call_messages
            if m.get("role") == "user" and m.get("content") == "问题二"
        )
        assert occurrences == 1
        assert last_call_messages[-1]["content"] == "问题二"
        # 内存 history 不重复：阶段 4 批 3 起 abort 轮会从树 resync
        #（"问题二"是 recorder 写进树的 live 记录，恰好一次），continue
        # 自身不追加任何 user 消息。
        assert _user_contents(app) == ["问题一", "问题二"]
        # SQLite 里这次 continue 的交换以空 user 内容 + continued 元数据记录
        #（meta 挂在 assistant 行上）。
        user_row = app.conn.execute(
            "SELECT content FROM chat_log WHERE role='user' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert user_row[0] == ""
        meta_row = app.conn.execute(
            "SELECT meta FROM chat_log WHERE role='assistant' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert "continued" in (meta_row[0] or "")
    finally:
        app.close()


def test_continue_drains_queued_follow_ups(tmp_path):
    """运行中排队的 follow-up 在 abort 后由 continue 消费。"""
    entered = threading.Event()
    app, client = _app(tmp_path, ModelResponse(text="答一", usage=Usage(2, 2)))
    try:
        assert app.respond("问题一", source="test").status == "completed"

        app.stream_fn = _parked_stream(entered)
        worker = threading.Thread(
            target=lambda: app.respond("问题二", source="test"), daemon=True
        )
        worker.start()
        assert entered.wait(5)
        assert app.follow_up("顺便做这个") is True
        app.abort()
        worker.join(5)

        app.stream_fn = client.as_stream_fn()
        client.responses.extend([
            ModelResponse(text="答二", usage=Usage(2, 2)),
            ModelResponse(text="答后续", usage=Usage(2, 2)),
        ])
        result = app.respond_continue(source="test")

        assert result.status == "completed"
        assert result.reply == "答后续"  # follow-up 驱动了外圈第二轮
        # 两次模型调用：第一次答"问题二"，第二次答 follow-up。
        assert len(client.calls) == 3  # 答一 + 答二 + 答后续
        assert client.calls[-1]["messages"][-1]["content"] == "顺便做这个"
        # 全程没有 continue 自己追加的 user 消息："问题二"恰好一次
        #（abort 轮由树 resync 进 history，阶段 4 批 3），follow-up 由
        # recorder 进树，不进 history 的 user 半边。
        assert _user_contents(app) == ["问题一", "问题二"]
    finally:
        app.close()


def test_continue_without_pending_work_raises(tmp_path):
    """正常完成且无排队工作时，continue 没有合法输入——直接拒绝。"""
    app, _client = _app(tmp_path, ModelResponse(text="答一", usage=Usage(2, 2)))
    try:
        assert app.respond("问题一", source="test").status == "completed"
        with pytest.raises(ValueError, match="nothing to continue"):
            app.respond_continue(source="test")
    finally:
        app.close()


def test_first_turn_abort_keeps_question_and_continue_answers_it(tmp_path):
    """新会话第一次提问就中断：问题不丢——持久化进 JSONL 树、history
    经 resync 含它、continue 直接回答它（模型看到的最后一条 user 消息
    正是该问题，且恰好一次）。"""
    from lsm_harness.ops.session_store import path_to_leaf, read_session_entries

    entered = threading.Event()
    app, client = _app(tmp_path)
    try:
        app.stream_fn = _parked_stream(entered)
        worker = threading.Thread(
            target=lambda: app.respond("首个问题", source="test"), daemon=True
        )
        worker.start()
        assert entered.wait(5)
        app.abort()
        worker.join(5)
        assert app.is_running is False

        # 问题进了树(事实来源):文件存在,当前路径末端消息就是这个 user。
        assert app.session.jsonl_path.exists()
        entries = read_session_entries(app.session.jsonl_path)
        path = path_to_leaf(entries, app.session.recorder.last_entry_id)
        messages = [e for e in path if e.type == "message"]
        assert [e.message.role for e in messages] == ["user"]
        # 展示历史从树 resync,问题在里头。
        assert _user_contents(app) == ["首个问题"]

        # continue:不追加 user 消息,直接回答被中断的首个问题。
        app.stream_fn = client.as_stream_fn()
        client.responses.append(ModelResponse(text="首个答复", usage=Usage(2, 2)))
        result = app.respond_continue(source="test")

        assert result.status == "completed"
        assert result.reply == "首个答复"
        last_call_messages = client.calls[-1]["messages"]
        occurrences = sum(
            1 for m in last_call_messages
            if m.get("role") == "user" and m.get("content") == "首个问题"
        )
        assert occurrences == 1
        assert last_call_messages[-1]["content"] == "首个问题"
        assert _user_contents(app) == ["首个问题"]
    finally:
        app.close()


def test_restart_continues_interrupted_question(tmp_path):
    """重启后继续中断的任务:合法性来自持久化会话树(当前路径末端是
    未回答的 user 消息),不是上一进程的内存状态。新进程没有
    _last_status,continue 照样成立并回答树梢那个问题。"""
    entered = threading.Event()
    app, _client = _app(tmp_path, ModelResponse(text="答一", usage=Usage(2, 2)))
    try:
        assert app.respond("问题一", source="test").status == "completed"
        app.stream_fn = _parked_stream(entered)
        worker = threading.Thread(
            target=lambda: app.respond("问题二", source="test"), daemon=True
        )
        worker.start()
        assert entered.wait(5)
        app.abort()
        worker.join(5)
        session_id = app.session.session_id
    finally:
        app.close()

    # 模拟重启:同 home 新进程,自动 resume 最近会话。
    app2, client2 = _app(tmp_path, ModelResponse(text="答二", usage=Usage(2, 2)))
    try:
        assert app2.session.session_id == session_id
        result = app2.respond_continue(source="test")
        assert result.status == "completed"
        assert result.reply == "答二"
        # 模型收到的最后一条 user 消息正是被中断的"问题二",恰好一次。
        last_call_messages = client2.calls[-1]["messages"]
        occurrences = sum(
            1 for m in last_call_messages
            if m.get("role") == "user" and m.get("content") == "问题二"
        )
        assert occurrences == 1
        assert last_call_messages[-1]["content"] == "问题二"
    finally:
        app2.close()


def test_restart_continues_when_tree_tip_is_tool_result(tmp_path):
    """工具结果落盘后、下一次模型调用前中断:树梢是 tool 消息(Pi 的
    toolResult,合法继续点)。重启后 continue 必须成立——模型接着工具
    结果继续,而不是被误判为"没什么可继续"。"""
    from lsm_harness.ai.types import ToolCall
    from lsm_harness.ops.session_store import path_to_leaf, read_session_entries

    entered = threading.Event()
    app, _client = _app(tmp_path)
    client = QueueClient(
        ModelResponse(
            tool_calls=[ToolCall("c1", "list_dir", {})],
            stop_reason="tool_calls",
            usage=Usage(2, 2),
        )
    )
    try:
        # 第一轮模型调用要求工具;工具执行完、结果落盘;第二轮模型调用
        # 停在流里被 abort——树梢停在 tool 消息上。
        responses = iter([
            client.as_stream_fn(),
            _parked_stream(entered),
        ])
        app.stream_fn = lambda m, c, o: next(responses)(m, c, o)
        worker = threading.Thread(
            target=lambda: app.respond("列个目录", source="test"), daemon=True
        )
        worker.start()
        assert entered.wait(5)
        app.abort()
        worker.join(5)

        entries = read_session_entries(app.session.jsonl_path)
        path = path_to_leaf(entries, app.session.recorder.last_entry_id)
        messages = [e for e in path if e.type == "message"]
        assert messages[-1].message.role == "tool"  # 树梢是工具结果
        session_id = app.session.session_id
    finally:
        app.close()

    # 重启:树梢是 tool,continue 合法,模型收到的最后一条消息是工具结果。
    app2, client2 = _app(tmp_path, ModelResponse(text="继续答复", usage=Usage(2, 2)))
    try:
        assert app2.session.session_id == session_id
        result = app2.respond_continue(source="test")
        assert result.status == "completed"
        assert result.reply == "继续答复"
        assert client2.calls[-1]["messages"][-1]["role"] == "tool"
    finally:
        app2.close()


def test_continue_with_assistant_tip_and_queued_follow_up_injects_first(tmp_path):
    """树梢是 assistant 但队列里有 follow-up(运行收尾阶段的入队竞争):
    continue 必须先把 follow-up 经 initial 通道注入,再调模型——否则
    第一次模型调用会拿 assistant 结尾的上下文(provider 会拒绝)。"""
    app, client = _app(tmp_path, ModelResponse(text="答一", usage=Usage(2, 2)))
    try:
        assert app.respond("问题一", source="test").status == "completed"
        # 公共 API 在空闲时拒绝入队;这里直接入队模拟"运行最后一刻入队、
        # loop 没来得及消费"的竞争残留。
        app.agent.follow_up_queue.enqueue("收尾时排队的后续")
        client.responses.append(ModelResponse(text="答后续", usage=Usage(2, 2)))

        result = app.respond_continue(source="test")

        assert result.status == "completed"
        assert result.reply == "答后续"
        # continue 只调一次模型,且末条消息就是被注入的 follow-up。
        assert len(client.calls) == 2
        last = client.calls[-1]["messages"][-1]
        assert last["role"] == "user"
        assert last["content"] == "收尾时排队的后续"
    finally:
        app.close()


def test_restart_after_completed_run_rejects_continue(tmp_path):
    """正常完成后重启:树梢是 assistant,没有未回答的问题——拒绝。"""
    app, _client = _app(tmp_path, ModelResponse(text="答一", usage=Usage(2, 2)))
    try:
        assert app.respond("问题一", source="test").status == "completed"
    finally:
        app.close()

    app2, _client2 = _app(tmp_path)
    try:
        with pytest.raises(ValueError, match="nothing to continue"):
            app2.respond_continue(source="test")
    finally:
        app2.close()


def test_first_turn_failure_keeps_question(tmp_path):
    """首轮运行失败(流抛错)同样保留问题:问题与 loop 补记的 assistant
    错误消息都留在树里。"""
    from lsm_harness.ops.session_store import path_to_leaf, read_session_entries

    app, _client = _app(tmp_path)

    def broken_stream(_model, _context, _options):
        raise ConnectionError("network down")
        yield  # pragma: no cover - 使其成为生成器

    try:
        app.stream_fn = broken_stream
        result = app.respond("失败的问题", source="test")
        assert result.status == "failed"

        assert app.session.jsonl_path.exists()
        entries = read_session_entries(app.session.jsonl_path)
        path = path_to_leaf(entries, app.session.recorder.last_entry_id)
        messages = [e for e in path if e.type == "message"]
        # 问题在前;loop 为失败的流补记了一条 assistant 错误消息——两者
        # 都是发生过的事实,都留在树里。
        assert [e.message.role for e in messages] == ["user", "assistant"]
        assert messages[0].message.content == "失败的问题"
        assert _user_contents(app) == ["失败的问题"]
    finally:
        app.close()
