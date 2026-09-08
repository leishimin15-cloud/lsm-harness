"""阶段 4 批 3:会话能力组合验收(计划验收场景端到端)。

一条链路走完全部会话能力:
开始任务 → 调用工具 → 中断 → 继续 → 压缩 → 回到旧节点 →
创建新分支 → 重启恢复。

每一步之后检查:上下文消息、模型状态、持久化记录(JSONL 树 +
SQLite 投影)都属于正确的分支;被弃分支的压缩摘要不得泄漏到新
分支;重启后 leaf = 文件末行(Pi _buildIndex),open 时即时恢复
该路径的模型状态。
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from lsm_harness.agent.messages import message_preview
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
from lsm_harness.ai.api.common import snapshot
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.ops.session_store import read_session_entries

from helpers import QueueClient, client_stream_fn

_API = "test-e2e-api"


def _fake_get_client(provider_name, model="", small_model="", **_kw):
    client = SimpleNamespace()
    client.model = Model(id=model or "fallback", api=_API, provider=provider_name)
    return client


def _make_app(tmp_path, monkeypatch, model_id="base-main"):
    """同一 home 上构造一个 Harness(模拟一次进程生命周期)。"""
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
        small_model="base-small",
        home=tmp_path,
        # 消息都很短,默认保留预算下手动压缩会"全保留"直接返回;
        # 调小 keep 预算让切割点落在中段,真正产生压缩条目。
        context_keep_recent_tokens=4,
    )
    client.model = Model(id=model_id, api=_API, provider="deepseek")
    app = Harness(
        settings=settings,
        client=client,
        conn=connect(tmp_path, check_same_thread=False),
        stream_fn=client.as_stream_fn(),
    )
    return app, client


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


def _queue(client, text="", **kw):
    client.responses.append(ModelResponse(text=text, usage=Usage(2, 2), **kw))


def _visible_texts(app):
    """当前路径上所有 message 条目的文本预览(含 tool 结果)。"""
    ctx = app.session.build_session_context()
    assert ctx is not None
    return [message_preview(m, limit=1_000_000) for m in ctx.messages]


def test_task_tool_abort_continue_compact_branch_restart(tmp_path, monkeypatch):
    app, client = _make_app(tmp_path, monkeypatch)
    sid = app.session.session_id

    try:
        # ── 1. 开始任务 + 调用工具 ─────────────────────────────
        client.responses.append(ModelResponse(
            tool_calls=[ToolCall("c1", "list_dir", {})],
            stop_reason="tool_calls",
            usage=Usage(2, 2),
        ))
        _queue(client, "完成一")
        first = app.respond("任务一", source="test")
        assert first.status == "completed" and first.reply == "完成一"
        assert [tc["tool"] for tc in first.tool_calls] == ["list_dir"]

        # ── 2. 中断:第二轮停在流里,abort 后问题挂在树梢 ───────
        entered = threading.Event()
        app.stream_fn = _parked_stream(entered)
        outcome: list = []
        worker = threading.Thread(
            target=lambda: outcome.append(app.respond("任务二", source="test")),
            daemon=True,
        )
        worker.start()
        assert entered.wait(5)
        app.abort()
        worker.join(5)
        assert outcome[0].status == "aborted"

        # ── 3. 继续:不重复追加用户消息,直接回答被中断的问题 ──
        app.stream_fn = client.as_stream_fn()
        _queue(client, "继续答二")
        continued = app.respond_continue(source="test")
        assert continued.reply == "继续答二"
        user_so_far = [
            m["content"] for m in app.session.history if m["role"] == "user"
        ]
        assert user_so_far == ["任务一", "任务二"]  # 任务二只出现一次

        # ── 4. 压缩(手动;摘要走 small model,从队列取)─────────
        _queue(client, "摘要:任务一与二")
        assert app.compact() is True

        # ── 5. 压缩后再跑一轮(被弃侧内容),并切模型到被弃侧 ────
        _queue(client, "答三")
        assert app.respond("任务三", source="test").status == "completed"
        app.switch_model("deepseek", model="e2e-switched", small_model="e2e-small")

        # ── 6. 回到旧节点:继续答二的 assistant 消息(压缩之前)──
        entries = read_session_entries(app.session.jsonl_path)
        a2 = next(
            e for e in entries
            if e.type == "message"
            and message_preview(e.message, limit=1_000_000) == "继续答二"
        )
        assert app.session.branch(a2.id, lambda *_: None) == a2.id

        # ── 7. 新分支:branch 不回退模型(批 2,Pi navigateTree)─
        _queue(client, "答四")
        assert app.respond("改做任务四", source="test").status == "completed"
        assert app.settings.model == "e2e-switched"
    finally:
        app.close()
        unregister_api_provider(_API)

    # ── 8. 重启恢复:leaf = 文件末行(新分支),open 即时恢复 ────
    app2, client2 = _make_app(tmp_path, monkeypatch, model_id="other-main")
    try:
        assert app2.session.session_id == sid  # 启动选中最近会话

        # 持久化:整个树(含被弃分支)都在,什么都没删
        all_entries = read_session_entries(app2.session.jsonl_path)
        all_texts = [
            message_preview(e.message, limit=1_000_000)
            for e in all_entries if e.type == "message"
        ]
        assert "任务三" in all_texts and "答三" in all_texts  # 被弃侧仍在文件里
        assert any(e.type == "compaction" for e in all_entries)

        # 展示历史 = 新分支路径(批 1),不含被弃侧
        history_texts = [m["content"] for m in app2.session.history]
        assert "任务三" not in history_texts and "答三" not in history_texts
        assert [m["content"] for m in app2.session.history if m["role"] == "user"] == [
            "任务一", "任务二", "改做任务四",
        ]

        # 上下文属于新分支;被弃侧的压缩摘要不得泄漏(路径作用域)
        texts = _visible_texts(app2)
        assert "改做任务四" in texts and "继续答二" in texts
        assert "任务三" not in texts and "答三" not in texts
        assert app2.session.summary() == ""  # 压缩条目在被弃侧

        # 模型状态:open 时从文件末行所在路径恢复——model_change 在
        # 被弃侧,所以回到 header 基线,而不是启动 settings 的 other-main,
        # 也不是关闭前 live 的 e2e-switched(Pi createAgentSession)。
        assert app2.settings.model == "base-main"

        # 恢复后可继续运行:下一轮请求的上下文仍属于新分支
        _queue(client2, "答五")
        fifth = app2.respond("继续新分支", source="test")
        assert fifth.status == "completed"
        sent = str(client2.calls[-1]["messages"])
        assert "改做任务四" in sent and "任务三" not in sent
    finally:
        app2.close()
        unregister_api_provider(_API)
