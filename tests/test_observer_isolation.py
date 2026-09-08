"""订阅者失败契约(阶段 3 · 批 1)。

- UI 观察者(observer)异常被隔离:运行继续、持久化完成,tracer 里留
  一条 observer.failed 记录;
- 记录层(tracer.write)异常不被吞:运行必须失败——不允许"模型调用了
  但没留下记录"。
"""

from __future__ import annotations

import json

import pytest

from lsm_harness.ai.types import ModelResponse, Usage
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect

from helpers import QueueClient


def _app(tmp_path, *responses):
    settings = Settings(api_key="scripted", home=tmp_path)
    conn = connect(tmp_path, check_same_thread=False)
    client = QueueClient(*responses)
    return (
        Harness(
            settings=settings,
            client=client,
            conn=conn,
            stream_fn=client.as_stream_fn(),
        ),
        client,
    )


def _trace_kinds(app) -> list[str]:
    return [
        json.loads(line)["type"]
        for line in app.tracer.path.read_text(encoding="utf-8").splitlines()
    ]


def test_observer_exception_is_isolated_and_logged(tmp_path):
    """前端渲染抛异常:运行照常完成、交换照常落库、tracer 留证。"""
    app, _client = _app(
        tmp_path,
        ModelResponse(text="答", usage=Usage(2, 2)),
        ModelResponse(text="答二", usage=Usage(2, 2)),
    )
    try:
        def bad_observer(_event):
            raise RuntimeError("ui boom")

        result = app.respond("问题", observer=bad_observer, source="test")

        assert result.status == "completed"
        assert result.reply == "答"
        # 持久化未被破坏:user + assistant 都进了 chat_log。
        row = app.conn.execute("SELECT COUNT(*) FROM chat_log").fetchone()
        assert row[0] >= 2
        kinds = _trace_kinds(app)
        assert "observer.failed" in kinds
        assert "trace.completed" in kinds
        # 第二次运行不受影响。
        second = app.respond("再问", source="test")
        assert second.status == "completed"
    finally:
        app.close()


def test_tracer_write_failure_fails_the_run(tmp_path):
    """记录层写不进去:运行必须失败,且模型调用绝不能先于记录发生。"""
    app, client = _app(tmp_path, ModelResponse(text="答", usage=Usage(2, 2)))
    try:
        def boom(_event):
            raise OSError("disk full")

        app.tracer.write = boom  # type: ignore[assignment]

        with pytest.raises(OSError, match="disk full"):
            app.respond("问题", source="test")
        assert client.calls == []
    finally:
        app.close()
