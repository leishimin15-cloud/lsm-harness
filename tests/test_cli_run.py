"""CLI 启动竞态(四问题之 4):主线程先 begin_run,再启动 worker。

RPC 的 _start_worker 已是正确形状(接受线程上 begin_run,worker 只拿
active_run 执行);CLI 的 run_chat 必须对齐——否则 worker 启动到
respond 内部 begin 之间存在"已接受但未 running"的窗口:此期间
is_running 为假,第二次提交会被当成新运行再 begin 一次。
"""

from __future__ import annotations

from io import StringIO

from lsm_harness.ai.types import ModelResponse, Usage
from lsm_harness.coding_agent.app import Harness
from lsm_harness.coding_agent.cli import _run_cli_trace, console
from lsm_harness.config import Settings
from lsm_harness.db import connect

from helpers import QueueClient


def test_run_cli_trace_executes_prebegun_run(tmp_path):
    """主线程 begin_run 后 is_running 立即为真(无线程、无等待);
    worker 侧 _run_cli_trace 拿 active_run 直接执行——respond 不再自行
    begin(重复 begin 会 RuntimeError,被当作本轮失败打印)。"""
    settings = Settings(api_key="scripted", home=tmp_path)
    conn = connect(tmp_path, check_same_thread=False)
    client = QueueClient(ModelResponse(text="CLI 答复", usage=Usage(2, 2)))
    app = Harness(
        settings=settings, client=client, conn=conn,
        stream_fn=client.as_stream_fn(),
    )
    capture = StringIO()
    original_file = console.file
    console._file = capture
    try:
        active = app.begin_run()
        assert app.is_running is True  # 接受即 running,无启动窗口
        _run_cli_trace(app, "问题", active)
        assert app.wait_for_idle(0) is True
        assert "本轮失败" not in capture.getvalue()
        assert client.calls  # 模型真的被调用了
    finally:
        console._file = original_file
        app.close()
