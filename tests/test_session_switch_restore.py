"""阶段 4 批 2:会话切换即时恢复运行状态(Pi createAgentSession 对齐)。

目标行为(以本地 Pi 为准):
- switch_session / 启动(会话 open)时,立即把该会话**当前路径**的
  model/small_model/thinking 应用到运行时——Pi 在 createAgentSession 里
  用 buildSessionContext 的 model/thinkingLevel 恢复,不是等下一次
  prompt 才懒恢复;
- 会话内 branch 只更新消息,**不改变模型**(Pi navigateTree 只设置
  agent.state.messages)——用户中途显式切过的模型不会因为回到旧节点
  就被静默回退;重启后 open 恢复自然从文件末行所在路径取状态;
- 恢复失败(目标 provider 未注册 / client 构造失败)不阻断切换,
  保留当前模型并记录 fallback 事件(类比 Pi modelFallbackMessage)。

这是行为对齐,不是认定此前"懒恢复"(code-review issue 二)是错误。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lsm_harness.agent.messages import AssistantMessage, UserMessage
from lsm_harness.ai.registry import (
    ApiProvider,
    register_api_provider,
    unregister_api_provider,
)
from lsm_harness.ai.types import Model, ModelResponse, Usage
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.ops.session_store import read_session_entries

from helpers import QueueClient, client_stream_fn

_API = "test-restore-api"


def _fake_get_client(provider_name, model="", small_model="", **_kw):
    """get_client 替身:返回带 Model 的轻量 client,api 指向注册表假实现。"""
    client = SimpleNamespace()
    client.model = Model(
        id=model or "fallback", api=_API, provider=provider_name, reasoning=True
    )
    return client


def _make_app(tmp_path, monkeypatch, *responses, model_id="old-main"):
    monkeypatch.delenv("LSM_API_KEY", raising=False)
    monkeypatch.delenv("WAKU_API_KEY", raising=False)
    client = QueueClient(*responses)
    # 切换模型后,注册表假 api 的流也从同一队列取响应。
    register_api_provider(
        ApiProvider(_API, lambda m, c, o: client_stream_fn(client)(m, c, o))
    )
    monkeypatch.setattr("lsm_harness.ai.providers.get_client", _fake_get_client)
    settings = Settings(
        api_key="k",
        provider="deepseek",
        model=model_id,
        small_model="old-small",
        home=tmp_path,
    )
    client.model = Model(
        id=model_id, api=_API, provider="deepseek", reasoning=True
    )
    app = Harness(settings=settings, client=client, stream_fn=client.as_stream_fn())
    return app, client


@pytest.fixture
def app_pair(tmp_path, monkeypatch):
    app, client = _make_app(tmp_path, monkeypatch)
    try:
        yield app, client
    finally:
        app.close()
        unregister_api_provider(_API)


def _record_exchange(app, user="问", assistant="答"):
    recorder = app.session.recorder
    recorder.record(UserMessage(content=user), source="test")
    recorder.record(AssistantMessage(text=assistant), source="test")


def test_switch_session_restores_model_and_thinking_immediately(app_pair):
    """切换会话后 settings 立即反映目标会话的路径状态(当前懒恢复
    要等到下一次 respond 才生效——此测试暴露该差距)。"""
    app, _ = app_pair
    sid_a = app.session.session_id
    _record_exchange(app)
    # 只记录到树,不动 live settings(模拟"这个会话当初用过的状态")
    app.session.record_model_change("deepseek", "a-model", "a-small")
    app.session.record_thinking_change("enabled")

    app.new_session()  # 离开 A;live 仍是 old-main/disabled
    assert app.settings.model == "old-main"

    assert app.switch_session(sid_a) == sid_a
    # 立即恢复,不需要先 respond
    assert app.settings.model == "a-model"
    assert app.settings.small_model == "a-small"
    assert app.settings.thinking == "high"  # 旧 JSONL "enabled" 恢复时归一


def test_in_session_branch_does_not_change_model(app_pair):
    """会话内 branch 只更新消息,模型保持用户显式切换后的值。

    当前懒恢复在下一次 respond 时按路径状态把 new-model 回退成
    header 基线 old-main——此测试暴露该差距(Pi navigateTree 不碰模型)。
    """
    app, client = app_pair
    client.responses.append(ModelResponse(text="答", usage=Usage(2, 2)))
    client.responses.append(ModelResponse(text="再答", usage=Usage(2, 2)))
    first = app.respond("问题", source="test")
    assert first.status == "completed"

    app.switch_model("deepseek", model="new-model", small_model="new-small")
    assert app.settings.model == "new-model"

    entries = [
        e for e in read_session_entries(app.session.jsonl_path)
        if e.type == "message"
    ]
    fork = entries[0].id  # 第一个 user 消息,在 model_change 之前
    assert app.session.branch(fork, lambda *_: None) == fork

    second = app.respond("重来", source="test")
    assert second.status == "completed"
    assert app.settings.model == "new-model"
    assert app.settings.small_model == "new-small"


def test_restore_failure_does_not_block_switch(app_pair, monkeypatch):
    """恢复时模型构造失败:切换照常完成,保留当前模型,记录 fallback。"""
    app, _ = app_pair
    sid_a = app.session.session_id
    _record_exchange(app)
    app.session.record_model_change("deepseek", "a-model", "a-small")

    app.new_session()

    def boom(**_kw):
        raise RuntimeError("client construction failed")

    monkeypatch.setattr("lsm_harness.ai.providers.get_client", boom)
    assert app.switch_session(sid_a) == sid_a  # 不抛出
    assert app.settings.model == "old-main"    # 保留当前模型


def test_startup_restores_recorded_state(tmp_path, monkeypatch):
    """启动(会话 open)即恢复:第一个 Harness 记录过状态后,同一 home
    上新建的 Harness 立即拿到该会话的模型/thinking。"""
    app1, _ = _make_app(tmp_path, monkeypatch)
    sid = app1.session.session_id
    _record_exchange(app1)
    app1.session.record_model_change("deepseek", "a-model", "a-small")
    app1.session.record_thinking_change("enabled")
    app1.close()

    # 模拟重启:settings 是另一份(模型 different),但 latest 会话是 sid
    app2, _ = _make_app(tmp_path, monkeypatch, model_id="other-main")
    try:
        assert app2.session.session_id == sid  # 启动选中最近会话
        assert app2.settings.model == "a-model"
        assert app2.settings.thinking == "high"  # 旧 JSONL "enabled" 恢复时归一
    finally:
        app2.close()
