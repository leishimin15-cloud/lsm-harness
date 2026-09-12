"""批次 R-1: switch_model 必须把新 Provider 的凭证绑进 stream_fn。

问题原因：旧 ``_apply_model_state`` 先调 ``_resolve_stream_fn()``（闭包捕获
当时的 ``settings.api_key``），之后才把新 Key 写入 settings → 配置显示已切换，
实际请求仍带旧 Provider 的 Key。

本文件验收：
- 跨 Provider 切换后，真实传给流式函数的 ``options.api_key`` 是新 Key；
- 同 Provider 换模型，Key 仍按该 Provider 的规则解析；
- client 构造失败 / api 未注册时，旧状态完整保留（无半更新混合态）；
- 全程假 Provider、假 Key，不访问真实 API。
"""

from __future__ import annotations

import pytest

from lsm_harness.ai.registry import (
    ApiProvider,
    register_api_provider,
    unregister_api_provider,
)
from lsm_harness.ai.providers import (
    ProviderAuthError,
    available_models,
    get_client,
)
from lsm_harness.ai.types import AIContext, Model, StreamOptions
from lsm_harness.coding_agent.app import Harness
from lsm_harness.coding_agent.cli import _cmd_model
from lsm_harness.config import Settings
from lsm_harness.smoke import ScriptedClient

_API = "test-switch-api"
OLD_KEY = "OLD_DUMMY_KEY"
NEW_KEY = "NEW_DUMMY_KEY"


def _capturing_stream(captured: list):
    def fake_stream(model, context, options):
        captured.append({"model": model, "options": options})
        return iter(())

    return fake_stream


def _fake_client(provider_name: str, model_id: str) -> ScriptedClient:
    client = ScriptedClient()
    client.model = Model(
        id=model_id, api=_API, provider=provider_name, reasoning=True
    )
    return client


def _make_harness(tmp_path, monkeypatch, captured: list) -> Harness:
    monkeypatch.delenv("LSM_API_KEY", raising=False)
    monkeypatch.delenv("WAKU_API_KEY", raising=False)
    register_api_provider(ApiProvider(_API, _capturing_stream(captured)))
    settings = Settings(
        api_key=OLD_KEY,
        provider="deepseek",
        model="old-main",
        small_model="old-small",
        home=tmp_path,
    )
    return Harness(settings=settings, client=_fake_client("deepseek", "old-main"))


def _call_stream(harness: Harness) -> None:
    list(
        harness.stream_fn(
            harness.model,
            AIContext(system_prompt="", messages=[], tools=[]),
            StreamOptions(max_tokens=100),
        )
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    captured: list = []
    app = _make_harness(tmp_path, monkeypatch, captured)
    try:
        yield app, captured
    finally:
        app.close()
        unregister_api_provider(_API)


def test_cross_provider_switch_binds_new_key(harness, monkeypatch):
    app, captured = harness
    monkeypatch.setenv("OPENAI_API_KEY", NEW_KEY)
    monkeypatch.setattr(
        "lsm_harness.ai.providers.get_client",
        lambda **kw: _fake_client(kw["provider_name"], kw.get("model") or "new-main"),
    )

    app.switch_model("openai", model="new-main", small_model="new-small")

    assert app.settings.provider == "openai"
    assert app.settings.api_key == NEW_KEY
    _call_stream(app)
    assert captured[-1]["options"].api_key == NEW_KEY
    assert captured[-1]["model"].id == "new-main"


def test_same_provider_model_change_resolves_that_providers_key(harness, monkeypatch):
    app, captured = harness
    monkeypatch.setenv("DEEPSEEK_API_KEY", NEW_KEY)
    monkeypatch.setattr(
        "lsm_harness.ai.providers.get_client",
        lambda **kw: _fake_client(kw["provider_name"], kw.get("model") or "m2"),
    )

    app.switch_model("deepseek", model="m2", small_model="s2")

    _call_stream(app)
    assert captured[-1]["options"].api_key == NEW_KEY
    assert captured[-1]["model"].id == "m2"


def test_failed_client_construction_keeps_old_state(harness, monkeypatch):
    app, captured = harness
    old_client = app.client
    old_stream_fn = app.stream_fn

    def boom(**_kw):
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr("lsm_harness.ai.providers.get_client", boom)

    with pytest.raises(RuntimeError):
        app.switch_model("openai", model="new-main", small_model="new-small")

    # 无半更新态：client/model/stream_fn/settings 全部仍是旧的。
    assert app.client is old_client
    assert app.stream_fn is old_stream_fn
    assert app.model.id == "old-main"
    assert app.settings.provider == "deepseek"
    assert app.settings.api_key == OLD_KEY
    assert app.settings.model == "old-main"
    _call_stream(app)
    assert captured[-1]["options"].api_key == OLD_KEY


def test_missing_key_is_recoverable_provider_error(monkeypatch):
    monkeypatch.delenv("LSM_API_KEY", raising=False)
    monkeypatch.delenv("WAKU_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    with pytest.raises(ProviderAuthError, match="KIMI_API_KEY"):
        get_client(provider_name="kimi-coding")


def test_cli_model_auth_failure_keeps_session_open(harness, monkeypatch):
    app, _captured = harness
    models = available_models()
    kimi_choice = str(next(
        i for i, model in enumerate(models, 1)
        if model.provider == "kimi-coding" and model.id == "k3"
    ))
    monkeypatch.delenv("LSM_API_KEY", raising=False)
    monkeypatch.delenv("WAKU_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt: kimi_choice)

    old_client = app.client
    old_model = app.settings.model
    _cmd_model(app)

    assert app.client is old_client
    assert app.settings.provider == "deepseek"
    assert app.settings.model == old_model


def test_unregistered_api_rejected_before_any_swap(harness, monkeypatch):
    app, captured = harness
    bad_client = ScriptedClient()
    bad_client.model = Model(id="new-main", api="no-such-api", provider="openai")
    monkeypatch.setattr(
        "lsm_harness.ai.providers.get_client", lambda **kw: bad_client
    )

    with pytest.raises(ValueError, match="not registered"):
        app.switch_model("openai", model="new-main", small_model="new-small")

    assert app.settings.provider == "deepseek"
    assert app.settings.api_key == OLD_KEY
    assert app.model.id == "old-main"
    _call_stream(app)
    assert captured[-1]["options"].api_key == OLD_KEY


def test_switch_model_and_thinking_sync_agent_state(harness, monkeypatch):
    """批 4:model/thinking 迁入 AgentState——switch_model /
    set_thinking 之后,agent.state 必须同步(config 不再逐 run 携带
    model/thinking,run 时从 state 解析)。"""
    app, _captured = harness
    monkeypatch.setenv("OPENAI_API_KEY", NEW_KEY)
    monkeypatch.setattr(
        "lsm_harness.ai.providers.get_client",
        lambda **kw: _fake_client(kw["provider_name"], kw.get("model") or "new-main"),
    )

    assert app.agent.state.model is not None
    assert app.agent.state.model.id == "old-main"
    app.switch_model("openai", model="new-main", small_model="new-small")
    assert app.agent.state.model.id == "new-main"
    assert app.agent.state.model.provider == "openai"

    app.set_thinking("enabled")  # 旧三档入参归一为 high
    assert app.agent.state.thinking_level == "high"
    app.cycle_thinking()
    assert app.agent.state.thinking_level == "off"  # high → 回绕到 off


def _k3_client(**_kw):
    """K3 式 thinking 映射:off:null = 不能真正关闭 thinking。"""
    client = ScriptedClient()
    client.model = Model(
        id="k3",
        api=_API,
        provider="kimi-coding",
        reasoning=True,
        thinking_level_map={
            "off": None,
            "minimal": "low",
            "low": "low",
            "medium": "high",
            "high": "high",
            "xhigh": "max",
            "max": "max",
        },
    )
    return client


def test_thinking_clamps_to_model_capabilities_on_switch(harness, monkeypatch):
    """换模型后 thinking 重新 clamp:当前 off 切到 K3(off:null)自动升
    minimal;Shift+Tab 只在 K3 支持的档位间循环;切回非 reasoning
    模型自动落 off。"""
    app, _captured = harness
    assert app.settings.thinking == "off"

    monkeypatch.setenv("KIMI_API_KEY", NEW_KEY)
    monkeypatch.setattr("lsm_harness.ai.providers.get_client", _k3_client)
    app.switch_model("kimi-coding", model="k3", small_model="k3")
    # K3 不支持 off:自动升到最近可用档 minimal(显示与实际请求一致)
    assert app.settings.thinking == "minimal"
    assert app.agent.state.thinking_level == "minimal"

    # Shift+Tab 只在 K3 支持档位间循环(off 不在其中)
    levels = [app.cycle_thinking() for _ in range(6)]
    assert levels == ["low", "medium", "high", "xhigh", "max", "minimal"]

    # 切到非 reasoning 模型:任何档位都落回 off
    def plain_client(**kw):
        client = ScriptedClient()
        client.model = Model(
            id=kw.get("model") or "plain",
            api=_API,
            provider=kw.get("provider_name") or "deepseek",
            reasoning=False,
        )
        return client

    monkeypatch.setattr("lsm_harness.ai.providers.get_client", plain_client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", NEW_KEY)
    app.switch_model("deepseek", model="plain", small_model="plain")
    assert app.settings.thinking == "off"
    assert app.cycle_thinking() == "off"  # 只有一档,原地循环
