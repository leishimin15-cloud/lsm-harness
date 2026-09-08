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

from types import SimpleNamespace

import pytest

from lsm_harness.ai.registry import (
    ApiProvider,
    register_api_provider,
    unregister_api_provider,
)
from lsm_harness.ai.types import AIContext, Model, StreamOptions
from lsm_harness.coding_agent.app import Harness
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
    client.model = Model(id=model_id, api=_API, provider=provider_name)
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
