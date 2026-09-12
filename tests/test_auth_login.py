"""API-key login, startup fallback, and concrete model-catalog tests."""

from __future__ import annotations

import asyncio
import json
import stat
import sys
from types import SimpleNamespace

from textual.widgets import Input

from lsm_harness.ai.providers import PROVIDERS, available_models, get_client
from lsm_harness.coding_agent.auth_storage import read_api_key, save_api_key
from lsm_harness.coding_agent.model_config import load_model_catalog
from lsm_harness.coding_agent.startup import (
    provider_auth_status,
    resolve_startup_settings,
)
from lsm_harness.config import Settings
from lsm_harness.gateway.tui import LSMTui
from lsm_harness.gateway.tui.screens import ApiKeyScreen


_KEY_ENVS = tuple(provider.key_env for provider in PROVIDERS.values())


def _clear_auth(monkeypatch) -> None:
    for name in ("LSM_API_KEY", "WAKU_API_KEY", *_KEY_ENVS):
        monkeypatch.delenv(name, raising=False)


def test_auth_storage_is_typed_and_user_only(tmp_path):
    path = save_api_key(tmp_path, "kimi", "secret-test-key")

    assert read_api_key(tmp_path, "kimi") == "secret-test-key"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert '"type": "api_key"' in path.read_text(encoding="utf-8")


def test_startup_falls_back_to_configured_provider(tmp_path, monkeypatch):
    _clear_auth(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    (tmp_path / "models.json").write_text(json.dumps({
        "providers": {
            "kimi": {
                "name": "kimi",
                "baseUrl": "https://example.invalid/v1",
                "api": "openai-completions",
                "models": [{"id": "kimi-k3"}],
            }
        }
    }))
    settings = Settings(
        provider="kimi",
        api_key="",
        model="kimi-k3",
        small_model="kimi-k2.6",
        home=tmp_path,
    )

    notice = resolve_startup_settings(settings)

    assert settings.provider == "deepseek"
    assert settings.model == PROVIDERS["deepseek"].model
    assert notice is not None and "no auth configured" in notice


def test_models_json_kimi_and_builtin_kimi_coding_coexist(tmp_path):
    (tmp_path / "models.json").write_text(json.dumps({
        "providers": {
            "kimi": {
                "baseUrl": "https://example.invalid/v1",
                "api": "openai-completions",
                "apiKey": "literal-test-key",
                "models": [
                    {"id": "custom-kimi-a"},
                    {"id": "custom-kimi-b", "contextWindow": 256000},
                ],
            }
        }
    }))

    catalog = load_model_catalog(tmp_path)

    assert "kimi" in catalog.providers
    assert "kimi-coding" in catalog.providers
    assert "llama.cpp" in catalog.providers
    assert len(catalog.providers) == 41
    assert catalog.providers["kimi"].models == (
        "custom-kimi-a", "custom-kimi-b"
    )
    assert catalog.providers["kimi-coding"].models == (
        "k3", "k3-256k", "kimi-for-coding",
        "kimi-for-coding-highspeed",
    )
    status = provider_auth_status("kimi", home=tmp_path, catalog=catalog)
    assert status.configured
    assert status.source == "models_json_key"


def test_builtin_catalog_is_synced_from_pi():
    assert len(PROVIDERS) == 39
    assert len(available_models()) == 1345
    assert len(PROVIDERS["openai"].models) == 39
    assert len(PROVIDERS["anthropic"].models) == 14
    assert len(PROVIDERS["google"].models) == 22
    assert len(PROVIDERS["openrouter"].models) == 366


def test_kimi_coding_catalog_matches_pi():
    ids = [
        model.id
        for model in available_models(providers=("kimi-coding",))
    ]

    assert ids == [
        "k3",
        "k3-256k",
        "kimi-for-coding",
        "kimi-for-coding-highspeed",
    ]
    k3 = available_models(providers=("kimi-coding",))[0]
    assert k3.context_window == 1_048_576
    assert k3.max_tokens == 131_072
    assert k3.force_adaptive_thinking
    assert k3.allow_empty_thinking_signature


def test_kimi_api_key_uses_pi_api_key_auth_not_bearer(monkeypatch):
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=FakeAnthropic),
    )

    get_client(provider_name="kimi-coding", api_key="test-kimi-key")

    assert captured["api_key"] == "test-kimi-key"
    assert "auth_token" not in captured
    assert captured["base_url"] == "https://api.kimi.com/coding"
    assert captured["max_retries"] == 0


def test_tui_without_credentials_stays_open_for_login(tmp_path, monkeypatch):
    _clear_auth(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=lambda **_kwargs: object()),
    )
    monkeypatch.setenv("LSM_PROVIDER", "kimi-coding")
    monkeypatch.setenv("LSM_HOME", str(tmp_path))
    app = LSMTui()

    async def main():
        async with app.run_test() as pilot:
            for _ in range(100):
                await pilot.pause(0.01)
                if isinstance(app.screen, ApiKeyScreen):
                    break
            assert app.is_running
            assert app.harness is None
            assert isinstance(app.screen, ApiKeyScreen)
            assert "Use /login" in " ".join(
                item.text for item in app.state.transcript
            )

            key_input = app.screen.query_one("#api-key", Input)
            key_input.value = "stored-kimi-test-key"
            await pilot.press("enter")
            for _ in range(200):
                await pilot.pause(0.01)
                if app.harness is not None:
                    break
            assert app.harness is not None
            assert app.harness.settings.provider == "kimi-coding"
            assert read_api_key(tmp_path, "kimi-coding") == "stored-kimi-test-key"
            assert app.query_one("#input", Input).value == ""
            assert all(
                "stored-kimi-test-key" not in item.text
                for item in app.state.transcript
            )
        app.shutdown()

    asyncio.run(main())
