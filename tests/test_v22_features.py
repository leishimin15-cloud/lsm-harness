"""Regression coverage for the v2.2 runtime wiring."""

from __future__ import annotations

from pathlib import Path

from lsm_harness.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.gateway.cli import _trace_status_markup
from lsm_harness.loop.hooks import LoopHooks
from lsm_harness.ops.file_state import FileState
from lsm_harness.smoke import ScriptedClient
from lsm_harness.tools.filesystem import _write_file_safe
from lsm_harness.tools.shell import _exec_shell
from lsm_harness.types import TraceResult


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "api_key": "scripted",
        "model": "scripted-main",
        "small_model": "scripted-small",
        "home": tmp_path / ".lsm",
        "sandbox_project_dir": str(tmp_path),
        "sandbox_enabled": False,
        "consolidate_every": 99,
    }
    values.update(overrides)
    return Settings(**values)


def test_host_shell_uses_workspace_and_blocks_escape(tmp_path):
    output = _exec_shell("pwd", home=tmp_path)
    assert str(tmp_path.resolve()) in output

    escaped = _exec_shell("pwd", cwd="/", home=tmp_path)
    assert escaped.startswith("Error:")
    assert "outside the allowed workspace" in escaped


def test_requested_sandbox_never_falls_back_to_host(tmp_path):
    output = _exec_shell("pwd", home=tmp_path, sandbox_required=True)
    assert output.startswith("Error:")
    assert "Host shell execution is disabled" in output


def test_file_state_tracks_and_undoes_new_and_empty_files(tmp_path):
    state = FileState(tmp_path)
    created = _write_file_safe("new.txt", "hello", tmp_path, state)
    assert "new" in created
    assert state.modified_files == [str((tmp_path / "new.txt").resolve())]
    assert state.undo(str(tmp_path / "new.txt"))
    assert not (tmp_path / "new.txt").exists()

    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    _write_file_safe("empty.txt", "changed", tmp_path, state)
    assert state.undo("empty.txt")
    assert empty.exists()
    assert empty.read_text(encoding="utf-8") == ""


def test_registry_wires_file_state(tmp_path):
    app = Harness(settings=_settings(tmp_path), client=ScriptedClient())
    try:
        result = app.tools.execute(
            "write_file", {"path": "tracked.txt", "content": "hello"}
        )
        assert not result.is_error
        assert app.file_state.modified_files
    finally:
        app.close()


def test_harness_passes_lifecycle_hooks(tmp_path):
    seen: list[str] = []
    hooks = LoopHooks(
        on_trace_start=lambda _message, _emit: seen.append("trace"),
        on_turn_start=lambda turn_index, _emit: seen.append(f"turn:{turn_index}"),
        on_trace_end=lambda result, _emit: seen.append(f"end:{result.status}"),
    )
    app = Harness(
        settings=_settings(tmp_path),
        client=ScriptedClient(),
        hooks=hooks,
    )
    try:
        app.respond("create a test event")
        assert seen == ["trace", "turn:1", "turn:2", "end:completed"]
    finally:
        app.close()


def test_default_cli_formats_trace_outcomes():
    assert _trace_status_markup(TraceResult(reply="ok")) is None
    assert _trace_status_markup(
        TraceResult(
            reply="failed",
            status="failed",
            stop_reason="error",
            error="provider unavailable",
        )
    ) == "[red]本轮失败：provider unavailable[/red]"
    assert _trace_status_markup(
        TraceResult(reply="任务已中断。", status="aborted", stop_reason="aborted")
    ) == "[yellow]本轮已中断。[/yellow]"


def test_model_switch_updates_all_consumers(tmp_path, monkeypatch):
    app = Harness(settings=_settings(tmp_path), client=ScriptedClient())
    replacement = ScriptedClient()
    monkeypatch.setattr("lsm_harness.ai.providers.get_client", lambda **_kw: replacement)
    try:
        app.switch_model("deepseek", model="new-main", small_model="new-small")
        assert app.client is replacement
        assert app.memory.client is replacement
        assert app.settings.model == "new-main"
        assert app.settings.small_model == "new-small"
    finally:
        app.close()


def test_provider_specific_key_and_thinking_payload(monkeypatch):
    from lsm_harness.config import Settings
    from lsm_harness.models import get_client

    monkeypatch.delenv("LSM_API_KEY", raising=False)
    monkeypatch.delenv("WAKU_API_KEY", raising=False)
    monkeypatch.setenv("LSM_PROVIDER", "openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-only")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only")
    assert Settings().api_key == "openai-only"

    openai_client = get_client(provider_name="openai", api_key="openai-only")
    kwargs = openai_client._build_kwargs("model", "", [], [], 10)
    assert "extra_body" not in kwargs

    deepseek_client = get_client(provider_name="deepseek", api_key="deepseek-only")
    with deepseek_client.thinking_context("enabled"):
        kwargs = deepseek_client._build_kwargs("model", "", [], [], 10)
    assert kwargs["extra_body"]["thinking"]["type"] == "enabled"


def test_inline_subagent_does_not_mutate_parent_runtime(tmp_path):
    app = Harness(settings=_settings(tmp_path), client=ScriptedClient())
    session_id = app.session.session_id
    tools = app.tools
    try:
        reply = app.subagents.run_inline(
            "create a test event",
            harness=app,
            tools_whitelist=["list_dir"],
        )
        assert "冒烟命令已执行" in reply
        assert app.session.session_id == session_id
        assert app.tools is tools
        assert not app.is_running
    finally:
        app.close()


# ── batch E: unified model-call entry (plan §8.3) ────────────────


def _registry_events(reply: str):
    from lsm_harness.ai.api.common import snapshot
    from lsm_harness.ai.types import AssistantMessageEvent, ModelResponse

    final = ModelResponse(text=reply, stop_reason="stop")
    return [
        AssistantMessageEvent("start", snapshot(text="", thinking="", pending={})),
        AssistantMessageEvent("text_start", ModelResponse()),
        AssistantMessageEvent("text_delta", final, text_delta=reply),
        AssistantMessageEvent("text_end", final),
        AssistantMessageEvent("done", final),
    ]


def test_harness_main_chain_goes_through_provider_registry(tmp_path):
    """§8.3: with a registered Model.api, the Harness main chain streams
    through stream_simple + the ApiProvider registry — the injected legacy
    client only serves the memory gate, and needs no real API key."""
    from lsm_harness.ai.registry import (
        ApiProvider,
        register_api_provider,
        unregister_api_provider,
    )
    from lsm_harness.ai.types import Model

    calls: list[dict] = []

    def fake_stream(model, context, options):
        calls.append({"model": model, "options": options})
        yield from _registry_events("registry reply")

    register_api_provider(ApiProvider("test-registry-api", fake_stream))
    client = ScriptedClient()
    client.model = Model(id="fake-main", api="test-registry-api", provider="test")
    app = Harness(settings=_settings(tmp_path), client=client)
    try:
        result = app.respond("hello")
        assert result.reply == "registry reply"
        # the registry provider received the model and an options object
        # carrying the settings API key
        assert len(calls) == 1
        assert calls[0]["model"].id == "fake-main"
        assert calls[0]["options"].api_key == "scripted"
    finally:
        app.close()
        unregister_api_provider("test-registry-api")


def test_harness_accepts_directly_injected_stream_function(tmp_path):
    """§8.2: DI tests may inject a StreamFunction straight into Harness."""
    from lsm_harness.ai.stream import fixed_stream_function

    app = Harness(
        settings=_settings(tmp_path),
        client=ScriptedClient(),
        stream_fn=fixed_stream_function(_registry_events("injected reply")),
    )
    try:
        result = app.respond("hello")
        assert result.reply == "injected reply"
    finally:
        app.close()


# ── batch F-3: CLI renderer wiring (refactor plan §9.3) ──────────


def test_cli_observer_prefers_custom_renderer_over_label_and_name():
    from lsm_harness.coding_agent.cli import _make_observer_and_stream, console
    from lsm_harness.events import make_event

    renderers = {
        "exec": (
            lambda args: f"RUN {args.get('command', '')}",
            lambda output, _details: f"DONE({len(output)})",
        ),
    }
    observer, _display = _make_observer_and_stream(renderers)

    from io import StringIO
    capture = StringIO()
    original_file = console.file
    console._file = capture
    try:
        observer(make_event("tool.requested", "t", {
            "tool": "exec", "label": "Run command", "tool_call_id": "c1",
            "args": {"command": "ls"},
        }))
        observer(make_event("tool.completed", "t", {
            "tool": "exec", "label": "Run command", "tool_call_id": "c1",
            "status": "ok", "output": "file1\nfile2",
        }))
        # No renderer registered → falls back to label.
        observer(make_event("tool.requested", "t", {
            "tool": "mystery", "label": "Fancy label", "tool_call_id": "c2",
            "args": {},
        }))
    finally:
        console._file = original_file

    out = capture.getvalue()
    assert "RUN ls" in out
    assert "DONE(11)" in out
    assert "Fancy label" in out


def test_build_registry_collects_renderers(tmp_path, monkeypatch):
    from lsm_harness.coding_agent.tools import ToolDefinition
    from lsm_harness.tools import build_registry
    import lsm_harness.tools as tools_pkg

    fancy = ToolDefinition(
        name="fancy",
        description="",
        parameters={"type": "object", "properties": {}},
        execute=lambda: "ok",
        prompt_snippet="Use fancy.",
        render_call=lambda args: f"CALL {args}",
        render_result=lambda output, _d: f"RES {output}",
    )
    # Inject a renderer-bearing ToolDefinition through an existing maker.
    monkeypatch.setattr(
        tools_pkg.web,
        "make_tools",
        lambda **_kwargs: [fancy],
    )

    app = Harness(settings=_settings(tmp_path), client=ScriptedClient())
    try:
        renderers: dict = {}
        snippets: list[str] = []
        registry = build_registry(
            app.conn, app.settings, app.memory,
            subagent_manager=app.subagents,
            workspace_root=tmp_path,
            prompt_snippets=snippets,
            renderers=renderers,
        )
        assert "fancy" in registry.tool_names()
        render_call, render_result = renderers["fancy"]
        assert render_call({"x": 1}) == "CALL {'x': 1}"
        assert render_result("out", None) == "RES out"
        assert "Use fancy." in snippets
    finally:
        app.close()
