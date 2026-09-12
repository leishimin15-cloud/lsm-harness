"""Chapter 4 contracts for models, translators, and unified streams."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from lsm_harness.ai.api.anthropic_messages import (
    build_anthropic_request,
    stream_anthropic_client,
    translate_anthropic_messages,
)
from lsm_harness.ai.api.openai_compat import (
    _translate_openai_message,
    stream_openai_client,
)
from lsm_harness.ai.messages import (
    AssistantMessage,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
)
from lsm_harness.ai.registry import (
    ApiProvider,
    register_api_provider,
    unregister_api_provider,
)
from lsm_harness.ai.models import clamp_thinking_level
from lsm_harness.ai.providers import get_model
from lsm_harness.ai.stream import stream_simple
from lsm_harness.ai.types import (
    AIContext,
    AssistantMessageEvent,
    Model,
    ModelResponse,
    StreamOptions,
    Tool,
)


def _model(api: str = "test") -> Model:
    return Model(id="test-model", api=api, provider="test", max_tokens=1000)


def _context(messages=None) -> AIContext:
    return AIContext(
        system_prompt="system",
        messages=messages or [UserMessage(content="hello")],
        tools=[Tool(
            name="echo",
            description="echo",
            parameters={"type": "object", "properties": {}},
        )],
    )


def _delta(**values):
    defaults = {"content": None, "reasoning_content": None, "tool_calls": None}
    defaults.update(values)
    return SimpleNamespace(**defaults)


def _chunk(*, delta=None, finish_reason=None, usage=None):
    choice = SimpleNamespace(
        delta=delta or _delta(),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class _OpenAICompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return iter(self.chunks)


def test_openai_translator_emits_complete_tool_stream():
    tool_start = SimpleNamespace(
        index=0,
        id="call-1",
        function=SimpleNamespace(name="echo", arguments=""),
    )
    tool_args = SimpleNamespace(
        index=0,
        id=None,
        function=SimpleNamespace(name=None, arguments='{"value":"x"}'),
    )
    completions = _OpenAICompletions([
        _chunk(delta=_delta(reasoning_content="think")),
        _chunk(delta=_delta(content="answer")),
        _chunk(delta=_delta(tool_calls=[tool_start])),
        _chunk(delta=_delta(tool_calls=[tool_args]), finish_reason="tool_calls"),
    ])
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )

    events = list(stream_openai_client(
        client,
        _model("openai-completions"),
        _context(),
        StreamOptions(max_tokens=100),
    ))

    assert [event.kind for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "text_start",
        "text_delta",
        "toolcall_start",
        "toolcall_delta",
        "thinking_end",
        "text_end",
        "toolcall_end",
        "done",
    ]
    final = events[-1].partial
    assert final.text == "answer"
    assert final.thinking == "think"
    assert final.tool_calls[0].arguments == {"value": "x"}
    assert final.stop_reason == "tool_calls"


def test_openai_payload_and_response_hooks_are_applied():
    completions = _OpenAICompletions([
        _chunk(delta=_delta(content="ok"), finish_reason="stop"),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    seen = {}

    def on_payload(payload, model):
        seen["payload_model"] = model.id
        return {**payload, "temperature": 0}

    def on_response(response, model):
        seen["response"] = response
        seen["response_model"] = model.id

    events = list(stream_openai_client(
        client,
        _model("openai-completions"),
        _context(),
        StreamOptions(
            max_tokens=100,
            on_payload=on_payload,
            on_response=on_response,
        ),
    ))

    assert events[-1].partial.text == "ok"
    assert completions.request["temperature"] == 0
    assert seen["payload_model"] == "test-model"
    assert seen["response_model"] == "test-model"
    assert seen["response"]["status"] == 200


def test_anthropic_message_translation_handles_tool_round_trip():
    translated = translate_anthropic_messages([
        UserMessage(content="run echo"),
        AssistantMessage(
            tool_calls=(
                ToolCallContent(id="call-1", name="echo", arguments={"value": "x"}),
            ),
        ),
        ToolResultMessage(tool_call_id="call-1", content="x", is_error=True),
    ])

    assert translated[1]["content"][0] == {
        "type": "tool_use",
        "id": "call-1",
        "name": "echo",
        "input": {"value": "x"},
    }
    assert translated[2]["role"] == "user"
    assert translated[2]["content"][0]["type"] == "tool_result"
    assert translated[2]["content"][0]["tool_use_id"] == "call-1"
    assert translated[2]["content"][0]["is_error"] is True


class _AnthropicStream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return iter(self.events)

    def __exit__(self, *_args):
        return False


class _AnthropicMessages:
    def __init__(self, events):
        self.events = events
        self.request = None

    def stream(self, **kwargs):
        self.request = kwargs
        return _AnthropicStream(self.events)


def test_anthropic_translator_preserves_streamed_tool_call():
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=7)),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(
                type="tool_use",
                id="tool-1",
                name="echo",
                input={},
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(
                type="input_json_delta",
                partial_json='{"value":"hello"}',
            ),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"),
            usage=SimpleNamespace(output_tokens=5),
        ),
    ]
    messages = _AnthropicMessages(events)
    client = SimpleNamespace(messages=messages)

    translated = list(stream_anthropic_client(
        client,
        _model("anthropic-messages"),
        _context(),
        StreamOptions(max_tokens=100),
    ))

    assert [event.kind for event in translated] == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    final = translated[-1].partial
    assert final.tool_calls[0].name == "echo"
    assert final.tool_calls[0].arguments == {"value": "hello"}
    assert final.usage.input_tokens == 7
    assert final.usage.output_tokens == 5


def test_thinking_and_cache_are_translated_by_model_capabilities():
    model = Model(
        id="claude",
        api="anthropic-messages",
        provider="anthropic",
        max_tokens=500,
        thinking_level_map={"off": None, "high": "16384"},
        cache_control_format="anthropic",
        supports_long_cache_retention=False,
    )
    request = build_anthropic_request(
        model,
        _context(),
        StreamOptions(
            max_tokens=1000,
            reasoning="high",
            cache_retention="short",
        ),
    )

    assert request["max_tokens"] == 1000
    assert request["thinking"]["budget_tokens"] == 16384
    assert request["system"][0]["cache_control"]["type"] == "ephemeral"
    assert request["tools"][-1]["cache_control"]["type"] == "ephemeral"


def test_stream_simple_clamps_capabilities_before_dispatch():
    captured = []

    def translator(model, _context, options):
        captured.append(options)
        partial = ModelResponse(text="ok")
        yield AssistantMessageEvent("start", ModelResponse())
        yield AssistantMessageEvent("done", partial)

    api = "test-capabilities"
    register_api_provider(ApiProvider(api, translator))
    try:
        model = Model(
            id="m",
            api=api,
            provider="p",
            max_tokens=200,
            reasoning=True,
            thinking_level_map={
                "off": None,
                "minimal": None,
                "low": None,
                "medium": None,
                "high": "high",
            },
            cache_control_format="anthropic",
            supports_long_cache_retention=False,
        )
        events = list(stream_simple(
            model,
            _context(),
            StreamOptions(
                max_tokens=500,
                reasoning="low",
                cache_retention="long",
            ),
        ))
    finally:
        unregister_api_provider(api)

    assert events[-1].partial.text == "ok"
    assert captured[0].max_tokens == 200
    assert captured[0].reasoning == "high"
    assert captured[0].cache_retention == "short"


def test_stream_simple_reserves_tokens_for_anthropic_thinking():
    captured = []

    def translator(_model, _context, options):
        captured.append(options)
        yield AssistantMessageEvent("done", ModelResponse(text="ok"))

    api = "test-thinking-budget"
    register_api_provider(ApiProvider(api, translator))
    try:
        model = Model(
            id="m",
            api=api,
            provider="p",
            reasoning=True,
            thinking_level_map={"off": None, "high": "16384"},
            thinking_format="anthropic",
        )
        list(stream_simple(
            model,
            _context(),
            StreamOptions(max_tokens=8192, reasoning="high"),
        ))
    finally:
        unregister_api_provider(api)

    assert captured[0].max_tokens == 17408


def test_explicit_thinking_budget_overrides_model_default():
    model = Model(
        id="claude",
        api="anthropic-messages",
        provider="anthropic",
        thinking_level_map={"off": None, "high": "16384"},
        thinking_format="anthropic",
    )
    request = build_anthropic_request(
        model,
        _context(),
        StreamOptions(
            max_tokens=8192,
            reasoning="high",
            thinking_budgets={"high": 4096},
        ),
    )
    assert request["thinking"]["budget_tokens"] == 4096


def test_usage_exposes_cache_totals_and_cost():
    from lsm_harness.ai.types import Usage

    model = Model(
        id="priced",
        api="test",
        provider="test",
        input_cost_per_million=2,
        output_cost_per_million=4,
        cache_read_cost_per_million=1,
        cache_write_cost_per_million=3,
    )
    usage = Usage(1000, 500, 200, 100).with_model_cost(model)
    assert usage.total_tokens == 1800
    assert usage.cost_total == 0.0045


def test_stream_simple_retries_only_before_semantic_output(monkeypatch):
    calls = 0

    def translator(_model, _context, _options):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield AssistantMessageEvent("start", ModelResponse())
            yield AssistantMessageEvent(
                "error",
                ModelResponse(
                    stop_reason="error",
                    error_message="connection reset",
                ),
                error_category="transient",
            )
            return
        yield AssistantMessageEvent("start", ModelResponse())
        yield AssistantMessageEvent("done", ModelResponse(text="recovered"))

    stream_module = importlib.import_module("lsm_harness.ai.stream")
    monkeypatch.setattr(stream_module.time, "sleep", lambda _delay: None)
    api = "test-retry"
    register_api_provider(ApiProvider(api, translator))
    try:
        events = list(stream_simple(
            _model(api),
            _context(),
            StreamOptions(max_tokens=100, max_retries=1),
        ))
    finally:
        unregister_api_provider(api)

    assert calls == 2
    assert [event.kind for event in events] == ["start", "done"]
    assert events[-1].partial.text == "recovered"


def test_stream_simple_does_not_retry_after_text_started(monkeypatch):
    calls = 0

    def translator(_model, _context, _options):
        nonlocal calls
        calls += 1
        yield AssistantMessageEvent("start", ModelResponse())
        yield AssistantMessageEvent("text_start", ModelResponse())
        yield AssistantMessageEvent(
            "text_delta",
            ModelResponse(text="partial"),
            text_delta="partial",
        )
        yield AssistantMessageEvent(
            "error",
            ModelResponse(
                text="partial",
                stop_reason="error",
                error_message="connection reset",
            ),
            error_category="transient",
        )

    stream_module = importlib.import_module("lsm_harness.ai.stream")
    monkeypatch.setattr(stream_module.time, "sleep", lambda _delay: None)
    api = "test-no-retry-after-output"
    register_api_provider(ApiProvider(api, translator))
    try:
        events = list(stream_simple(
            _model(api),
            _context(),
            StreamOptions(max_tokens=100, max_retries=2),
        ))
    finally:
        unregister_api_provider(api)

    assert calls == 1
    assert events[-1].kind == "error"
    assert events[-1].partial.text == "partial"


def test_stream_simple_interrupts_retry_backoff_before_next_request():
    import threading

    interrupt = threading.Event()
    calls = 0

    def translator(_model, _context, _options):
        nonlocal calls
        calls += 1
        yield AssistantMessageEvent("start", ModelResponse())
        yield AssistantMessageEvent(
            "error",
            ModelResponse(stop_reason="error", error_message="connection reset"),
            error_category="transient",
        )

    def on_retry(_attempt, _category, _message):
        interrupt.set()

    api = "test-interrupt-retry-backoff"
    register_api_provider(ApiProvider(api, translator))
    try:
        events = list(stream_simple(
            _model(api),
            _context(),
            StreamOptions(
                max_tokens=100,
                max_retries=2,
                interrupt=interrupt,
                on_retry=on_retry,
            ),
        ))
    finally:
        unregister_api_provider(api)

    assert calls == 1
    assert [event.kind for event in events] == ["error"]
    assert events[0].partial.stop_reason == "aborted"


def test_stream_simple_ignores_provider_events_after_terminal_event():
    def translator(_model, _context, _options):
        yield AssistantMessageEvent("start", ModelResponse())
        yield AssistantMessageEvent("done", ModelResponse(text="done"))
        yield AssistantMessageEvent(
            "text_delta",
            ModelResponse(text="must not leak"),
            text_delta="must not leak",
        )

    api = "test-terminal-boundary"
    register_api_provider(ApiProvider(api, translator))
    try:
        events = list(stream_simple(
            _model(api),
            _context(),
            StreamOptions(max_tokens=100),
        ))
    finally:
        unregister_api_provider(api)

    assert [event.kind for event in events] == ["start", "done"]
    assert events[-1].partial.text == "done"


def test_translator_runtime_error_is_encoded_not_raised():
    class BrokenCompletions:
        def create(self, **_kwargs):
            raise ConnectionError("offline")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=BrokenCompletions()),
    )
    events = list(stream_openai_client(
        client,
        _model("openai-completions"),
        _context(),
        StreamOptions(max_tokens=100),
    ))

    assert events[-1].kind == "error"
    assert events[-1].partial.stop_reason == "error"
    assert events[-1].error_category == "transient"


def test_get_model_populates_token_limits_from_provider_catalog():
    from lsm_harness.ai.providers import get_model

    model = get_model("anthropic")
    assert model.context_window == 200_000
    assert model.max_tokens == 64_000

    fallback = get_model("deepseek")
    assert fallback.context_window > 0
    assert fallback.max_tokens > 0


def test_stream_simple_uses_catalog_limits_when_clamping():
    from lsm_harness.ai.providers import get_model

    captured = []

    def translator(_model, _context, options):
        captured.append(options)
        yield AssistantMessageEvent("done", ModelResponse(text="ok"))

    api = "anthropic-messages"
    register_api_provider(ApiProvider(api, translator))
    try:
        model = get_model("anthropic")
        list(stream_simple(
            model,
            _context(),
            StreamOptions(max_tokens=200_000, reasoning="off"),
        ))
    finally:
        # Restore the built-in — plain unregister would delete it for the
        # rest of the session (module-level registration is global state).
        from lsm_harness.ai.api.anthropic_messages import stream_anthropic_messages

        register_api_provider(ApiProvider(api, stream_anthropic_messages))

    assert captured[0].max_tokens == 64_000


# ── batch A: strict typed boundary (refactor plan §4.7) ───────────


def test_anthropic_translator_round_trips_signed_thinking():
    """Signed reasoning stays thinking; unsigned reasoning becomes text."""
    signed = AssistantMessage(thinking="chain", thinking_signature="sig-1", text="answer")
    unsigned = AssistantMessage(thinking="chain", text="answer")

    translated = translate_anthropic_messages([signed, unsigned])
    assistant_blocks = [
        block
        for message in translated
        if message["role"] == "assistant"
        for block in message["content"]
    ]
    thinking_blocks = [b for b in assistant_blocks if b["type"] == "thinking"]
    assert thinking_blocks == [{
        "type": "thinking",
        "thinking": "chain",
        "signature": "sig-1",
    }]
    assert any(
        block == {"type": "text", "text": "chain"}
        for block in assistant_blocks
    )


def test_kimi_coding_anthropic_compat_matches_pi():
    model = get_model("kimi-coding", "k3")
    assert clamp_thinking_level(model, "off") == "low"
    request = build_anthropic_request(
        get_model("kimi-coding", "kimi-for-coding"),
        AIContext(
            system_prompt="",
            messages=[AssistantMessage(thinking="chain", text="answer")],
            tools=[],
        ),
        StreamOptions(max_tokens=1000, reasoning="medium"),
    )

    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "medium"}
    assert request["messages"][0]["content"][0] == {
        "type": "thinking",
        "thinking": "chain",
        "signature": "",
    }


def test_openai_translator_does_not_echo_reasoning():
    message = AssistantMessage(thinking="chain", thinking_signature="sig", text="answer")
    wire = _translate_openai_message(message)
    assert "reasoning_content" not in wire
    assert "thinking" not in wire
    assert wire["content"] == "answer"


def test_translators_reject_non_standard_messages():
    import pytest

    legacy_dict = {"role": "user", "content": "hello"}
    with pytest.raises(TypeError, match="unsupported message"):
        translate_anthropic_messages([legacy_dict])
    with pytest.raises(TypeError, match="unsupported message"):
        _translate_openai_message(legacy_dict)


def test_anthropic_stream_captures_thinking_signature():
    events = [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="thinking", thinking=""),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="thinking_delta", thinking="chain"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="signature_delta", signature="sig-42"),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=5),
        ),
    ]
    messages = _AnthropicMessages(events)
    client = SimpleNamespace(messages=messages)

    translated = list(stream_anthropic_client(
        client,
        _model("anthropic-messages"),
        _context(),
        StreamOptions(max_tokens=100),
    ))

    final = translated[-1].partial
    assert final.thinking == "chain"
    assert final.thinking_signature == "sig-42"


def test_unregistered_api_fails_with_structured_error():
    """§8.3: an unknown Model.api fails fast, listing what IS registered."""
    import pytest

    import lsm_harness.ai.providers  # noqa: F401 — registers the built-ins
    from lsm_harness.ai.registry import resolve_api_provider

    with pytest.raises(ValueError, match="Unknown model API 'no-such-api'"):
        resolve_api_provider("no-such-api")
    # the message doubles as documentation: it names the registered APIs
    try:
        resolve_api_provider("no-such-api")
    except ValueError as exc:
        assert "openai-completions" in str(exc)
        assert "anthropic-messages" in str(exc)
