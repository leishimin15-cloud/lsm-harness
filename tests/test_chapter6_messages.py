"""Chapter 6: two-tier typed messages — constructors, registry, convert_to_llm.

Batch A rewrite: the inner tier is a typed union (UserMessage /
AssistantMessage / ToolResultMessage / CustomMessage); the LLM boundary
is an exhaustive whitelist — unknown shapes raise instead of leaking.
"""

import pytest

from lsm_harness.agent.messages import (
    AgentToolResultMessage,
    AssistantMessage,
    CustomMessage,
    CustomMessageType,
    TextContent,
    ToolCallContent,
    ToolResultMessage as AiToolResultMessage,
    UserMessage,
    assistant_message,
    clear_custom_message_registry,
    custom_message,
    default_convert_to_llm,
    get_custom_message_type,
    is_custom_message,
    is_excluded_from_context,
    message_from_legacy,
    message_preview,
    message_to_legacy,
    register_custom_message_type,
    tool_result_message,
    user_message,
)
from lsm_harness.agent.tools import ToolResultMessage


@pytest.fixture(autouse=True)
def clean_registry():
    clear_custom_message_registry()
    yield
    clear_custom_message_registry()


# ── typed constructors ────────────────────────────────────────────


def test_user_message_shape():
    assert user_message("hello") == UserMessage(content="hello")
    assert user_message("hello").role == "user"


def test_user_message_content_blocks():
    message = user_message([
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
    ])
    assert isinstance(message.content, tuple)
    assert message.content[0] == TextContent(text="look")
    assert message.content[1].url == "data:image/png;base64,AA"


def test_assistant_message_minimal_and_full():
    assert assistant_message("hi") == AssistantMessage(text="hi")

    calls = [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
    message = assistant_message("let me look", thinking="hmm", tool_calls=calls)
    assert message.role == "assistant"
    assert message.text == "let me look"
    assert message.thinking == "hmm"
    assert message.tool_calls == (
        ToolCallContent(id="c1", name="read", arguments={}),
    )


def test_assistant_message_accepts_typed_tool_calls():
    call = ToolCallContent(id="c2", name="write", arguments={"path": "a.py"})
    message = assistant_message(tool_calls=[call])
    assert message.tool_calls == (call,)


def test_tool_result_message_folds_envelope():
    envelope = ToolResultMessage(
        output="file contents",
        is_error=False,
        terminate=False,
        details={"path": "a.py"},
        tool_call_id="c1",
        tool_name="read",
    )
    message = tool_result_message(envelope)
    assert message == AgentToolResultMessage(
        tool_call_id="c1",
        tool_name="read",
        content="file contents",
        is_error=False,
        details={"path": "a.py"},
        terminate=False,
    )
    assert message.role == "tool"


def test_custom_message_shape_and_flag():
    message = custom_message("bash_execution", "ls output", exclude_from_context=True, exit_code=0)
    assert message.role == "custom"
    assert message.custom_type == "bash_execution"
    assert message.exclude_from_context is True
    assert message.fields == {"exit_code": 0}
    assert is_custom_message(message)
    assert not is_custom_message(user_message("x"))


# ── custom message registry ───────────────────────────────────────


def test_registry_register_and_lookup():
    spec = CustomMessageType(name="note", to_llm=lambda m: UserMessage(content=m.content))
    register_custom_message_type(spec)
    assert get_custom_message_type("note") == spec
    assert get_custom_message_type("missing") is None


def test_registry_rejects_duplicate_registration():
    register_custom_message_type(CustomMessageType(name="note"))
    with pytest.raises(ValueError, match="already registered"):
        register_custom_message_type(CustomMessageType(name="note"))


def test_message_level_exclude_overrides_type_default():
    register_custom_message_type(
        CustomMessageType(name="secret", exclude_from_context=True)
    )
    type_default = custom_message("secret", "s")
    assert is_excluded_from_context(type_default) is True

    message_override = custom_message("secret", "s", exclude_from_context=False)
    assert is_excluded_from_context(message_override) is False


# ── default_convert_to_llm: the exhaustive whitelist edge ─────────


def test_convert_keeps_standard_roles_and_order():
    messages = [
        user_message("u1"),
        assistant_message("a1"),
        tool_result_message(ToolResultMessage(output="t1", tool_call_id="c", tool_name="read")),
        user_message("u2"),
    ]
    converted = default_convert_to_llm(messages)
    assert [m.role for m in converted] == ["user", "assistant", "tool", "user"]
    assert converted[0] == UserMessage(content="u1")
    assert converted[2].content == "t1"


def test_convert_strips_agent_only_fields():
    """details / terminate never cross the LLM boundary (field leakage)."""
    message = tool_result_message(
        ToolResultMessage(output="t", details={"x": 1}, terminate=True, tool_call_id="c", tool_name="read")
    )
    converted = default_convert_to_llm([message])
    assert len(converted) == 1
    # the base type, exactly — no AgentToolResultMessage extras survive
    assert type(converted[0]) is AiToolResultMessage
    assert converted[0].tool_call_id == "c"
    assert converted[0].is_error is False
    assert not hasattr(converted[0], "details") or converted[0].__class__ is AiToolResultMessage


def test_convert_filters_excluded_and_rejects_unknown_types():
    messages = [
        custom_message("note", "visible note"),
        custom_message("note", "hidden note", exclude_from_context=True),
        user_message("u"),
    ]
    register_custom_message_type(
        CustomMessageType(name="note", to_llm=lambda m: UserMessage(content=m.content))
    )
    converted = default_convert_to_llm(messages)
    assert [m.content for m in converted] == ["visible note", "u"]
    assert all(isinstance(m, (UserMessage, AssistantMessage, AiToolResultMessage)) for m in converted)


def test_convert_custom_uses_registered_translator():
    register_custom_message_type(
        CustomMessageType(
            name="summary",
            to_llm=lambda m: UserMessage(content=f"<summary>{m.content}</summary>"),
        )
    )
    converted = default_convert_to_llm([custom_message("summary", "we did X")])
    assert converted == [UserMessage(content="<summary>we did X</summary>")]


def test_convert_unregistered_custom_is_an_error():
    """Batch A breaking change: no silent fallback at the LLM boundary."""
    with pytest.raises(ValueError, match="no LLM translator registered"):
        default_convert_to_llm([custom_message("anything", "payload")])


def test_convert_illegal_translator_return_is_an_error():
    register_custom_message_type(
        CustomMessageType(name="bad", to_llm=lambda m: {"role": "user", "content": "x"})
    )
    with pytest.raises(TypeError, match="illegal message type"):
        default_convert_to_llm([custom_message("bad", "x")])


def test_convert_unknown_message_type_is_an_error():
    with pytest.raises(TypeError, match="unknown AgentMessage type"):
        default_convert_to_llm([{"role": "user", "content": "dict leaked"}])


# ── legacy adapters (batch A dict edges: session / chat_log / TUI) ─


def test_legacy_round_trip_assistant_with_tool_calls_and_thinking():
    legacy = {
        "role": "assistant",
        "content": "let me look",
        "thinking": "hmm",
        "thinking_signature": "sig-1",
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read", "arguments": '{"path": "a.py"}'}}
        ],
    }
    typed = message_from_legacy(legacy)
    assert isinstance(typed, AssistantMessage)
    assert typed.thinking_signature == "sig-1"
    assert typed.tool_calls[0].arguments == {"path": "a.py"}

    wire = message_to_legacy(typed)
    assert wire["role"] == "assistant"
    assert wire["thinking_signature"] == "sig-1"
    assert wire["tool_calls"][0]["function"]["arguments"] == '{"path": "a.py"}'


def test_legacy_round_trip_tool_result_and_custom():
    tool_legacy = message_to_legacy(AgentToolResultMessage(
        tool_call_id="c1", tool_name="read", content="out",
        is_error=True, details={"path": "a.py"}, terminate=True,
    ))
    assert tool_legacy["details"] == {"path": "a.py"}
    assert tool_legacy["terminate"] is True
    typed = message_from_legacy(tool_legacy)
    assert isinstance(typed, AgentToolResultMessage)
    assert typed.terminate is True

    custom_legacy = message_to_legacy(
        custom_message("note", "hi", exit_code=0, exclude_from_context=True)
    )
    assert custom_legacy["role"] == "custom"
    assert custom_legacy["custom_type"] == "note"
    assert custom_legacy["exit_code"] == 0
    typed_custom = message_from_legacy(custom_legacy)
    assert typed_custom.fields == {"exit_code": 0}
    assert typed_custom.exclude_from_context is True


def test_message_preview_handles_all_tiers():
    assert message_preview(user_message("hello")) == "hello"
    assert message_preview(assistant_message("answer")) == "answer"
    assert message_preview(
        tool_result_message(ToolResultMessage(output="t", tool_call_id="c", tool_name="read"))
    ) == "t"
    assert message_preview(custom_message("note", "custom text")) == "custom text"
    assert message_preview(user_message([{"type": "text", "text": "block"}])) == "block"


# ── coding_agent custom message registration ─────────────────────

from lsm_harness.coding_agent.messages import (
    COMPACTION_SUMMARY,
    compaction_summary_message,
    register_coding_agent_messages,
)


def test_compaction_summary_message_shape():
    message = compaction_summary_message("我们完成了 X", through_chat_id=42, version=3)
    assert message.role == "custom"
    assert message.custom_type == COMPACTION_SUMMARY
    assert message.content == "我们完成了 X"
    assert message.fields == {"through_chat_id": 42, "version": 3}
    # summaries must stay visible to the LLM
    assert is_excluded_from_context(message) is False


def test_register_coding_agent_messages_is_idempotent():
    register_coding_agent_messages()
    register_coding_agent_messages()  # second Harness() construction must not raise
    spec = get_custom_message_type(COMPACTION_SUMMARY)
    assert spec is not None

    message = compaction_summary_message("we did X", through_chat_id=7, version=2)
    expected = (
        "The conversation history before this point was compacted "
        "into the following summary:\n\n<summary>\nwe did X\n</summary>"
    )
    assert spec.to_llm(message) == UserMessage(content=expected)
    assert spec.render(message) == "🗜 摘要（覆盖至 #7）"

    converted = default_convert_to_llm([message])
    assert converted == [UserMessage(content=expected)]


def test_branch_summary_message_registration():
    from lsm_harness.coding_agent.messages import (
        BRANCH_SUMMARY,
        branch_summary_message,
    )

    register_coding_agent_messages()
    spec = get_custom_message_type(BRANCH_SUMMARY)
    assert spec is not None

    message = branch_summary_message("tried triggers, too slow", from_id="abc123")
    assert message.role == "custom"
    assert message.custom_type == BRANCH_SUMMARY
    assert message.fields == {"from_id": "abc123"}
    assert is_excluded_from_context(message) is False

    llm = spec.to_llm(message)
    assert isinstance(llm, UserMessage)
    assert "explored a different conversation branch" in llm.content
    assert "<summary>" in llm.content
    assert "tried triggers, too slow" in llm.content
