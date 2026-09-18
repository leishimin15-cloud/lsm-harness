"""Request-only replay normalization aligned with Pi transformMessages."""

from lsm_harness.ai.api.transform_messages import transform_messages
from lsm_harness.ai.messages import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
)
from lsm_harness.ai.types import Model


def _model(*, provider: str = "target", images: bool = False) -> Model:
    return Model(
        id="target-model",
        api="target-api",
        provider=provider,
        input_modalities=("text", "image") if images else ("text",),
    )


def test_cross_model_reasoning_becomes_plain_text_and_signature_is_removed():
    source = AssistantMessage(
        text="answer",
        thinking="private reasoning",
        thinking_signature="source-only-signature",
        provider="source",
        api="source-api",
        model="source-model",
    )

    transformed = transform_messages([source], _model())

    assert transformed == [AssistantMessage(
        text="private reasoning\nanswer",
        provider="source",
        api="source-api",
        model="source-model",
    )]


def test_same_model_preserves_signed_reasoning_and_original_tool_call_id():
    source = AssistantMessage(
        thinking="reasoning",
        thinking_signature="signature",
        tool_calls=(ToolCallContent("unsafe|but-same-model", "read", {}),),
        provider="target",
        api="target-api",
        model="target-model",
    )

    transformed = transform_messages([source], _model())

    assert transformed[0] == source
    assert isinstance(transformed[1], ToolResultMessage)
    assert transformed[1].tool_call_id == "unsafe|but-same-model"


def test_cross_model_tool_id_is_normalized_and_result_pairing_is_preserved():
    source = AssistantMessage(
        tool_calls=(ToolCallContent("call|with unsafe chars", "read", {}),),
        provider="source",
        api="source-api",
        model="source-model",
    )
    tool_result = ToolResultMessage(
        tool_call_id="call|with unsafe chars",
        tool_name="read",
        content="ok",
    )

    transformed = transform_messages([source, tool_result], _model())

    assistant = transformed[0]
    result = transformed[1]
    assert isinstance(assistant, AssistantMessage)
    assert isinstance(result, ToolResultMessage)
    assert assistant.tool_calls[0].id != "call|with unsafe chars"
    assert result.tool_call_id == assistant.tool_calls[0].id


def test_non_vision_model_collapses_consecutive_images_to_one_placeholder():
    message = UserMessage(content=(
        ImageContent("data:image/png;base64,a"),
        ImageContent("data:image/png;base64,b"),
        TextContent("caption"),
        ImageContent("data:image/png;base64,c"),
    ))

    transformed = transform_messages([message], _model())

    assert transformed == [UserMessage(content=(
        TextContent("(image omitted: model does not support images)"),
        TextContent("caption"),
        TextContent("(image omitted: model does not support images)"),
    ))]


def test_orphan_tool_call_gets_request_only_synthetic_error_result():
    assistant = AssistantMessage(
        tool_calls=(ToolCallContent("call-1", "read", {}),),
    )
    user = UserMessage("continue")

    transformed = transform_messages([assistant, user], _model())

    assert transformed[0] == assistant
    assert transformed[1] == ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content="No result provided",
        is_error=True,
    )
    assert transformed[2] is user


def test_error_and_aborted_assistant_messages_are_not_replayed():
    messages = [
        UserMessage("start"),
        AssistantMessage(text="partial", stop_reason="error"),
        AssistantMessage(text="partial", stop_reason="aborted"),
        UserMessage("retry"),
    ]

    assert transform_messages(messages, _model()) == [messages[0], messages[3]]


def test_loose_dict_messages_are_coerced_for_one_shot_callers():
    """压缩摘要/分支摘要/ModelJudge 传 dict 消息;anthropic 路径必须接受,
    否则 complete() TypeError → 手动压缩静默 skip(eval 实测,2026-09)。"""
    transformed = transform_messages(
        [{"role": "user", "content": "总结一下"}],
        _model(),
    )
    assert transformed == [UserMessage("总结一下")]

    # 不认识的 dict 形状保持原样,仍由后续类型检查拒绝
    try:
        transform_messages([{"role": "tool", "content": "x"}], _model())
    except TypeError:
        pass
    else:
        raise AssertionError("unexpected dict shape must still raise")
