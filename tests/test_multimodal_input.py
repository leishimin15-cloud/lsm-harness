"""批次 R-3: 多模态输入必须在产品入口规范化为内部消息块。

问题原因：``prepare_context`` 把整条消息字典直接赋给 ``UserMessage.content``，
而 Provider translator 只接受 ``str | tuple[TextContent | ImageContent]`` →
图片输入在 OpenAI 请求构造阶段报 ``unsupported user content block: str``。

验收：纯文本不回归；文字+图片经 prepare_context → OpenAI/Anthropic
translator 后请求结构正确；不支持的块类型给出清晰错误且不泄漏图片载荷；
持久化仍走 ``[image]`` 占位符，与发给模型的载荷分离。
"""

from __future__ import annotations

import pytest

from lsm_harness.ai.api.anthropic_messages import build_anthropic_request
from lsm_harness.ai.api.openai_compat import build_openai_request
from lsm_harness.ai.messages import ImageContent, TextContent, UserMessage
from lsm_harness.ai.types import AIContext, Model, StreamOptions
from lsm_harness.coding_agent.session import Session
from lsm_harness.config import Settings
from lsm_harness.db import connect

IMAGE_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"

PAYLOAD = {
    "role": "user",
    "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
    ],
}


@pytest.fixture
def session(tmp_path, monkeypatch):
    settings = Settings(
        home=tmp_path,
        api_key="dummy",
        provider="deepseek",
        model="dummy",
        small_model="dummy",
        base_url="",
        context_budget_tokens=10000,
        context_reserve_tokens=1000,
    )
    conn = connect(tmp_path)
    value = Session(settings, conn=conn, client=None)
    monkeypatch.setattr(value, "build_system", lambda *_: "")
    try:
        yield value
    finally:
        conn.close()


def _prepare(session, user_message):
    _, _hist, _cur = session.prepare_context(user_message, lambda *_: None, [])
    messages = [*_hist, *([_cur] if _cur is not None else [])]
    return messages[-1]


def test_text_only_input_unchanged(session):
    current = _prepare(session, "纯文本")
    assert isinstance(current, UserMessage)
    assert current.content == "纯文本"

    request = build_openai_request(
        Model(id="m", api="openai", provider="openai"),
        AIContext(system_prompt="", messages=[current], tools=[]),
        StreamOptions(max_tokens=100),
    )
    assert request["messages"][-1]["content"] == "纯文本"


def test_text_and_image_normalized_then_openai_request_intact(session):
    current = _prepare(session, PAYLOAD)
    assert isinstance(current.content, tuple)
    assert current.content == (
        TextContent(text="describe"),
        ImageContent(url=IMAGE_URL),
    )

    request = build_openai_request(
        Model(
            id="m",
            api="openai",
            provider="openai",
            input_modalities=("text", "image"),
        ),
        AIContext(system_prompt="", messages=[current], tools=[]),
        StreamOptions(max_tokens=100),
    )
    assert request["messages"][-1]["content"] == PAYLOAD["content"]


def test_text_and_image_anthropic_request_intact(session):
    current = _prepare(session, PAYLOAD)
    request = build_anthropic_request(
        Model(
            id="m",
            api="anthropic",
            provider="anthropic",
            input_modalities=("text", "image"),
        ),
        AIContext(system_prompt="", messages=[current], tools=[]),
        StreamOptions(max_tokens=100),
    )
    blocks = request["messages"][-1]["content"]
    assert blocks[0] == {"type": "text", "text": "describe"}
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"]["type"] == "base64"
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert blocks[1]["source"]["data"] == "iVBORw0KGgoAAAANSUhEUg"


def test_unsupported_block_type_is_a_clear_error(session):
    payload = {"role": "user", "content": [{"type": "video", "source": "x"}]}
    with pytest.raises(ValueError, match="video"):
        session.prepare_context(payload, lambda *_: None, [])


def test_malformed_image_block_is_a_clear_error(session):
    payload = {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
    with pytest.raises(ValueError, match="image_url"):
        session.prepare_context(payload, lambda *_: None, [])


def test_error_messages_never_embed_image_payload(session):
    payload = {"role": "user", "content": [{"type": "audio", "data": IMAGE_URL}]}
    with pytest.raises(ValueError) as excinfo:
        session.prepare_context(payload, lambda *_: None, [])
    assert IMAGE_URL not in str(excinfo.value)


def test_persistence_placeholder_stays_separate_from_wire_payload(session):
    persisted = session.persistable_user_message(PAYLOAD)
    assert persisted.content == "describe\n[image]"
    assert IMAGE_URL not in persisted.content
