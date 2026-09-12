"""阶段 A 批 3:Harness 的 prompt 摄入走 kernel 事件。

钉住:
- user 消息经 listener 落树**恰好一次**(显式 recorder.record 已删,
  一事件=一条目不变量彻底化);
- prompt 路径不产生 loop.steered / loop.followed_up / loop.user
  字符串事件(source="user" 闸门);
- 首轮模型调用前的 instant-abort 仍把问题留在树里(initial_pending
  注入已前移到 abort 检查之前);
- 多模态:typed persist transform 与 persistable_user_message 序列化
  等价——树条目带 [image] 占位符,不含 base64。
"""

from __future__ import annotations

import json

from lsm_harness.ai.types import ModelResponse, Usage
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect

from helpers import QueueClient


def _app(tmp_path, client: QueueClient) -> Harness:
    settings = Settings(api_key="scripted", home=tmp_path)
    conn = connect(tmp_path, check_same_thread=False)
    return Harness(
        settings=settings,
        client=client,
        conn=conn,
        stream_fn=client.as_stream_fn(),
    )


def _read_tree_entries(app) -> list[dict]:
    path = app.session.jsonl_path
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _user_entries(app) -> list[dict]:
    return [
        e for e in _read_tree_entries(app)
        if e.get("type") == "message" and e.get("message", {}).get("role") == "user"
    ]


def test_user_message_recorded_exactly_once_via_events(tmp_path):
    """respond 一次,树里 user 条目恰好一条(显式 record 已删;若恢复
    显式 record 会变成两条——反向变异①钉这里)。"""
    client = QueueClient(ModelResponse(text="答", usage=Usage(1, 1)))
    app = _app(tmp_path, client)
    try:
        result = app.respond("只问一次", source="test")
        assert result.status == "completed"
        users = _user_entries(app)
        assert len(users) == 1
        assert users[0]["message"]["content"] == "只问一次"
        # source 统一为 kernel 事件通道的 "user"(与 steering/follow_up 同约定)。
        assert users[0]["source"] == "user"
    finally:
        app.close()


def test_prompt_path_emits_no_loop_string_events(tmp_path):
    """prompt 摄入经 initial_pending(source="user")通道,但字符串层
    必须沉默:loop.steered/loop.followed_up 只属于队列消息,也不允许
    出现 loop.user 新事件(反向变异②钉这里)。"""
    client = QueueClient(ModelResponse(text="答", usage=Usage(1, 1)))
    app = _app(tmp_path, client)
    kinds: list[str] = []
    try:
        result = app.respond(
            "正事", source="test",
            observer=lambda e: kinds.append(e.type),
        )
        assert result.status == "completed"
        assert "loop.steered" not in kinds
        assert "loop.followed_up" not in kinds
        assert "loop.user" not in kinds
    finally:
        app.close()


def test_instant_abort_still_keeps_question_in_tree(tmp_path):
    """begin 后、worker 开跑前就 abort:注入已前移到 abort 检查之前,
    问题必须仍在树里(否则 continue 无从答起)。这是注入前移的驱动钉测。"""
    client = QueueClient(ModelResponse(text="不应出现", usage=Usage(1, 1)))
    app = _app(tmp_path, client)
    try:
        active = app.begin_run()
        app.abort()  # 秒中断:loop 起跑即见 interrupt
        result = app.respond("被秒中断的问题", source="test", active_run=active)
        assert result.status == "aborted"
        users = _user_entries(app)
        assert any("被秒中断的问题" in str(u["message"].get("content")) for u in users)
        assert client.calls == []  # 模型从未被调用
    finally:
        app.close()


def test_multimodal_persist_transform_matches_legacy_placeholder(tmp_path):
    """typed persist transform 的产物与旧 persistable_user_message 序列化
    等价:文本拼接、图片变 [image]、树条目不含 base64。"""
    from lsm_harness.agent.messages import UserMessage

    payload = {
        "role": "user",
        "content": [
            {"type": "text", "text": "看这张图"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }
    client = QueueClient(ModelResponse(text="看到了", usage=Usage(2, 2)))
    app = _app(tmp_path, client)
    try:
        legacy = app.session.persistable_user_message(payload)
        runtime = app.session._runtime_user_message(payload)
        transformed = app.session._persistable_typed_user_message(runtime)
        assert isinstance(transformed, UserMessage)
        # 序列化等价:字典形态逐键一致。
        from lsm_harness.ops.session_store import _message_to_dict

        assert _message_to_dict(transformed) == _message_to_dict(legacy)

        # 端到端:多模态 prompt 落树的条目带占位符、不含 base64。
        result = app.respond(payload, source="test")
        assert result.status == "completed"
        users = _user_entries(app)
        assert len(users) == 1
        content = users[0]["message"]["content"]
        assert "[image]" in str(content)
        assert "AAAA" not in json.dumps(users[0], ensure_ascii=False)
    finally:
        app.close()
