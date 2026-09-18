"""Approval broker 单元测试:决策矩阵、事件发射、模式映射。"""

from __future__ import annotations

import pytest

from lsm_harness.coding_agent.approval import (
    PolicyApprovalBroker,
    PromptApprovalBroker,
    ScriptedApprovalBroker,
    broker_for_mode,
)


def _request(broker, *, effect="external_write", tool="exec", emit=None):
    return broker.request(
        turn_id="t1",
        session_id="s1",
        tool_name=tool,
        effect=effect,
        arguments={"command": "openssl version"},
        emit=emit,
    )


class TestPolicyApprovalBroker:
    def test_defaults_approve_local_deny_external(self):
        broker = PolicyApprovalBroker()
        assert _request(broker, effect="local_write", tool="write_file")[0]
        approved, reason = _request(broker, effect="external_write")
        assert not approved
        assert "requires interactive approval" in reason

    def test_external_write_opt_in(self):
        broker = PolicyApprovalBroker(approve_external_write=True)
        assert _request(broker, effect="external_write")[0]

    def test_local_write_opt_out(self):
        broker = PolicyApprovalBroker(approve_local_write=False)
        assert not _request(broker, effect="local_write")[0]

    def test_emits_required_and_resolved_events(self):
        events: list[tuple[str, dict]] = []
        broker = PolicyApprovalBroker()
        _request(broker, emit=lambda kind, data: events.append((kind, data)))
        assert [kind for kind, _ in events] == [
            "tool.approval.required",
            "tool.approval.resolved",
        ]
        assert events[0][1]["tool"] == "exec"
        assert events[0][1]["effect"] == "external_write"
        assert events[1][1]["approved"] is False
        # 参数摘要进事件(可审计),长参数被截断
        assert "openssl version" in events[0][1]["summary"]

    def test_summary_truncates_long_arguments(self):
        events: list[tuple[str, dict]] = []
        broker = PolicyApprovalBroker()
        broker.request(
            turn_id="t1", session_id="s1", tool_name="exec",
            effect="external_write",
            arguments={"command": "x" * 500},
            emit=lambda kind, data: events.append((kind, data)),
        )
        assert len(events[0][1]["summary"]) <= 120

    def test_reject_all_is_a_no_op_but_exists(self):
        # app.py 的 finally 无条件调用;v1 broker 无挂起状态
        PolicyApprovalBroker().reject_all("trace_finished")


class TestPromptApprovalBroker:
    def test_user_approve_and_deny(self):
        approve = PromptApprovalBroker(input_fn=lambda _p: "y",
                                       output_fn=lambda _s: None)
        assert _request(approve)[0]
        deny = PromptApprovalBroker(input_fn=lambda _p: "n",
                                    output_fn=lambda _s: None)
        assert not _request(deny)[0]

    def test_empty_answer_denies(self):
        broker = PromptApprovalBroker(input_fn=lambda _p: "",
                                      output_fn=lambda _s: None)
        approved, reason = _request(broker)
        assert not approved
        assert reason == "user denied"

    def test_eof_denies_instead_of_crashing(self):
        def _raise(_prompt):
            raise EOFError

        broker = PromptApprovalBroker(input_fn=_raise, output_fn=lambda _s: None)
        approved, reason = _request(broker)
        assert not approved
        assert "interrupted" in reason

    def test_local_write_skips_prompt(self):
        def _boom(_prompt):
            raise AssertionError("local_write 不应触发询问")

        broker = PromptApprovalBroker(input_fn=_boom, output_fn=lambda _s: None)
        assert _request(broker, effect="local_write", tool="write_file")[0]


class TestScriptedApprovalBroker:
    def test_replays_verdicts_then_falls_back_to_default(self):
        broker = ScriptedApprovalBroker(True, (False, "nope"), default=True)
        assert _request(broker)[0]
        approved, reason = _request(broker)
        assert not approved and reason == "nope"
        assert _request(broker) == (True, "scripted default")

    def test_records_requests_before_decision(self):
        # 被拒的请求也必须留在记录里(可观测性)
        broker = ScriptedApprovalBroker(False)
        _request(broker)
        assert len(broker.requests) == 1
        assert broker.requests[0].tool_name == "exec"
        assert broker.requests[0].effect == "external_write"

    def test_positional_false_means_verdict_not_default(self):
        # 场景套件里的 ScriptedApprovalBroker(False) 是“第一个裁决为拒”
        broker = ScriptedApprovalBroker(False)
        assert not _request(broker)[0]


class TestBrokerForMode:
    def test_off_returns_none(self):
        assert broker_for_mode("off", interactive=True) is None
        assert broker_for_mode("", interactive=False) is None
        assert broker_for_mode(None, interactive=True) is None

    def test_prompt_requires_interactive_frontend(self):
        assert isinstance(
            broker_for_mode("prompt", interactive=True), PromptApprovalBroker
        )

    def test_headless_prompt_degrades_to_policy(self):
        # print/RPC 不能交互询问;拿到 prompt 时降级为确定性策略
        assert isinstance(
            broker_for_mode("prompt", interactive=False), PolicyApprovalBroker
        )
        assert isinstance(
            broker_for_mode("policy", interactive=False), PolicyApprovalBroker
        )

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown approval mode"):
            broker_for_mode("yolo", interactive=True)
