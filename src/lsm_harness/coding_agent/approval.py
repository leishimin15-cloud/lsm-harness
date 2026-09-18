"""Approval brokers: the policy/human gate for side-effecting tools.

When an ``approval_broker`` is present on the run config, the tool
registry intercepts every non-read tool call *before* execution and
asks the broker (``agent/tools.py``). The safety boundary therefore
lives in the execution layer — a prompt can talk the model into
anything, but the tool does not run without an approval.

Three implementations:

- ``PolicyApprovalBroker`` — headless deterministic policy (eval, -p,
  RPC): local workspace writes auto-approve (the workspace boundary is
  already enforced by ``LocalFileOperations``); external effects
  (shell) default-deny unless configured otherwise.
- ``PromptApprovalBroker`` — interactive stdin y/n for the plain CLI.
- ``ScriptedApprovalBroker`` — eval/tests: replay queued verdicts and
  record every request for assertions.

All brokers emit ``tool.approval.required`` / ``tool.approval.resolved``
through the run's emit channel so approvals land in traces and the TUI
flow projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ApprovalRequest:
    turn_id: str
    session_id: str
    tool_name: str
    effect: str
    arguments: dict[str, Any]


def _summarize_arguments(arguments: dict[str, Any], limit: int = 120) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class _BaseBroker:
    """Shared emit plumbing; subclasses implement ``_decide``."""

    def _decide(self, request: ApprovalRequest) -> tuple[bool, str]:
        raise NotImplementedError

    def request(
        self,
        *,
        turn_id: str,
        session_id: str,
        tool_name: str,
        effect: str,
        arguments: dict[str, Any],
        emit: Callable[[str, dict[str, Any]], None] | None = None,
        interrupt: Any = None,
    ) -> tuple[bool, str]:
        del interrupt  # v1 brokers don't support mid-prompt abort
        request = ApprovalRequest(
            turn_id=turn_id,
            session_id=session_id,
            tool_name=tool_name,
            effect=effect,
            arguments=arguments,
        )
        if emit is not None:
            emit(
                "tool.approval.required",
                {
                    "turn_id": turn_id,
                    "tool": tool_name,
                    "effect": effect,
                    "summary": _summarize_arguments(arguments),
                },
            )
        approved, reason = self._decide(request)
        if emit is not None:
            emit(
                "tool.approval.resolved",
                {
                    "turn_id": turn_id,
                    "tool": tool_name,
                    "effect": effect,
                    "approved": approved,
                    "reason": reason,
                },
            )
        return approved, reason

    def reject_all(self, reason: str) -> None:
        """Run 收尾钩子(app.py finally 调用):拒绝所有仍挂起的请求。

        v1 的 broker 都不持有挂起状态(每次 request 同步决策),
        所以默认是 no-op;TUI 审批 UI 之类的异步 broker 应覆盖它,
        在 run 结束/中断时拒掉未裁决的队列项。
        """
        del reason


class PolicyApprovalBroker(_BaseBroker):
    """Deterministic headless policy.

    ``approve_local_write``: workspace-bound writes (write_file, edit)
    — safe to auto-approve because the writable boundary is enforced by
    ``LocalFileOperations`` regardless of what the model asks for.
    ``approve_external_write``: shell commands and other host effects.
    Default False: with no human available, external effects are denied.
    """

    def __init__(
        self,
        *,
        approve_local_write: bool = True,
        approve_external_write: bool = False,
    ) -> None:
        self.approve_local_write = approve_local_write
        self.approve_external_write = approve_external_write

    def _decide(self, request: ApprovalRequest) -> tuple[bool, str]:
        if request.effect == "local_write" and self.approve_local_write:
            return True, "policy: workspace-local write auto-approved"
        if request.effect == "external_write" and self.approve_external_write:
            return True, "policy: external effect auto-approved"
        return (
            False,
            f"policy: {request.effect} requires interactive approval "
            "(headless mode denies by default)",
        )


class PromptApprovalBroker(_BaseBroker):
    """Interactive stdin approval for the plain CLI chat frontend."""

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        approve_local_write: bool = True,
    ) -> None:
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.approve_local_write = approve_local_write

    def _decide(self, request: ApprovalRequest) -> tuple[bool, str]:
        if request.effect == "local_write" and self.approve_local_write:
            return True, "policy: workspace-local write auto-approved"
        self.output_fn(
            f"\n[approval] {request.tool_name} ({request.effect}): "
            f"{_summarize_arguments(request.arguments)}"
        )
        try:
            answer = self.input_fn("允许执行?[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False, "user interrupted the approval prompt"
        if answer in {"y", "yes"}:
            return True, "user approved"
        return False, "user denied"


class ScriptedApprovalBroker(_BaseBroker):
    """Eval/tests: replay queued verdicts, record every request.

    Verdicts are ``(approved, reason)`` pairs or bare bools. Once the
    queue is empty, ``default`` decides. Recording happens BEFORE the
    decision, so a denied request is still observable.

    注意:eval 的场景对象在 compare 两腿和 repetitions 之间复用,
    排队裁决只够第一条腿用——场景套件里请用 ``default=`` 表达
    可重复的裁决;排队模式只用于单次运行的单元测试。
    """

    def __init__(self, *verdicts: bool | tuple[bool, str], default: bool = True) -> None:
        self._verdicts: list[tuple[bool, str]] = [
            (v, "scripted") if isinstance(v, bool) else v for v in verdicts
        ]
        self.default = default
        self.requests: list[ApprovalRequest] = []

    def request(self, **kwargs: Any) -> tuple[bool, str]:
        # 记录先于决策:被拒的请求也要可观测。
        self.requests.append(
            ApprovalRequest(
                turn_id=kwargs.get("turn_id", ""),
                session_id=kwargs.get("session_id", ""),
                tool_name=kwargs.get("tool_name", ""),
                effect=kwargs.get("effect", ""),
                arguments=kwargs.get("arguments", {}),
            )
        )
        return super().request(**kwargs)

    def _decide(self, request: ApprovalRequest) -> tuple[bool, str]:
        del request
        if self._verdicts:
            return self._verdicts.pop(0)
        return self.default, "scripted default"


def broker_for_mode(mode: str, *, interactive: bool):
    """Build the broker a frontend should pass to ``respond``.

    ``off``/空 → None(无审批门,既有行为)。``prompt`` 只在交互前端
    可用;print/RPC 等 headless 前端拿到 ``prompt`` 时降级为
    ``policy``(headless 无法询问,外部副作用默认拒)。
    """
    mode = (mode or "off").strip().lower()
    if mode in ("", "off"):
        return None
    if mode == "prompt" and interactive:
        return PromptApprovalBroker()
    if mode in ("policy", "prompt"):
        return PolicyApprovalBroker()
    raise ValueError(
        f"unknown approval mode {mode!r} (expected off|policy|prompt)"
    )


__all__ = [
    "ApprovalRequest",
    "PolicyApprovalBroker",
    "PromptApprovalBroker",
    "ScriptedApprovalBroker",
    "broker_for_mode",
]
