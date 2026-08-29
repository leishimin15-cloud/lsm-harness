"""Blocking, thread-safe approval broker used by the local Web gateway."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Callable
from uuid import uuid4


@dataclass
class ApprovalRequest:
    id: str
    turn_id: str
    session_id: str
    tool_name: str
    effect: str
    arguments: dict[str, Any]
    sandboxed: bool
    created_at: str
    expires_at: str
    status: str = "pending"
    reason: str = ""
    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        # dataclasses.asdict deep-copies every field and therefore tries to
        # pickle threading.Event's internal lock.  Build the public payload
        # explicitly so synchronisation primitives can never enter SSE/Trace.
        return {
            "id": self.id,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "effect": self.effect,
            "arguments": self.arguments,
            "sandboxed": self.sandboxed,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status,
            "reason": self.reason,
        }


class ApprovalBroker:
    def __init__(
        self,
        timeout: float = 120.0,
        redactor: Callable[[Any], Any] | None = None,
    ):
        self.timeout = max(1.0, float(timeout))
        self.redactor = redactor or (lambda value: value)
        self._requests: dict[str, ApprovalRequest] = {}
        self._history: list[ApprovalRequest] = []
        self._lock = threading.Lock()

    def request(
        self,
        *,
        turn_id: str,
        session_id: str,
        tool_name: str,
        effect: str,
        arguments: dict[str, Any],
        sandboxed: bool,
        emit: Callable[[str, dict], None] | None,
        interrupt: threading.Event | None = None,
    ) -> tuple[bool, str]:
        now = datetime.now(UTC)
        request = ApprovalRequest(
            id=uuid4().hex,
            turn_id=turn_id,
            session_id=session_id,
            tool_name=tool_name,
            effect=effect,
            arguments=self.redactor(arguments),
            sandboxed=sandboxed,
            created_at=now.isoformat(timespec="seconds"),
            expires_at=(now + timedelta(seconds=self.timeout)).isoformat(timespec="seconds"),
        )
        with self._lock:
            self._requests[request.id] = request
        if emit:
            emit("tool.approval.required", request.public())

        deadline = time.monotonic() + self.timeout
        while not request._event.wait(0.1):
            if interrupt and interrupt.is_set():
                self.resolve(request.id, "reject", reason="turn aborted")
                break
            if time.monotonic() >= deadline:
                self.resolve(request.id, "reject", reason="approval timed out")
                break

        with self._lock:
            approved = request.status == "approved"
            reason = request.reason or ("approved" if approved else "rejected")
            self._requests.pop(request.id, None)
            if request not in self._history:
                self._history.append(request)
                self._history = self._history[-200:]
        if emit:
            emit("tool.approval.resolved", {
                "id": request.id,
                "tool_name": request.tool_name,
                "decision": "approve" if approved else "reject",
                "reason": reason,
            })
        return approved, reason

    def resolve(self, request_id: str, decision: str, *, reason: str = "") -> bool:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.status != "pending":
                return False
            request.status = "approved" if decision == "approve" else "rejected"
            request.reason = reason or request.status
            request._event.set()
            return True

    def reject_all(self, reason: str = "turn ended") -> None:
        with self._lock:
            pending = list(self._requests.values())
        for request in pending:
            self.resolve(request.id, "reject", reason=reason)

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {
                "pending": [r.public() for r in self._requests.values()],
                "history": [r.public() for r in reversed(self._history[-50:])],
            }
