"""Judges: decide whether a scenario run passed.

Two strategies:
  - :class:`DeterministicJudge` — a run passes iff every assertion passed
    (the default; no model involved).
  - :class:`ModelJudge` — ask a model to score the final answer against the
    scenario's expectations.  For offline/CI use it can wrap a scripted
    client; with a real client it becomes an integration judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from lsm_harness.ai.types import ModelClient

from lsm_harness.ops.eval.assertions import Assertion, AssertionContext, evaluate


@dataclass
class JudgeVerdict:
    passed: bool
    score: float = 0.0
    reason: str = ""


class Judge(Protocol):
    def judge(
        self,
        name: str,
        final_reply: str,
        assertions: list[Assertion],
        ctx: AssertionContext,
    ) -> JudgeVerdict: ...


class DeterministicJudge:
    """Pass iff every assertion passes (score 1.0 / 0.0)."""

    def judge(
        self,
        name: str,
        final_reply: str,
        assertions: list[Assertion],
        ctx: AssertionContext,
    ) -> JudgeVerdict:
        failures: list[str] = []
        for assertion in assertions:
            passed, message = evaluate(assertion, ctx)
            if not passed:
                failures.append(f"{assertion.describe()}: {message}")
        if not failures:
            return JudgeVerdict(passed=True, score=1.0, reason="all assertions passed")
        return JudgeVerdict(passed=False, score=0.0, reason="; ".join(failures))


class ModelJudge:
    """Ask a model to score the final answer (0–1) against the assertions.

    The prompt lists the assertions (the ground-truth expectations) and the
    produced reply, and asks for a JSON verdict ``{"passed": bool, "score":
    float, "reason": str}``.  The client may be a real API client or a
    scripted one for offline runs.
    """

    def __init__(self, client: ModelClient, model_id: str = ""):
        self.client = client
        self.model_id = model_id

    def judge(
        self,
        name: str,
        final_reply: str,
        assertions: list[Assertion],
        ctx: AssertionContext,
    ) -> JudgeVerdict:
        if not assertions:
            return JudgeVerdict(passed=True, score=1.0, reason="no assertions")
        expected = [a.describe() for a in assertions]
        prompt = (
            "You are scoring a coding agent's final answer.\n"
            f"Scenario: {name}\n"
            f"Expectations:\n" + "\n".join(f"- {e}" for e in expected) +
            f"\n\nAgent's final answer:\n{final_reply}\n\n"
            'Reply with JSON only: {"passed": bool, "score": float, "reason": str}.'
        )
        try:
            response = self.client.complete(
                model=self.model_id or _model_id(self.client),
                system="",
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                max_tokens=512,
            )
        except Exception as exc:  # a judge failure must not crash the eval
            return JudgeVerdict(passed=False, score=0.0, reason=f"judge error: {exc}")
        return _parse_verdict(response.text)


def _parse_verdict(text: str) -> JudgeVerdict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return JudgeVerdict(passed=False, score=0.0, reason=f"unparseable judge: {text[:200]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return JudgeVerdict(passed=False, score=0.0, reason=f"unparseable judge: {text[:200]!r}")
    return JudgeVerdict(
        passed=bool(data.get("passed", False)),
        score=float(data.get("score", 0.0)),
        reason=str(data.get("reason", "")),
    )


def _model_id(client: ModelClient) -> str:
    """Resolve a valid model id from a real client (small model first).

    ``get_client()`` attaches ``.model`` / ``.small_model`` (each a ``Model``
    with an ``.id``); scripted clients have neither, so this degrades to ``""``
    and the client ignores the model argument.
    """
    for attr in ("small_model", "model"):
        model = getattr(client, attr, None)
        model_id = getattr(model, "id", None)
        if model_id:
            return model_id
    return ""
