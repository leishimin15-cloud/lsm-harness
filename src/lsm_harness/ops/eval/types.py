"""Eval data model (moved verbatim from the former ``ops/eval.py``).

These three types are the backward-compatible core of Eval 1.0.  The richer
scenario model (``scenario.py``) builds on top of them for Eval 2.0 without
disturbing the deterministic-case contract that ``test_eval_framework.py``
pins down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsm_harness.ai.types import ModelResponse


@dataclass
class EvalCase:
    """One evaluation test case."""

    name: str
    description: str = ""
    user_message: str = ""
    # Expected tool calls (exact match on tool name)
    expect_tools: list[str] = field(default_factory=list)
    # Patterns that MUST appear in the reply
    expect_in_reply: list[str] = field(default_factory=list)
    # Patterns that must NOT appear
    forbid_in_reply: list[str] = field(default_factory=list)
    # Whether to use real API (default: deterministic with QueueClient)
    use_real_api: bool = False
    # Maximum iterations allowed
    max_iterations: int = 5
    # Scripted model behavior is an INPUT fixture, independent from the
    # assertions above.  Deriving it from expect_* would make the eval
    # self-fulfilling.
    scripted_responses: list[ModelResponse] = field(default_factory=list)


@dataclass
class EvalResult:
    case_name: str
    passed: bool
    duration_ms: float = 0
    reply: str = ""
    tools_called: list[str] = field(default_factory=list)
    iterations: int = 0
    failures: list[str] = field(default_factory=list)
    # For integration tests
    judge_score: float | None = None
    judge_reason: str = ""


@dataclass
class EvalSuite:
    """Collection of eval cases with aggregate stats."""

    name: str
    cases: list[EvalCase] = field(default_factory=list)
