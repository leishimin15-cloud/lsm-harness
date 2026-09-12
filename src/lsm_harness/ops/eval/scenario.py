"""Multi-step scenario model for Eval 2.0.

A scenario is an ordered list of steps executed against one harness (with an
optional ``reload`` step replacing it in-place to simulate a restart).  Steps
cover the full coding-agent lifecycle: prompt → tool call → abort → reload →
continue → compact → shell command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from lsm_harness.ai.types import ModelResponse

from lsm_harness.ops.eval.fixture import EvalFixture
from lsm_harness.ops.eval.assertions import Assertion

StepKind = Literal["prompt", "abort", "reload", "continue", "compact", "command"]


@dataclass
class EvalStep:
    """One action in a scenario."""

    kind: StepKind
    text: str = ""          # prompt text, or the shell command for "command"
    park: bool = False      # prompt only: park the model stream (await abort)
    cwd: str = ""           # command only
    timeout: float = 0.0    # seconds; 0 = no timeout (prompt & command)


@dataclass
class EvalScenario:
    """A self-contained, deterministic coding-agent scenario."""

    name: str
    description: str = ""
    fixture: EvalFixture | None = None
    steps: list[EvalStep] = field(default_factory=list)
    # Scripted model responses (in order).  A summary response for compaction
    # is supplied via ``summary_text`` and detected by the client, not queued
    # here — compaction fires an unbounded number of summarizer calls.
    scripted_responses: list[ModelResponse] = field(default_factory=list)
    # Scripted compaction summary text (empty → compaction summarizer returns
    # a generic summary).  Used only when a ``compact`` step is present.
    summary_text: str = ""
    use_real_api: bool = False
    max_iterations: int = 8
    settings_overrides: dict[str, Any] = field(default_factory=dict)
    assertions: list[Assertion] = field(default_factory=list)


# ── step constructors ─────────────────────────────────────────────


def prompt(text: str, *, park: bool = False, timeout: float = 0.0) -> EvalStep:
    """A normal (or parked) user prompt; ``timeout`` bounds the whole step."""
    return EvalStep(kind="prompt", text=text, park=park, timeout=timeout)


def abort() -> EvalStep:
    """Abort the currently running (parked) run."""
    return EvalStep(kind="abort")


def reload() -> EvalStep:
    """Close the harness and rebuild one on the same home/session."""
    return EvalStep(kind="reload")


def continue_() -> EvalStep:
    """``respond_continue`` — re-run the loop without a fresh user message."""
    return EvalStep(kind="continue")


def compact() -> EvalStep:
    """Manual compaction of the current session path."""
    return EvalStep(kind="compact")


def command(args: str, *, cwd: str = "", timeout: float = 60.0) -> EvalStep:
    """Run a shell command inside the workspace."""
    return EvalStep(kind="command", text=args, cwd=cwd, timeout=timeout)
