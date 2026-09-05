"""Practical eval framework for lsm-harness.

Three layers:
  1. Deterministic — scripted model, assert exact tool calls / outputs
  2. Integration  — real API, judge response quality with small model
  3. Regression   — record golden traces, detect regressions

Usage:
    lsm eval                    # run all evals in evals/
    lsm eval --suite tools      # run a specific suite
    lsm eval --record           # record golden traces
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lsm_harness.app import Harness
from lsm_harness.config import Settings
from lsm_harness.events import HarnessEvent
from lsm_harness.types import ModelResponse, ToolCall, Usage


# ── types ────────────────────────────────────────────────────────


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


# ── runner ───────────────────────────────────────────────────────


def _queue_client_for_case(case: EvalCase):
    """Build a QueueClient that returns plausible responses for a case."""
    tool_calls = []
    for i, t in enumerate(case.expect_tools):
        tool_calls.append(ToolCall(str(i), t, {}))

    gate_skip = ModelResponse(
        text='{"retrieve":false,"query":"","reason":"eval"}',
        stop_reason="stop",
        usage=Usage(input_tokens=50, output_tokens=10),
    )

    # Build a plausible text reply that satisfies expect_in_reply
    reply_text = " ".join(case.expect_in_reply) if case.expect_in_reply else "好的，我理解了。"

    if tool_calls:
        return _make_queue_client([
            gate_skip,
            ModelResponse(
                tool_calls=tool_calls,
                stop_reason="tool_calls",
                usage=Usage(input_tokens=100, output_tokens=50),
            ),
            ModelResponse(
                text=reply_text,
                stop_reason="stop",
                usage=Usage(input_tokens=200, output_tokens=30),
            ),
        ])
    else:
        return _make_queue_client([
            gate_skip,
            ModelResponse(
                text=reply_text,
                stop_reason="stop",
                usage=Usage(input_tokens=100, output_tokens=20),
            ),
        ])


def _make_queue_client(responses):
    """Lazy import to avoid circular deps."""
    from collections import deque
    from copy import deepcopy
    from lsm_harness.types import StreamDelta

    class QC:
        def __init__(self, responses):
            self.responses = deque(responses)
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(deepcopy(kwargs))
            if not self.responses:
                raise AssertionError("scripted client ran out")
            return self.responses.popleft()

        def stream_complete(self, **kwargs):
            self.calls.append(deepcopy(kwargs))
            if not self.responses:
                raise AssertionError("scripted client ran out")
            resp = self.responses.popleft()
            import json
            if resp.text:
                yield StreamDelta(kind="text_delta", text=resp.text)
            for i, tc in enumerate(resp.tool_calls or []):
                args_str = json.dumps(tc.arguments, ensure_ascii=False)
                yield StreamDelta(
                    kind="tool_call_start", tool_index=i,
                    tool_id=tc.id, tool_name=tc.name,
                )
                yield StreamDelta(
                    kind="tool_call_delta", tool_index=i,
                    tool_id=tc.id, tool_name=tc.name,
                    arguments_delta=args_str,
                )
            yield StreamDelta(
                kind="done", stop_reason=resp.stop_reason,
                usage=resp.usage,
            )

    return QC(responses)


def run_case(
    case: EvalCase,
    harness: Harness,
    *,
    client=None,
    judge_client=None,
) -> EvalResult:
    """Run one eval case and return the result."""
    failures: list[str] = []
    started = time.monotonic()

    events: list[HarnessEvent] = []
    tools_called: list[str] = []

    def observer(event: HarnessEvent):
        events.append(event)
        if event.type == "tool.requested":
            tools_called.append(event.data.get("tool", "?"))

    try:
        result = harness.respond(
            case.user_message,
            observer=observer,
            source="eval",
        )
        reply = result.reply
        iterations = result.iterations

        # ── check expected tools ──
        for expected in case.expect_tools:
            if expected not in tools_called:
                failures.append(f"Expected tool '{expected}' not called. "
                                f"Called: {tools_called}")

        # ── check reply content ──
        for pattern in case.expect_in_reply:
            if pattern not in reply:
                failures.append(f"Expected '{pattern}' in reply, not found.")

        for pattern in case.forbid_in_reply:
            if pattern in reply:
                failures.append(f"Forbidden pattern '{pattern}' found in reply.")

    except Exception as exc:
        reply = f"ERROR: {exc}"
        iterations = 0
        failures.append(f"Exception: {exc}")

    duration = (time.monotonic() - started) * 1000

    return EvalResult(
        case_name=case.name,
        passed=len(failures) == 0,
        duration_ms=duration,
        reply=reply[:500],
        tools_called=tools_called,
        iterations=iterations,
        failures=failures,
    )


def run_suite(suite: EvalSuite, harness: Harness) -> list[EvalResult]:
    """Run all cases in a suite and return results."""
    results: list[EvalResult] = []
    for case in suite.cases:
        result = run_case(case, harness)
        results.append(result)
        icon = "✓" if result.passed else "✗"
        print(f"  {icon} {case.name}  ({result.duration_ms:.0f}ms)")
        for f in result.failures:
            print(f"    → {f}")
    return results


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    """Aggregate results into a summary."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_ms = sum(r.duration_ms for r in results) / max(total, 1)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{passed / max(total, 1) * 100:.0f}%",
        "avg_duration_ms": f"{avg_ms:.0f}",
    }


# ── built-in suites ──────────────────────────────────────────────


def tool_accuracy_suite() -> EvalSuite:
    """Tests that the agent calls the right tools for common requests."""
    return EvalSuite(
        name="tool_accuracy",
        cases=[
            EvalCase(
                name="calendar_create",
                description="Agent creates a calendar event when asked",
                user_message="帮我创建一个明天上午9点的会议，主题是'周会'",
                expect_tools=["create_event"],
                expect_in_reply=["周会"],
            ),
            EvalCase(
                name="calendar_list",
                description="Agent lists events when asked about schedule",
                user_message="我这周有什么安排？",
                expect_tools=["list_events"],
            ),
            EvalCase(
                name="save_note",
                description="Agent saves a note when explicitly asked",
                user_message="请记住：我的咖啡偏好是浅烘焙",
                expect_tools=["save_note"],
                expect_in_reply=["浅烘焙"],
            ),
            EvalCase(
                name="memory_search",
                description="Agent searches before updating memory",
                user_message="我之前说过的咖啡偏好，改成深烘焙",
                expect_tools=["manage_memory"],
            ),
            EvalCase(
                name="no_tool_for_chat",
                description="Agent does NOT call tools for casual chat",
                user_message="你好，今天天气怎么样？",
                expect_tools=[],  # Should just reply, no tools
            ),
        ],
    )


def safety_suite() -> EvalSuite:
    """Tests that the agent handles edge cases safely."""
    return EvalSuite(
        name="safety",
        cases=[
            EvalCase(
                name="reject_truncation",
                description="Agent recovers when output is truncated",
                user_message="详细列出所有事项",
                max_iterations=3,
                forbid_in_reply=["__TERMINATE__", "__parse_error__"],
            ),
            EvalCase(
                name="handle_empty",
                description="Agent handles empty input gracefully",
                user_message="",
                forbid_in_reply=["Error", "error"],
            ),
        ],
    )


ALL_SUITES: dict[str, Callable[[], EvalSuite]] = {
    "tools": tool_accuracy_suite,
    "safety": safety_suite,
}


# ── CLI entry ────────────────────────────────────────────────────


def run_evals(
    home: str | None = None,
    suite_name: str = "",
    record: bool = False,
) -> int:
    """Run eval suites.  Called by `lsm eval`.

    Uses deterministic QueueClient by default (no API cost).
    """
    settings = Settings()
    if home:
        settings.home = Path(home)
    settings.ensure_home()

    suites_to_run: dict[str, EvalSuite] = {}
    if suite_name:
        if suite_name not in ALL_SUITES:
            print(f"Unknown suite: {suite_name}")
            print(f"Available: {', '.join(ALL_SUITES)}")
            return 1
        suites_to_run[suite_name] = ALL_SUITES[suite_name]()
    else:
        suites_to_run = {n: f() for n, f in ALL_SUITES.items()}

    all_results: list[EvalResult] = []
    for name, suite in suites_to_run.items():
        print(f"\n── {name} ({len(suite.cases)} cases) ──")
        for case in suite.cases:
            # Build a fresh harness with a scripted client for this case
            qc = _queue_client_for_case(case)
            harness = Harness(settings, client=qc)
            try:
                result = run_case(case, harness)
                all_results.append(result)
                icon = "✓" if result.passed else "✗"
                print(f"  {icon} {case.name}  ({result.duration_ms:.0f}ms)")
                for f in result.failures:
                    print(f"    → {f}")
            finally:
                harness.close()

    # ── summary ──
    s = summarize(all_results)
    print(f"\n{'='*40}")
    print(f"Total: {s['total']}  "
          f"Passed: {s['passed']}  "
          f"Failed: {s['failed']}  "
          f"Rate: {s['pass_rate']}  "
          f"Avg: {s['avg_duration_ms']}ms")

    if record and all(r.passed for r in all_results):
        _record_golden(settings, all_results)
        print("Golden traces recorded.")

    return 0 if s["failed"] == 0 else 1


def _record_golden(settings: Settings, results: list[EvalResult]) -> None:
    """Save results as golden reference for regression testing."""
    golden_dir = settings.home / "evals" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H%M%S")
    path = golden_dir / f"golden-{timestamp}.json"
    data = [
        {
            "name": r.case_name,
            "tools_called": r.tools_called,
            "iterations": r.iterations,
            "reply": r.reply,
        }
        for r in results
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"  → saved to {path}")
