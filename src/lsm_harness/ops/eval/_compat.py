"""Eval 1.0 runtime, moved verbatim out of the former ``ops/eval.py``.

This module exists purely for backward compatibility: ``run_case`` /
``run_suite`` / ``summarize`` / ``run_evals`` / the golden helpers and the
scripted-client builders all keep their exact names and signatures.  The
Eval 2.0 scenario runner (``runner.py``) is a superset and does not go
through these paths.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.events import HarnessEvent
from lsm_harness.ai.types import ModelResponse, ToolCall, Usage

from lsm_harness.ops.eval.types import EvalCase, EvalResult, EvalSuite


# ── scripted client builders ─────────────────────────────────────


def _queue_client_for_case(case: EvalCase):
    """Build a QueueClient from an explicit, expectation-independent fixture."""
    if not case.scripted_responses:
        raise ValueError(
            f"deterministic eval case '{case.name}' has no scripted responses"
        )
    return _make_queue_client(case.scripted_responses)


def _make_queue_client(responses):
    """Lazy import to avoid circular deps."""
    from collections import deque
    from copy import deepcopy

    class QC:
        def __init__(self, responses):
            self.responses = deque(responses)
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(deepcopy(kwargs))
            if not self.responses:
                raise AssertionError("scripted client ran out")
            return self.responses.popleft()

    return QC(responses)


def _client_stream_fn(client):
    """Canonical StreamFunction view of a scripted eval client."""
    from lsm_harness.ai.messages import message_to_wire
    from lsm_harness.ai.stream import response_stream_function

    def respond(model, context, options):
        return client.complete(
            model=model.id,
            system=context.system_prompt,
            messages=[message_to_wire(m) for m in context.messages],
            tools=context.tools,
            max_tokens=options.max_tokens,
        )

    return response_stream_function(respond)


# ── runner ───────────────────────────────────────────────────────


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
                name="write_file",
                description="Agent writes a file when asked",
                user_message="在当前目录创建 hello.txt，内容写 hello world",
                expect_tools=["write_file"],
                scripted_responses=[
                    ModelResponse(
                        tool_calls=[ToolCall("write-1", "write_file", {})],
                        stop_reason="tool_calls",
                        usage=Usage(100, 50),
                    ),
                    ModelResponse(text="已处理。", usage=Usage(200, 30)),
                ],
            ),
            EvalCase(
                name="list_dir",
                description="Agent lists the directory when asked",
                user_message="看看当前目录里有什么文件",
                expect_tools=["list_dir"],
                scripted_responses=[
                    ModelResponse(
                        tool_calls=[ToolCall("list-1", "list_dir", {})],
                        stop_reason="tool_calls",
                        usage=Usage(100, 50),
                    ),
                    ModelResponse(text="目录已检查。", usage=Usage(200, 30)),
                ],
            ),
            EvalCase(
                name="no_tool_for_chat",
                description="Agent does NOT call tools for casual chat",
                user_message="你好，今天天气怎么样？",
                expect_tools=[],  # Should just reply, no tools
                scripted_responses=[
                    ModelResponse(text="你好。", usage=Usage(100, 20))
                ],
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
                scripted_responses=[
                    ModelResponse(text="事项列表。", usage=Usage(100, 20))
                ],
            ),
            EvalCase(
                name="handle_empty",
                description="Agent handles empty input gracefully",
                user_message="",
                forbid_in_reply=["Error", "error"],
                scripted_responses=[
                    ModelResponse(text="请输入你的问题。", usage=Usage(100, 20))
                ],
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
            harness = Harness(settings, client=qc, stream_fn=_client_stream_fn(qc))
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

    golden_failures = [] if record else _compare_latest_golden(settings, all_results)
    for failure in golden_failures:
        print(f"  golden mismatch: {failure}")

    if record and all(r.passed for r in all_results):
        _record_golden(settings, all_results)
        print("Golden traces recorded.")

    return 0 if s["failed"] == 0 and not golden_failures else 1


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


def _compare_latest_golden(
    settings: Settings,
    results: list[EvalResult],
) -> list[str]:
    """Compare deterministic outputs with the newest recorded baseline."""
    golden_dir = settings.home / "evals" / "golden"
    paths = sorted(golden_dir.glob("golden-*.json")) if golden_dir.exists() else []
    if not paths:
        return []
    try:
        expected_rows = json.loads(paths[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read {paths[-1].name}: {exc}"]
    expected = {str(row.get("name")): row for row in expected_rows}
    failures: list[str] = []
    for result in results:
        baseline = expected.get(result.case_name)
        if baseline is None:
            failures.append(f"{result.case_name}: missing from baseline")
            continue
        actual = {
            "tools_called": result.tools_called,
            "iterations": result.iterations,
            "reply": result.reply,
        }
        for field_name, value in actual.items():
            if baseline.get(field_name) != value:
                failures.append(
                    f"{result.case_name}.{field_name}: "
                    f"expected {baseline.get(field_name)!r}, got {value!r}"
                )
    return failures
