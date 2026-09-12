"""Coding-agent evaluation harness (Eval 2.0).

Layers:
  1. Deterministic  — scripted model, assert exact tool calls / outputs.
  2. Scenario       — multi-step scenarios (prompt → tool → file change →
                      reload → continue/compact → verify final result) over an
                      isolated temporary workspace.
  3. Artifacts      — final answer, tool-call records, workspace diff, session
                      JSONL, trace, usage (token/cache/cost) and duration.
  4. Comparison     — baseline vs candidate, repetitions, deterministic or
                      model-scored judge.

Backward-compatible with Eval 1.0: ``EvalCase`` / ``EvalResult`` /
``run_evals`` / the golden helpers keep their names and signatures.
"""

from __future__ import annotations

# ── Eval 1.0 backward-compatible surface ─────────────────────────
from lsm_harness.ops.eval.types import EvalCase, EvalResult, EvalSuite
from lsm_harness.ops.eval._compat import (
    ALL_SUITES,
    _client_stream_fn,
    _compare_latest_golden,
    _make_queue_client,
    _queue_client_for_case,
    _record_golden,
    run_case,
    run_evals,
    run_suite,
    safety_suite,
    summarize,
    tool_accuracy_suite,
)

# ── Eval 2.0 surface ─────────────────────────────────────────────
from lsm_harness.ops.eval.fixture import (
    EvalFixture,
    FixtureRegistry,
    load_dir_fixtures,
    materialize,
)
from lsm_harness.ops.eval.assertions import (
    Assertion,
    AssertionContext,
    evaluate,
    command_exit,
    context_user_once,
    expect_tool,
    file_contains,
    file_equals,
    file_not_contains,
    reply_contains,
    reply_not_contains,
    session_tree_has,
    summary_contains,
    tool_args,
    tool_output_contains,
)
from lsm_harness.ops.eval.scenario import (
    EvalScenario,
    EvalStep,
    abort,
    compact,
    command,
    continue_,
    prompt,
    reload,
)
from lsm_harness.ops.eval.judge import (
    DeterministicJudge,
    JudgeVerdict,
    ModelJudge,
)
from lsm_harness.ops.eval.artifacts import (
    EvalRunArtifacts,
    diff_snapshots,
    snapshot_workspace,
)
from lsm_harness.ops.eval.runner import (
    ScenarioResult,
    StepResult,
    run_scenario,
)
from lsm_harness.ops.eval.suites import core_scenarios, core_tasks
from lsm_harness.ops.eval.task import (
    EvalTask,
    run_task,
    run_task_comparison,
    task_to_scenario,
)
from lsm_harness.ops.eval.compare import (
    ComparisonReport,
    RepetitionResult,
    VariantRun,
    run_comparison,
)
from lsm_harness.ops.eval.variant import (
    EvalToolPolicy,
    EvalVariant,
    apply_tool_policy,
    apply_variant_settings,
    build_variant_session,
)

__all__ = [
    # 1.0
    "EvalCase",
    "EvalResult",
    "EvalSuite",
    "ALL_SUITES",
    "run_case",
    "run_suite",
    "summarize",
    "run_evals",
    "tool_accuracy_suite",
    "safety_suite",
    "_client_stream_fn",
    "_make_queue_client",
    "_queue_client_for_case",
    "_record_golden",
    "_compare_latest_golden",
    # 2.0 — fixture
    "EvalFixture",
    "FixtureRegistry",
    "load_dir_fixtures",
    "materialize",
    # 2.0 — assertions
    "Assertion",
    "AssertionContext",
    "evaluate",
    "expect_tool",
    "tool_args",
    "tool_output_contains",
    "reply_contains",
    "reply_not_contains",
    "file_contains",
    "file_not_contains",
    "file_equals",
    "command_exit",
    "session_tree_has",
    "summary_contains",
    "context_user_once",
    # 2.0 — scenario
    "EvalScenario",
    "EvalStep",
    "prompt",
    "abort",
    "reload",
    "continue_",
    "compact",
    "command",
    # 2.0 — judge
    "JudgeVerdict",
    "DeterministicJudge",
    "ModelJudge",
    # 2.0 — artifacts
    "EvalRunArtifacts",
    "snapshot_workspace",
    "diff_snapshots",
    # 2.0 — runner
    "StepResult",
    "ScenarioResult",
    "run_scenario",
    # 2.0 — suites
    "core_scenarios",
    "core_tasks",
    # 2.0 — task
    "EvalTask",
    "task_to_scenario",
    "run_task",
    "run_task_comparison",
    # 2.0 — comparison
    "RepetitionResult",
    "VariantRun",
    "ComparisonReport",
    "run_comparison",
    # 2.0 — variant
    "EvalVariant",
    "EvalToolPolicy",
    "apply_variant_settings",
    "apply_tool_policy",
    "build_variant_session",
]
