"""Eval 2.0 end-to-end: scenarios, artifacts, comparison.

Covers the five core scenarios over isolated temporary workspaces, artifact
writing, and the baseline-vs-candidate comparison runner — all fully
deterministic (scripted model, no API key).
"""

from __future__ import annotations

import json

from lsm_harness.ops.eval import (
    EvalFixture,
    EvalRunArtifacts,
    EvalScenario,
    command,
    run_comparison,
    run_scenario,
)
from lsm_harness.ops.eval.suites import core_scenarios


def test_all_core_scenarios_pass():
    scenarios = core_scenarios()
    assert len(scenarios) == 23
    for scenario in scenarios:
        result = run_scenario(scenario)
        assert result.passed, (
            f"{scenario.name} failed: {result.failures}"
        )


def test_read_and_answer_scenario_artifacts(tmp_path):
    scenario = next(
        s for s in core_scenarios() if s.name == "read_and_answer"
    )
    bundle = tmp_path / "read_and_answer"
    result = run_scenario(scenario, artifacts_dir=bundle)
    assert result.passed
    assert "42" in result.final_reply
    assert any(
        call["tool"] == "read_file" for call in result.tool_calls
    )

    assert bundle.is_dir()
    assert (bundle / "result.json").exists()
    assert (bundle / "reply.txt").exists()
    assert (bundle / "tool_calls.json").exists()
    assert (bundle / "session.jsonl").exists()

    data = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    assert data["name"] == "read_and_answer"
    assert data["status"] == "ok"


def test_calc_bug_scenario_changes_workspace(tmp_path):
    scenario = next(s for s in core_scenarios() if s.name == "calc_bug")
    bundle = tmp_path / "calc_bug"
    result = run_scenario(scenario, artifacts_dir=bundle)
    assert result.passed
    changed = {path for _, path in result.artifacts.changed_files}
    assert "calc.py" in changed

    patch = (bundle / "diff.patch").read_text(encoding="utf-8")
    assert "return a - b" in patch
    assert "return a + b" in patch


def test_comparison_aggregates_pass_rate():
    scenarios = [s for s in core_scenarios() if s.name == "read_and_answer"]
    reports = run_comparison(scenarios, repetitions=2)
    assert len(reports) == 1
    report = reports[0]
    assert report.scenario == "read_and_answer"
    assert len(report.variants) == 1  # default single "candidate" variant
    variant = report.variants[0]
    assert variant.total == 2
    assert variant.passed == 2
    assert variant.pass_rate == 1.0
    assert variant.avg_duration_ms >= 0.0
    assert "candidate" in report.render()


def test_artifacts_round_trip_dict():
    artifacts = EvalRunArtifacts(
        name="x",
        status="ok",
        duration_ms=12.5,
        final_answer="hello",
        tool_calls=[{"tool": "read_file", "args": {"path": "a.txt"}}],
        changed_files=[("modified", "a.txt")],
    )
    d = artifacts.as_dict()
    assert d["name"] == "x"
    assert d["changed_files"] == [{"status": "modified", "path": "a.txt"}]


def _failing_scenario() -> EvalScenario:
    return EvalScenario(
        name="failing_command",
        description="命令失败应导致 run 失败",
        fixture=EvalFixture.inline({"x.txt": "hi\n"}),
        steps=[command("exit 1")],
        scripted_responses=[],
        assertions=[],
    )


def test_failed_step_fails_the_run():
    result = run_scenario(_failing_scenario())
    assert result.passed is False
    assert any("step command" in f for f in result.failures)


def test_failed_run_artifacts_are_accurate(tmp_path):
    bundle = tmp_path / "failing_command"
    result = run_scenario(_failing_scenario(), artifacts_dir=bundle)
    assert result.passed is False
    assert result.artifacts is not None
    assert result.artifacts.status == "failed"
    assert result.artifacts.duration_ms > 0.0
    assert result.artifacts.failures

    data = json.loads(
        (bundle / "result.json").read_text(encoding="utf-8")
    )
    assert data["status"] == "failed"
    assert data["duration_ms"] > 0.0
    assert data["failures"]


def test_variants_do_not_overwrite_artifacts(tmp_path):
    scenarios = [s for s in core_scenarios() if s.name == "read_and_answer"]
    run_comparison(
        scenarios,
        variants={"baseline": {}, "candidate": {}},
        repetitions=2,
        artifacts_dir=tmp_path,
    )
    root = tmp_path / "read_and_answer"
    for variant in ("baseline", "candidate"):
        for rep in ("rep00", "rep01"):
            bundle = root / variant / rep
            assert bundle.is_dir(), f"missing artifact dir {bundle}"
            assert (bundle / "result.json").exists()
            assert (bundle / "session.jsonl").exists()
