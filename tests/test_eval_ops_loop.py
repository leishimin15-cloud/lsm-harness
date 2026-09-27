"""Trace → Observe → Diagnose → Gate → Release contract."""

from __future__ import annotations

import json

from lsm_harness.events import make_event
from lsm_harness.ops.eval.artifacts import EvalRunArtifacts
from lsm_harness.ops.eval.diagnose import diagnose_artifacts
from lsm_harness.ops.eval.gate import GatePolicy, evaluate_gate
from lsm_harness.ops.eval.observe import observe_artifacts
from lsm_harness.ops.eval.release import write_release_manifest


def _trace(*events) -> str:
    return "".join(json.dumps(event.as_dict()) + "\n" for event in events)


def test_trace_events_have_portable_span_hierarchy():
    root = make_event("trace.started", "run-1", {}, sequence=1)
    llm = make_event("llm.started", "run-1", {"iteration": 2}, sequence=2)
    tool = make_event(
        "tool.started", "run-1",
        {"tool": "exec", "tool_call_id": "call-1"}, sequence=3,
    )

    assert root.span_id == "run-1"
    assert root.parent_span_id == ""
    assert llm.span_id == "run-1:turn:2"
    assert llm.parent_span_id == "run-1"
    assert llm.span_kind == "LLM"
    assert tool.span_id == "run-1:tool:call-1"
    assert tool.span_kind == "TOOL"


def test_observe_and_diagnose_recovered_tool_error():
    artifacts = EvalRunArtifacts(
        name="recovery",
        run_id="run-1",
        status="ok",
        duration_ms=120,
        tool_calls=[{"tool": "exec", "is_error": True, "output": "exit code 2"}],
        trace_jsonl=_trace(
            make_event("trace.started", "run-1", {}, sequence=1, duration_ms=0),
            make_event("llm.started", "run-1", {"iteration": 1}, sequence=2, duration_ms=10),
            make_event("llm.text.start", "run-1", {"iteration": 1}, sequence=3, duration_ms=25),
            make_event("llm.completed", "run-1", {"iteration": 1}, sequence=4, duration_ms=50),
            make_event("tool.started", "run-1", {"tool": "exec", "tool_call_id": "c1"}, sequence=5, duration_ms=60),
            make_event("tool.completed", "run-1", {"tool": "exec", "tool_call_id": "c1", "is_error": True}, sequence=6, duration_ms=80),
            make_event("trace.completed", "run-1", {}, sequence=7, duration_ms=120),
        ),
        usage=[{
            "input_tokens": 100, "output_tokens": 20,
            "cache_read_tokens": 30, "cache_write_tokens": 0,
            "total_tokens": 150, "cost_total": 0.01,
        }],
    )

    observation = observe_artifacts(artifacts)
    diagnosis = diagnose_artifacts(artifacts, observation)

    assert observation.first_token_ms == 15
    assert observation.llm_ms == 40
    assert observation.tool_ms == 20
    assert observation.total_tokens == 150
    assert observation.tool_errors == 1
    assert diagnosis.category == "recovered_tool_error"
    assert diagnosis.self_corrected is True


def test_gate_writes_report_and_release_manifest(tmp_path):
    record = {
        "run_id": "run-1", "variant": "default", "passed": True,
        "git_revision": "abc", "config": {"model": "scripted"},
        "observation": {"total_tokens": 100, "total_ms": 20},
        "diagnosis": {"category": "none", "result": "passed"},
    }
    (tmp_path / "runs.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "summary.json").write_text(json.dumps({"reports": []}))

    report = evaluate_gate(tmp_path)
    manifest = write_release_manifest(tmp_path, report)

    assert report.passed
    assert (tmp_path / "gate_report.json").exists()
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["git_revisions"] == ["abc"]
    assert payload["gate"]["passed"] is True


def test_gate_blocks_candidate_regression(tmp_path):
    rows = [
        {"variant": "baseline", "passed": True, "observation": {}, "diagnosis": {"category": "none"}},
        {"variant": "candidate", "passed": True, "observation": {}, "diagnosis": {"category": "none"}},
    ]
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {
        "reports": [{
            "scenario": "task",
            "pair": {
                "lift": 0.0,
                "tokens": {"baseline_mean": 100, "candidate_mean": 130},
                "duration_ms": {"baseline_mean": 100, "candidate_mean": 100},
                "cost": {"baseline_mean": 1, "candidate_mean": 1},
            },
        }]
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    report = evaluate_gate(tmp_path, GatePolicy(max_token_regression_ratio=0.10))

    assert not report.passed
    assert any("tokens 回归" in violation for violation in report.violations)
