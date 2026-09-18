"""Pi-style Eval 2.0 reporting and durable invocation ledger."""

from __future__ import annotations

import json

from lsm_harness.ops.eval.compare import (
    ComparisonReport,
    RepetitionResult,
    VariantRun,
)
from lsm_harness.ops.eval.reporter import (
    create_eval_artifact_dir,
    render_comparison_report,
    write_comparison_run_files,
    write_single_run_files,
)
from lsm_harness.ops.eval.runner import ScenarioResult
from lsm_harness.ops.eval.variant import EvalVariant


def _report() -> ComparisonReport:
    return ComparisonReport(
        scenario="memory_recall",
        variants=[
            VariantRun(
                name="memory-off",
                repetitions=[
                    RepetitionResult(
                        run_index=0,
                        passed=False,
                        duration_ms=100,
                        total_tokens=1000,
                        cost=0.01,
                    )
                ],
            ),
            VariantRun(
                name="memory-on",
                repetitions=[
                    RepetitionResult(
                        run_index=0,
                        passed=True,
                        duration_ms=120,
                        total_tokens=900,
                        cost=0.009,
                    )
                ],
            ),
        ],
    )


def test_create_eval_artifact_dir_uses_unique_run_directory(tmp_path):
    first = create_eval_artifact_dir(tmp_path)
    second = create_eval_artifact_dir(tmp_path)

    assert first.parent == tmp_path / "evals"
    assert second.parent == tmp_path / "evals"
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_comparison_report_contains_paired_metrics():
    rendered = render_comparison_report(
        [_report()],
        eval_set="memory",
        variant_labels={
            "memory-off": "kimi/k3 (memory off)",
            "memory-on": "kimi/k3 (memory on)",
        },
    )

    assert "Eval Comparisons" in rendered
    assert "Baseline  kimi/k3 (memory off)" in rendered
    assert "Candidate  kimi/k3 (memory on) (1/1 pairs)" in rendered
    assert "+100.0 pp" in rendered
    assert "Tokens" in rendered
    assert "Latency" in rendered
    assert "Est. cost" in rendered


def test_comparison_ledger_writes_one_record_per_observation(tmp_path):
    report = _report()
    variants = [
        EvalVariant(name="memory-off", provider="kimi", model="k3"),
        EvalVariant(name="memory-on", provider="kimi", model="k3"),
    ]

    write_comparison_run_files(
        tmp_path,
        [report],
        eval_set="memory",
        mode="real",
        variants=variants,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert {record["variant"] for record in records} == {"memory-off", "memory-on"}
    assert all(record["provider"] == "kimi" for record in records)
    assert all(record["model"] == "k3" for record in records)
    assert (tmp_path / "summary.json").exists()


def test_ledger_and_report_carry_cache_and_iteration_metrics(tmp_path):
    """成本基线实验需要的字段:cache 命中率、iterations、tool_errors、P95。"""
    report = ComparisonReport(
        scenario="cost_baseline",
        variants=[
            VariantRun(
                name="baseline",
                repetitions=[
                    RepetitionResult(
                        run_index=0, passed=True, duration_ms=100,
                        total_tokens=1000, cost=0.01,
                        input_tokens=800, cache_read_tokens=100,
                        iterations=4, tool_errors=1,
                    ),
                    RepetitionResult(
                        run_index=1, passed=True, duration_ms=300,
                        total_tokens=1200, cost=0.012,
                        input_tokens=900, cache_read_tokens=200,
                        iterations=5, tool_errors=0,
                    ),
                ],
            ),
            VariantRun(
                name="candidate",
                repetitions=[
                    RepetitionResult(
                        run_index=0, passed=True, duration_ms=90,
                        total_tokens=700, cost=0.007,
                        input_tokens=400, cache_read_tokens=250,
                        iterations=3, tool_errors=0,
                    ),
                    RepetitionResult(
                        run_index=1, passed=True, duration_ms=110,
                        total_tokens=800, cost=0.008,
                        input_tokens=500, cache_read_tokens=250,
                        iterations=3, tool_errors=0,
                    ),
                ],
            ),
        ],
    )
    candidate = report.variants[1]
    # 聚合命中率 = Σcache_read / Σ(cache_read+input)
    assert candidate.cache_hit_rate == 500 / 1400
    assert candidate.avg_iterations == 3.0
    assert candidate.avg_tool_errors == 0.0
    # nearest-rank P95:2 个样本时退化为最大值
    assert report.variants[0].p95_duration_ms == 300

    d = report.as_dict()
    assert d["variants"][1]["cache_hit_rate"] == 500 / 1400
    assert d["pair"]["iterations"]["baseline_mean"] == 4.5
    assert d["pair"]["tool_errors"]["delta"] == -0.5
    assert "cache_hit_rate" in d["pair"]

    rendered = render_comparison_report([report], eval_set="cost")
    assert "Cache hit" in rendered
    assert "Iterations" in rendered

    write_comparison_run_files(
        tmp_path, [report], eval_set="cost", mode="real",
        variants=[EvalVariant(name="baseline"), EvalVariant(name="candidate")],
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    baseline_rep0 = next(
        r for r in records
        if r["variant"] == "baseline" and r["repetition"] == 0
    )
    assert baseline_rep0["usage"]["cache_hit_rate"] == 100 / 900
    assert baseline_rep0["usage"]["cache_read_tokens"] == 100
    assert baseline_rep0["iterations"] == 4
    assert baseline_rep0["tool_errors"] == 1


def test_offline_comparison_records_scripted_runtime_and_requested_variant(tmp_path):
    report = _report()
    variants = [
        EvalVariant(name="memory-off", provider="kimi", model="k3"),
        EvalVariant(name="memory-on", provider="deepseek", model="deepseek-chat"),
    ]

    write_comparison_run_files(
        tmp_path,
        [report],
        eval_set="core-suite",
        mode="offline",
        variants=variants,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["execution_mode"] == "offline" for record in records)
    assert all(record["provider"] == "scripted" for record in records)
    assert {record["variant_config"]["provider"] for record in records} == {
        "kimi",
        "deepseek",
    }


def test_offline_ledger_identifies_the_scripted_runtime(tmp_path):
    result = ScenarioResult(name="smoke", passed=True, duration_ms=2)

    write_single_run_files(
        tmp_path,
        [result],
        mode="offline",
        variant=None,
    )

    record = json.loads(
        (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["provider"] == "scripted"
    assert record["model"] == "scripted"
