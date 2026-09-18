"""Product-level reporting and run ledger for Eval 2.0.

The runner owns execution.  This module owns the durable experiment view:
one directory per invocation, one JSONL observation per run, and a compact
baseline/candidate summary inspired by Pi's eval reporter.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from lsm_harness.ops.eval.compare import ComparisonReport, RepetitionResult
from lsm_harness.ops.eval.runner import ScenarioResult
from lsm_harness.ops.eval.variant import EvalVariant


def create_eval_artifact_dir(
    home: Path,
    explicit: str | Path | None = None,
) -> Path:
    """Create the artifact directory for one complete eval invocation."""
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        root = (home / "evals" / f"{stamp}_{uuid4()}").resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root


def render_results(results: list[ScenarioResult], *, mode: str) -> str:
    """Render one-pass results without pretending they are an A/B report."""
    lines = ["Eval Results", f"  mode: {mode}"]
    for result in results:
        marker = "✓" if result.passed else "✗"
        lines.append(f"  {marker} {result.name}  {result.duration_ms:.0f}ms")
        for failure in result.failures:
            lines.append(f"      {failure}")
    passed = sum(result.passed for result in results)
    lines.append(f"  Summary: {passed}/{len(results)} passed")
    return "\n".join(lines)


def render_comparison_report(
    reports: list[ComparisonReport],
    *,
    eval_set: str,
    variant_labels: Mapping[str, str] | None = None,
) -> str:
    """Aggregate all paired scenarios into one Pi-style comparison report."""
    pairs: list[tuple[RepetitionResult, RepetitionResult]] = []
    baseline_name = "baseline"
    candidate_name = "candidate"
    for report in reports:
        if len(report.variants) != 2:
            continue
        baseline, candidate = report.variants
        baseline_name = baseline.name
        candidate_name = candidate.name
        pairs.extend(zip(baseline.repetitions, candidate.repetitions))

    lines = ["Eval Comparisons", f"  {eval_set}"]
    if not pairs:
        lines.append("    comparison unavailable: no paired observations")
        return "\n".join(lines)

    baseline_passes = sum(item.passed for item, _ in pairs)
    candidate_passes = sum(item.passed for _, item in pairs)
    total = len(pairs)
    baseline_rate = baseline_passes / total
    candidate_rate = candidate_passes / total
    lift = candidate_rate - baseline_rate
    wins = sum(candidate.passed and not baseline.passed for baseline, candidate in pairs)
    losses = sum(baseline.passed and not candidate.passed for baseline, candidate in pairs)
    ties = total - wins - losses

    labels = variant_labels or {}
    baseline_label = labels.get(baseline_name, baseline_name)
    candidate_label = labels.get(candidate_name, candidate_name)
    lines.extend([
        f"     Baseline  {baseline_label}",
        f"    Candidate  {candidate_label} ({total}/{total} pairs)",
        "    Pass rate  "
        f"{lift * 100:+.1f} pp "
        f"(candidate {candidate_rate * 100:.1f}%, "
        f"baseline {baseline_rate * 100:.1f}%)",
        _metric_line(
            "Tokens", pairs, lambda item: float(item.total_tokens),
            "{:.1f}", "{:+.1f}",
        ),
        _metric_line(
            "Cache hit", pairs, lambda item: item.cache_hit_rate,
            "{:.1%}", "{:+.1%}",
        ),
        _metric_line(
            "Iterations", pairs, lambda item: float(item.iterations),
            "{:.1f}", "{:+.1f}",
        ),
        _metric_line(
            "Latency", pairs, lambda item: item.duration_ms,
            "{:.1f}ms", "{:+.1f}ms",
        ),
        _metric_line(
            "Est. cost", pairs, lambda item: item.cost,
            "${:.4f}", "${:+.4f}",
        ),
        f"    Outcomes: candidate wins {wins}, ties {ties}, losses {losses}",
    ])
    return "\n".join(lines)


def write_single_run_files(
    root: Path,
    results: list[ScenarioResult],
    *,
    mode: str,
    variant: EvalVariant | None,
) -> None:
    """Write one observation per single-run scenario plus summary.json."""
    records = []
    for result in results:
        artifacts = result.artifacts
        if mode == "offline":
            provider = "scripted"
            model = "scripted"
        else:
            provider = artifacts.provider if artifacts is not None else ""
            model = artifacts.model if artifacts is not None else ""
        records.append({
            "schema_version": 1,
            "run_id": str(uuid4()),
            "eval_set": mode,
            "execution_mode": mode,
            "scenario": result.name,
            "variant": variant.name if variant is not None else "default",
            "repetition": 0,
            "provider": provider,
            "model": model,
            "passed": result.passed,
            "score": _score(result),
            "usage": result.usage,
            "iterations": result.iterations,
            "tool_errors": result.tool_error_count,
            "timings": {"total_ms": result.duration_ms},
            "errors": list(result.failures),
            "artifacts": {"directory": result.name},
        })
    _write_jsonl(root / "runs.jsonl", records)
    _write_json(root / "summary.json", {
        "schema_version": 1,
        "mode": mode,
        "passed": sum(result.passed for result in results),
        "total": len(results),
        "scenarios": [result.name for result in results],
    })


def write_comparison_run_files(
    root: Path,
    reports: list[ComparisonReport],
    *,
    eval_set: str,
    mode: str,
    variants: list[EvalVariant],
) -> None:
    """Write paired observations and the machine-readable comparison summary."""
    specs = {variant.name: variant for variant in variants}
    records: list[dict[str, Any]] = []
    for report in reports:
        for variant_run in report.variants:
            spec = specs.get(variant_run.name, EvalVariant(name=variant_run.name))
            for repetition in variant_run.repetitions:
                provider = "scripted" if mode == "offline" else spec.provider
                model = "scripted" if mode == "offline" else spec.model
                records.append({
                    "schema_version": 1,
                    "run_id": str(uuid4()),
                    "eval_set": eval_set,
                    "execution_mode": mode,
                    "scenario": report.scenario,
                    "variant": variant_run.name,
                    "repetition": repetition.run_index,
                    "provider": provider,
                    "model": model,
                    "variant_config": {
                        "provider": spec.provider,
                        "model": spec.model,
                    },
                    "passed": repetition.passed,
                    "score": (
                        repetition.judge_score
                        if repetition.judge_score is not None
                        else float(repetition.passed)
                    ),
                    "usage": {
                        "total_tokens": repetition.total_tokens,
                        "input_tokens": repetition.input_tokens,
                        "cache_read_tokens": repetition.cache_read_tokens,
                        "cache_hit_rate": repetition.cache_hit_rate,
                        "estimated_cost_usd": repetition.cost,
                    },
                    "iterations": repetition.iterations,
                    "tool_errors": repetition.tool_errors,
                    "timings": {"total_ms": repetition.duration_ms},
                    "errors": list(repetition.failures),
                    "artifacts": {
                        "directory": (
                            f"{report.scenario}/{variant_run.name}/"
                            f"rep{repetition.run_index:02d}"
                        )
                    },
                })
    _write_jsonl(root / "runs.jsonl", records)
    _write_json(root / "summary.json", {
        "schema_version": 1,
        "eval_set": eval_set,
        "execution_mode": mode,
        "reports": [report.as_dict() for report in reports],
    })


def _score(result: ScenarioResult) -> float:
    verdict = result.judge_verdict
    return verdict.score if verdict is not None else float(result.passed)


def _metric_line(
    label: str,
    pairs: list[tuple[RepetitionResult, RepetitionResult]],
    select,
    value_template: str,
    delta_template: str,
) -> str:
    baseline = sum(select(left) for left, _ in pairs) / len(pairs)
    candidate = sum(select(right) for _, right in pairs) / len(pairs)
    delta = candidate - baseline
    return (
        f"    {label:>9}  {delta_template.format(delta)} "
        f"(candidate {value_template.format(candidate)}, "
        f"baseline {value_template.format(baseline)})"
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "create_eval_artifact_dir",
    "render_comparison_report",
    "render_results",
    "write_comparison_run_files",
    "write_single_run_files",
]
