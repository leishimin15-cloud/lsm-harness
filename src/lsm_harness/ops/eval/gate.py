"""Policy gate between completed evals and a releasable Harness config."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GatePolicy:
    min_pass_rate: float = 1.0
    max_incomplete_runs: int = 0
    max_unrecovered_errors: int = 0
    max_pass_rate_regression: float = 0.0
    max_token_regression_ratio: float = 0.10
    max_latency_regression_ratio: float = 0.20
    max_cost_regression_ratio: float = 0.10


@dataclass
class GateReport:
    passed: bool
    artifact_dir: str
    checked_runs: int
    pass_rate: float
    incomplete_runs: int
    unrecovered_errors: int
    violations: list[str] = field(default_factory=list)
    policy: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_gate(root: str | Path, policy: GatePolicy | None = None) -> GateReport:
    root = Path(root).expanduser().resolve()
    policy = policy or GatePolicy()
    records = _read_jsonl(root / "runs.jsonl")
    variants = {str(record.get("variant", "")) for record in records}
    target = [record for record in records if record.get("variant") == "candidate"] \
        if "baseline" in variants and "candidate" in variants else records

    incomplete = sum(
        1 for record in target
        if not isinstance(record.get("passed"), bool)
        or not isinstance(record.get("observation"), dict)
        or not isinstance(record.get("diagnosis"), dict)
    )
    complete = [record for record in target if isinstance(record.get("passed"), bool)]
    pass_rate = (
        sum(bool(record["passed"]) for record in complete) / len(complete)
        if complete else 0.0
    )
    unrecovered = sum(
        1 for record in target
        if isinstance(record.get("diagnosis"), dict)
        and record["diagnosis"].get("category") not in {"none", "recovered_tool_error"}
    )
    violations: list[str] = []
    if not records:
        violations.append("runs.jsonl 没有可评测记录")
    if pass_rate < policy.min_pass_rate:
        violations.append(
            f"通过率 {pass_rate:.1%} 低于门槛 {policy.min_pass_rate:.1%}"
        )
    if incomplete > policy.max_incomplete_runs:
        violations.append(
            f"不完整观测 {incomplete} 超过门槛 {policy.max_incomplete_runs}"
        )
    if unrecovered > policy.max_unrecovered_errors:
        violations.append(
            f"未恢复错误 {unrecovered} 超过门槛 {policy.max_unrecovered_errors}"
        )
    _check_comparison_regressions(root, policy, violations)

    report = GateReport(
        passed=not violations,
        artifact_dir=str(root),
        checked_runs=len(target),
        pass_rate=pass_rate,
        incomplete_runs=incomplete,
        unrecovered_errors=unrecovered,
        violations=violations,
        policy=asdict(policy),
    )
    (root / "gate_report.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def latest_eval_dir(home: Path) -> Path | None:
    root = home / "evals"
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _check_comparison_regressions(
    root: Path,
    policy: GatePolicy,
    violations: list[str],
) -> None:
    path = root / "summary.json"
    if not path.exists():
        return
    try:
        reports = json.loads(path.read_text(encoding="utf-8")).get("reports", [])
    except (json.JSONDecodeError, OSError):
        violations.append("summary.json 无法解析")
        return
    for report in reports:
        pair = report.get("pair")
        if not isinstance(pair, dict):
            continue
        scenario = report.get("scenario", "unknown")
        lift = float(pair.get("lift", 0.0) or 0.0)
        if lift < -policy.max_pass_rate_regression:
            violations.append(f"{scenario}: 通过率回归 {lift * 100:.1f} pp")
        for name, limit in (
            ("tokens", policy.max_token_regression_ratio),
            ("duration_ms", policy.max_latency_regression_ratio),
            ("cost", policy.max_cost_regression_ratio),
        ):
            metric = pair.get(name) or {}
            baseline = metric.get("baseline_mean")
            candidate = metric.get("candidate_mean")
            if not isinstance(baseline, (int, float)) or baseline <= 0:
                continue
            if isinstance(candidate, (int, float)) and candidate > baseline * (1 + limit):
                ratio = candidate / baseline - 1
                violations.append(
                    f"{scenario}: {name} 回归 {ratio:.1%} 超过门槛 {limit:.1%}"
                )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out
