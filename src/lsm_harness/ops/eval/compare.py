"""Pi-style comparison: baseline vs candidate, repetitions, judge.

``run_comparison`` runs each scenario once per repetition per variant and
aggregates pass rate + duration, so a change can be measured for both
correctness and flakiness.  A "variant" is a named bundle of extra
``run_scenario`` kwargs — the default is a single ``candidate`` variant; pass
``variants={"baseline": {...}, "candidate": {...}}`` for an A/B run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lsm_harness.ops.eval.runner import ScenarioResult, run_scenario
from lsm_harness.ops.eval.scenario import EvalScenario
from lsm_harness.ops.eval.variant import EvalVariant


@dataclass
class RepetitionResult:
    run_index: int
    passed: bool
    duration_ms: float
    failures: list[str] = field(default_factory=list)
    judge_score: float | None = None
    judge_reason: str = ""
    total_tokens: int = 0
    cost: float = 0.0


@dataclass
class PairedMetric:
    """baseline vs candidate 的单指标成对统计(Pi PairedMetricSummary)。"""

    baseline_mean: float | None = None
    candidate_mean: float | None = None
    delta: float | None = None


@dataclass
class PairSummary:
    """两个 variant 的成对比较(Pi CorrectnessLiftSummary + 三项 delta)。"""

    baseline: str
    candidate: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    lift: float
    wins: int
    ties: int
    losses: int
    tokens: PairedMetric
    duration_ms: PairedMetric
    cost: PairedMetric


def _paired_metric(baseline_values: list[float], candidate_values: list[float]) -> PairedMetric:
    if not baseline_values or not candidate_values:
        return PairedMetric()
    b = sum(baseline_values) / len(baseline_values)
    c = sum(candidate_values) / len(candidate_values)
    return PairedMetric(baseline_mean=b, candidate_mean=c, delta=c - b)


@dataclass
class VariantRun:
    name: str
    repetitions: list[RepetitionResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.repetitions if r.passed)

    @property
    def total(self) -> int:
        return len(self.repetitions)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_duration_ms(self) -> float:
        if not self.repetitions:
            return 0.0
        return sum(r.duration_ms for r in self.repetitions) / len(self.repetitions)

    @property
    def avg_judge_score(self) -> float | None:
        """Mean judge score over scored repetitions (None when unscored)."""
        scores = [
            r.judge_score for r in self.repetitions
            if r.judge_score is not None
        ]
        if not scores:
            return None
        return sum(scores) / len(scores)

    @property
    def avg_tokens(self) -> float:
        if not self.repetitions:
            return 0.0
        return sum(r.total_tokens for r in self.repetitions) / len(self.repetitions)

    @property
    def avg_cost(self) -> float:
        if not self.repetitions:
            return 0.0
        return sum(r.cost for r in self.repetitions) / len(self.repetitions)


@dataclass
class ComparisonReport:
    scenario: str
    variants: list[VariantRun] = field(default_factory=list)

    def pair_summary(self) -> PairSummary | None:
        """当恰有两个 variant 时给出成对比较(首个为 baseline)。"""
        if len(self.variants) != 2:
            return None
        baseline, candidate = self.variants
        pairs = min(baseline.total, candidate.total)
        wins = ties = losses = 0
        for index in range(pairs):
            b = baseline.repetitions[index].passed
            c = candidate.repetitions[index].passed
            if c and not b:
                wins += 1
            elif b and not c:
                losses += 1
            else:
                ties += 1
        return PairSummary(
            baseline=baseline.name,
            candidate=candidate.name,
            baseline_pass_rate=baseline.pass_rate,
            candidate_pass_rate=candidate.pass_rate,
            lift=candidate.pass_rate - baseline.pass_rate,
            wins=wins,
            ties=ties,
            losses=losses,
            tokens=_paired_metric(
                [r.total_tokens for r in baseline.repetitions],
                [r.total_tokens for r in candidate.repetitions],
            ),
            duration_ms=_paired_metric(
                [r.duration_ms for r in baseline.repetitions],
                [r.duration_ms for r in candidate.repetitions],
            ),
            cost=_paired_metric(
                [r.cost for r in baseline.repetitions],
                [r.cost for r in candidate.repetitions],
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scenario": self.scenario,
            "variants": [
                {
                    "name": v.name,
                    "passed": v.passed,
                    "total": v.total,
                    "pass_rate": v.pass_rate,
                    "avg_duration_ms": v.avg_duration_ms,
                    "avg_judge_score": v.avg_judge_score,
                    "avg_tokens": v.avg_tokens,
                    "avg_cost": v.avg_cost,
                }
                for v in self.variants
            ],
        }
        pair = self.pair_summary()
        if pair is not None:
            out["pair"] = {
                "baseline": pair.baseline,
                "candidate": pair.candidate,
                "baseline_pass_rate": pair.baseline_pass_rate,
                "candidate_pass_rate": pair.candidate_pass_rate,
                "lift": pair.lift,
                "wins": pair.wins,
                "ties": pair.ties,
                "losses": pair.losses,
                "tokens": vars(pair.tokens),
                "duration_ms": vars(pair.duration_ms),
                "cost": vars(pair.cost),
            }
        return out

    def render(self) -> str:
        lines = [f"── {self.scenario} ──"]
        for variant in self.variants:
            line = (
                f"  {variant.name:12s} {variant.passed}/{variant.total} "
                f"({variant.pass_rate * 100:.0f}%)  "
                f"{variant.avg_duration_ms:.0f}ms  "
                f"{variant.avg_tokens:.0f}tok"
            )
            if variant.avg_judge_score is not None:
                line += f"  judge {variant.avg_judge_score:.2f}"
            if variant.avg_cost:
                line += f"  ${variant.avg_cost:.4f}"
            lines.append(line)
        pair = self.pair_summary()
        if pair is not None:
            sign = "+" if pair.lift >= 0 else ""
            lines.append(
                f"  ─ pair: {pair.candidate} vs {pair.baseline}  "
                f"lift {sign}{pair.lift * 100:.0f}pp  "
                f"wins {pair.wins} / ties {pair.ties} / losses {pair.losses}"
            )
            if pair.tokens.delta is not None:
                lines.append(
                    f"    tokens {pair.tokens.baseline_mean:.0f} → "
                    f"{pair.tokens.candidate_mean:.0f} "
                    f"(Δ{pair.tokens.delta:+.0f})  "
                    f"ms {pair.duration_ms.baseline_mean:.0f} → "
                    f"{pair.duration_ms.candidate_mean:.0f} "
                    f"(Δ{pair.duration_ms.delta:+.0f})"
                )
        return "\n".join(lines)


def run_comparison(
    scenarios: list[EvalScenario],
    *,
    variants: list[EvalVariant] | dict[str, dict[str, Any]] | None = None,
    repetitions: int = 3,
    artifacts_dir: str | Path | None = None,
    judge=None,
    judge_client=None,
    client=None,
    stream_fn=None,
    client_factory: Callable[[], tuple[Any, Any]] | None = None,
    parallel: bool = False,
) -> list[ComparisonReport]:
    """Run every scenario ``repetitions`` times per variant.

    Returns one :class:`ComparisonReport` per scenario.  ``variants`` is
    either a list of :class:`EvalVariant` (the first-class form — each
    variant builds its own CodingSession) or, for backward compatibility,
    a legacy ``dict[str, dict]`` mapping a variant name to extra
    ``run_scenario`` kwargs.  Defaults to a single ``candidate`` variant.

    ``client_factory`` (optional) returns a fresh ``(client, stream_fn)``
    per repetition — required for stateful scripted clients; real API
    clients are stateless and can be passed once via ``client``/``stream_fn``.

    ``parallel`` runs the (scenario, variant, repetition) jobs on a thread
    pool — safe now that ``run_scenario`` no longer chdir's the process.
    """
    normalized: list[tuple[EvalVariant, dict[str, Any]]] = []
    if variants is None:
        normalized = [(EvalVariant(name="candidate"), {})]
    elif isinstance(variants, dict):
        normalized = [
            (EvalVariant(name=name), dict(extra))
            for name, extra in variants.items()
        ]
    else:
        normalized = [(variant, {}) for variant in variants]

    artifacts_root = Path(artifacts_dir) if artifacts_dir else None
    reports: list[ComparisonReport] = []

    for scenario in scenarios:
        jobs = [
            (variant, extra, index)
            for variant, extra in normalized
            for index in range(max(1, repetitions))
        ]

        def run_one(job: tuple[EvalVariant, dict[str, Any], int]):
            variant, extra, index = job
            rep_dir = None
            if artifacts_root is not None:
                rep_dir = (
                    artifacts_root / scenario.name / variant.name / f"rep{index:02d}"
                )
            run_client, run_stream_fn = client, stream_fn
            if client_factory is not None:
                run_client, run_stream_fn = client_factory()
            result: ScenarioResult = run_scenario(
                scenario,
                artifacts_dir=rep_dir,
                judge=judge,
                judge_client=judge_client,
                client=run_client,
                stream_fn=run_stream_fn,
                variant=variant,
                **extra,
            )
            verdict = result.judge_verdict
            usage = result.usage or {}
            return (
                variant.name,
                index,
                RepetitionResult(
                    run_index=index,
                    passed=result.passed,
                    duration_ms=result.duration_ms,
                    failures=list(result.failures),
                    judge_score=verdict.score if verdict else None,
                    judge_reason=verdict.reason if verdict else "",
                    total_tokens=int(usage.get("total_tokens", 0) or 0),
                    cost=float(usage.get("total_cost", 0.0) or 0.0),
                ),
            )

        if parallel and len(jobs) > 1:
            # 上限钉住:场景 × 变体 × repetitions 多时不能无线开线程
            with ThreadPoolExecutor(max_workers=min(len(jobs), 8)) as executor:
                results = list(executor.map(run_one, jobs))
        else:
            results = [run_one(job) for job in jobs]

        # group by variant, preserving declaration order and rep order
        by_variant: dict[str, list[tuple[int, RepetitionResult]]] = {}
        for name, index, rep in results:
            by_variant.setdefault(name, []).append((index, rep))
        variant_runs = [
            VariantRun(
                name=variant.name,
                repetitions=[
                    rep for _, rep in sorted(by_variant[variant.name])
                ],
            )
            for variant, _extra in normalized
        ]
        reports.append(ComparisonReport(scenario=scenario.name, variants=variant_runs))

    return reports
