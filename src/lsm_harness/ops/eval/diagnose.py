"""Deterministic failure attribution for the Diagnose stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TYPE_CHECKING

from lsm_harness.ops.eval.observe import RunObservation

if TYPE_CHECKING:
    from lsm_harness.ops.eval.artifacts import EvalRunArtifacts


@dataclass
class RunDiagnosis:
    run_id: str
    result: str
    category: str
    failure_stage: str
    first_error: str = ""
    tool: str = ""
    retries: int = 0
    self_corrected: bool = False
    likely_causes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_artifacts(
    artifacts: "EvalRunArtifacts",
    observation: RunObservation,
) -> RunDiagnosis:
    failed_tools = [call for call in artifacts.tool_calls if call.get("is_error")]
    first_error = ""
    tool = ""
    if failed_tools:
        tool = str(failed_tools[0].get("tool", ""))
        first_error = str(failed_tools[0].get("output", ""))[:500]
    if not first_error and artifacts.failures:
        first_error = artifacts.failures[0][:500]
    if not first_error and observation.error_events:
        first_error = str(observation.error_events[0].get("data", ""))[:500]

    passed = artifacts.status == "ok"
    self_corrected = passed and bool(failed_tools)
    haystack = " ".join([first_error, *artifacts.failures]).lower()
    category, stage, causes = _classify(haystack, observation, passed)
    if self_corrected:
        category = "recovered_tool_error"
        stage = "tool.execute"
        causes = ["工具错误已返回模型，后续步骤完成了任务。"]
    elif passed:
        category, stage, causes = "none", "", []

    return RunDiagnosis(
        run_id=observation.run_id,
        result="passed" if passed else "failed",
        category=category,
        failure_stage=stage,
        first_error=first_error,
        tool=tool,
        retries=observation.retries,
        self_corrected=self_corrected,
        likely_causes=causes,
    )


def _classify(
    text: str,
    observation: RunObservation,
    passed: bool,
) -> tuple[str, str, list[str]]:
    rules = [
        (("401", "authentication", "api key"), "authentication_error", "provider.auth"),
        (("context length", "length_exhausted", "overflow"), "context_overflow", "context.build"),
        (("invalid arguments", "schema", "required property"), "schema_validation_error", "tool.validate"),
        (("approval", "rejected"), "approval_rejected", "tool.approval"),
        (("repeated", "连续"), "repeated_tool_error", "tool.execute"),
        (("max_iterations", "maximum iterations", "最大迭代"), "max_iterations", "agent.loop"),
        (("provider", "model unavailable", "rate limit", "429"), "provider_error", "model.call"),
        (("command", "exit code", "returned non-zero"), "command_failure", "verify.command"),
    ]
    for needles, category, stage in rules:
        if any(needle in text for needle in needles):
            return category, stage, [f"首个可验证错误位于 {stage}。"]
    if observation.tool_errors:
        return "tool_execution_error", "tool.execute", ["工具执行失败且任务未恢复。"]
    if not passed:
        return "outcome_failure", "eval.assertion", ["最终状态或断言未满足场景要求。"]
    return "none", "", []
