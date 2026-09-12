"""Non-scripted eval tasks (Eval 2.0 step 2).

A :class:`EvalTask` is the *unscripted* sibling of :class:`EvalScenario`:
it declares a task prompt and a starting fixture, but **no** model responses
and no tool-call sequence.  The agent (a real model) decides what to read,
change and run; correctness is scored by observable effects — workspace
files, verify commands and behavioural assertions — not by matching a
pre-written transcript.  ``run_task_comparison`` reruns the task ``n`` times
per variant so a change can be measured for both correctness and flakiness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lsm_harness.ai.types import ModelClient, StreamFunction

from lsm_harness.ops.eval.assertions import Assertion
from lsm_harness.ops.eval.compare import ComparisonReport, run_comparison
from lsm_harness.ops.eval.fixture import EvalFixture
from lsm_harness.ops.eval.runner import ScenarioResult, run_scenario
from lsm_harness.ops.eval.scenario import EvalScenario, command, prompt
from lsm_harness.ops.eval.variant import EvalVariant


@dataclass
class EvalTask:
    """A real-agent task: task prompt + fixture + observable assertions.

    ``verify_commands`` run after the prompt inside the workspace; a non-zero
    exit fails the run (e.g. ``python test_calc.py``).  ``setup_commands`` run
    before the prompt (e.g. ``pip install -r requirements.txt``).
    """

    name: str
    prompt: str
    description: str = ""
    fixture: EvalFixture | None = None
    assertions: list[Assertion] = field(default_factory=list)
    verify_commands: list[str] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)
    max_iterations: int = 20
    settings_overrides: dict[str, Any] = field(default_factory=dict)


def task_to_scenario(task: EvalTask) -> EvalScenario:
    """Translate a task into an unscripted (``use_real_api``) scenario."""
    steps = [command(c) for c in task.setup_commands]
    steps.append(prompt(task.prompt))
    steps.extend(command(c) for c in task.verify_commands)
    return EvalScenario(
        name=task.name,
        description=task.description,
        fixture=task.fixture,
        steps=steps,
        use_real_api=True,
        max_iterations=task.max_iterations,
        settings_overrides=task.settings_overrides,
        assertions=task.assertions,
    )


def run_task(
    task: EvalTask,
    *,
    client: ModelClient,
    stream_fn: StreamFunction,
    variant: EvalVariant | None = None,
    artifacts_dir: str | Path | None = None,
    judge=None,
    judge_client: ModelClient | None = None,
) -> ScenarioResult:
    """Run one task once against a real (or injected) client."""
    return run_scenario(
        task_to_scenario(task),
        client=client,
        stream_fn=stream_fn,
        variant=variant,
        artifacts_dir=artifacts_dir,
        judge=judge,
        judge_client=judge_client,
    )


def run_task_comparison(
    tasks: list[EvalTask],
    *,
    variants: list[EvalVariant] | dict[str, dict[str, Any]] | None = None,
    repetitions: int = 3,
    artifacts_dir: str | Path | None = None,
    judge=None,
    judge_client: ModelClient | None = None,
    client: ModelClient | None = None,
    stream_fn: StreamFunction | None = None,
    client_factory: Callable[[], tuple[ModelClient, StreamFunction]] | None = None,
    parallel: bool = False,
) -> list[ComparisonReport]:
    """Run every task ``repetitions`` times per variant (stability stats)."""
    return run_comparison(
        [task_to_scenario(t) for t in tasks],
        variants=variants,
        repetitions=repetitions,
        artifacts_dir=artifacts_dir,
        judge=judge,
        judge_client=judge_client,
        client=client,
        stream_fn=stream_fn,
        client_factory=client_factory,
        parallel=parallel,
    )
