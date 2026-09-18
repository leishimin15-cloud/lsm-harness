"""Eval 2.0 non-scripted tasks (step 2).

A task declares a prompt + fixture but no model transcript; the agent (here a
scripted QueueClient standing in for a real model) drives itself, and
correctness is scored by file effects, verify commands and behavioural
assertions.  ``run_task_comparison`` proves the stability-rerun path.
"""

from __future__ import annotations

import json

from lsm_harness.ai.types import ModelResponse, ToolCall, Usage
from lsm_harness.ops.eval import (
    EvalFixture,
    EvalTask,
    EvalVariant,
    file_equals,
    run_task,
    run_task_comparison,
    task_to_scenario,
)
from lsm_harness.ops.eval.assertions import reply_contains
from lsm_harness.ops.eval.suites import core_real_scenarios, core_tasks

from helpers import QueueClient

_CALC_FIXED = "def add(a, b):\n    return a + b\n"


def _scripted_calc_client():
    client = QueueClient()
    client.responses.append(ModelResponse(
        tool_calls=[ToolCall("c1", "read_file", {"path": "calc.py"})],
        stop_reason="tool_calls", usage=Usage(10, 2),
    ))
    client.responses.append(ModelResponse(
        tool_calls=[ToolCall("c2", "write_file", {"path": "calc.py", "content": _CALC_FIXED})],
        stop_reason="tool_calls", usage=Usage(10, 2),
    ))
    client.responses.append(ModelResponse(
        tool_calls=[ToolCall("c3", "exec", {"command": "python test_calc.py"})],
        stop_reason="tool_calls", usage=Usage(10, 2),
    ))
    client.responses.append(ModelResponse(text="已修复，测试通过。", usage=Usage(10, 2)))
    return client


def test_task_to_scenario_is_unscripted():
    task = core_tasks()[0]
    scenario = task_to_scenario(task)
    assert scenario.use_real_api is True
    assert scenario.scripted_responses == []
    # steps: prompt → verify command
    assert [s.kind for s in scenario.steps] == ["prompt", "command"]
    assert scenario.steps[-1].text == "python test_calc.py"


def test_run_task_scores_by_file_effect_and_verify_command(tmp_path):
    task = next(t for t in core_tasks() if t.name == "fix_calc_add")
    client = _scripted_calc_client()

    result = run_task(
        task,
        client=client,
        stream_fn=client.as_stream_fn(),
        artifacts_dir=tmp_path / "fix_calc_add",
    )
    assert result.passed, result.failures
    # 行为断言:文件被改对 + exec 输出 OK
    assert any(c["tool"] == "exec" for c in result.tool_calls)
    assert result.artifacts is not None
    assert ("modified", "calc.py") in result.artifacts.changed_files

    data = json.loads(
        (tmp_path / "fix_calc_add" / "result.json").read_text(encoding="utf-8")
    )
    assert data["name"] == "fix_calc_add"
    assert data["status"] == "ok"


def test_run_task_fails_when_verify_command_fails(tmp_path):
    # 脚本 agent 只读文件、不改 calc.py → verify 命令 exit 非 0 → 失败
    task = next(t for t in core_tasks() if t.name == "fix_calc_add")
    client = QueueClient()
    client.responses.append(ModelResponse(
        tool_calls=[ToolCall("c1", "read_file", {"path": "calc.py"})],
        stop_reason="tool_calls", usage=Usage(10, 2),
    ))
    client.responses.append(ModelResponse(text="我看了，但没改。", usage=Usage(10, 2)))

    result = run_task(task, client=client, stream_fn=client.as_stream_fn())
    assert not result.passed
    assert any("step command" in f for f in result.failures)


def test_run_task_comparison_reruns_per_variant(tmp_path):
    task = next(t for t in core_tasks() if t.name == "report_answer_value")

    def make_client():
        client = QueueClient()
        client.responses.append(ModelResponse(
            tool_calls=[ToolCall("c1", "read_file", {"path": "answer.txt"})],
            stop_reason="tool_calls", usage=Usage(10, 2),
        ))
        client.responses.append(ModelResponse(text="答案是 42", usage=Usage(10, 2)))
        return client, client.as_stream_fn()

    reports = run_task_comparison(
        [task],
        variants=[
            EvalVariant(name="baseline", model="m-baseline"),
            EvalVariant(name="candidate", model="m-candidate"),
        ],
        repetitions=2,
        artifacts_dir=tmp_path,
        client_factory=make_client,
    )
    report = reports[0]
    assert [v.name for v in report.variants] == ["baseline", "candidate"]
    for variant in report.variants:
        assert variant.total == 2 and variant.passed == 2

    for name in ("baseline", "candidate"):
        for rep in ("rep00", "rep01"):
            assert (tmp_path / "report_answer_value" / name / rep / "result.json").exists()


def test_core_tasks_declare_no_scripted_responses():
    tasks = core_tasks()
    # 9 个任务 + core_real_scenarios() 的 1 个真模型场景 = 10 个真实评测项
    assert len(tasks) == 9
    for task in tasks:
        assert task.name and task.prompt
        # 非脚本:任务不携带任何 model 响应
        assert not hasattr(task, "scripted_responses")
        # 每个任务至少有一种可观察判定:行为断言或 verify 命令
        assert task.assertions or task.verify_commands, (
            f"{task.name} 需要断言或 verify_commands"
        )


def test_core_real_scenarios_are_unscripted_and_multi_step():
    scenarios = core_real_scenarios()
    assert scenarios, "core_real_scenarios 不能为空"
    for scenario in scenarios:
        assert scenario.use_real_api is True
        assert scenario.scripted_responses == []
        assert len(scenario.steps) > 1
        assert scenario.assertions


def test_new_task_fixtures_are_self_consistent():
    """脚本 agent 直接写出正确答案 → verify 必须通过。

    在真实模型花钱跑之前,先离线证明 fixture + verify 命令自洽
    (答案本身正确、命令写法正确)。
    """
    cases = {
        "count_csv_rows": [("count.txt", "10\n")],
        "json_filter_transform": [("result.json", '["bob", "cy"]\n')],
        "heal_broken_test": [("mather.py", "def add(a, b):\n    return a + b\n")],
        "rename_function_across_files": [
            ("ops.py", "def compute(a, b):\n    return a + b + 1\n"),
            ("main.py", "from ops import compute\n\n\ndef run():\n    return compute(2, 2)\n"),
        ],
        "write_test_for_util": [
            ("test_even.py", "from even import is_even\n\nassert is_even(2)\nassert not is_even(3)\n"),
        ],
    }
    for name, writes in cases.items():
        task = next(t for t in core_tasks() if t.name == name)
        client = QueueClient()
        for index, (path, content) in enumerate(writes):
            client.responses.append(ModelResponse(
                tool_calls=[ToolCall(
                    f"w{index}", "write_file", {"path": path, "content": content}
                )],
                stop_reason="tool_calls", usage=Usage(10, 2),
            ))
        client.responses.append(ModelResponse(text="完成", usage=Usage(10, 2)))
        result = run_task(task, client=client, stream_fn=client.as_stream_fn())
        assert result.passed, f"{name}: {result.failures}"


def test_parse_overrides_type_coercion():
    from lsm_harness.__main__ import _parse_overrides

    assert _parse_overrides([
        "context_keep_recent_tokens=1000000",
        "governance_offload_threshold=99999999",
        "system_prompt=hello",
        "ratio=0.5",
    ]) == {
        "context_keep_recent_tokens": 1000000,
        "governance_offload_threshold": 99999999,
        "system_prompt": "hello",
        "ratio": 0.5,
    }


def test_parse_overrides_rejects_malformed():
    import pytest

    from lsm_harness.__main__ import _parse_overrides

    with pytest.raises(ValueError):
        _parse_overrides(["no-equals-sign"])
    with pytest.raises(ValueError):
        _parse_overrides(["=1"])
