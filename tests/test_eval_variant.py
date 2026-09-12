"""Eval 2.0 variants: first-class A/B configuration (EvalVariant).

Covers the variant data model, settings/system-prompt/tool-policy application,
independent CodingSession construction, and ``run_comparison`` with an explicit
variant list — all deterministic (scripted client, no API key).
"""

from __future__ import annotations

import json

from lsm_harness.config import Settings
from lsm_harness.ops.eval import (
    EvalToolPolicy,
    EvalVariant,
    apply_tool_policy,
    apply_variant_settings,
    build_variant_session,
    run_comparison,
    run_scenario,
)
from lsm_harness.ops.eval.suites import core_scenarios

from helpers import QueueClient


def test_apply_variant_settings_writes_fields():
    s = Settings(home="/tmp/x")
    apply_variant_settings(
        s,
        EvalVariant(
            name="v",
            provider="kimi",
            model="k3",
            small_model="k3-small",
            system_prompt="自定义指令 XYZ",
            settings_overrides={"max_iterations": 7},
        ),
    )
    assert s.provider == "kimi"
    assert s.model == "k3"
    assert s.small_model == "k3-small"
    assert s.system_prompt == "自定义指令 XYZ"
    assert s.max_iterations == 7


def test_apply_variant_settings_empty_fields_do_not_overwrite():
    s = Settings(home="/tmp/x")
    s.provider = "deepseek"
    s.model = "deepseek-chat"
    apply_variant_settings(s, EvalVariant(name="v"))
    assert s.provider == "deepseek"
    assert s.model == "deepseek-chat"


def test_apply_tool_policy_filters_registry(tmp_path):
    client = QueueClient()
    harness = build_variant_session(
        EvalVariant(name="v"), home=tmp_path,
        client=client, stream_fn=client.as_stream_fn(),
    )
    try:
        all_names = harness.tools.tool_names()
        assert "read_file" in all_names and "exec" in all_names

        # deny removes a single tool; the agent state follows the registry.
        apply_tool_policy(harness, EvalToolPolicy(deny=["exec"]))
        assert "exec" not in harness.tools.tool_names()
        assert "exec" not in harness.agent.state.tools.tool_names()
        assert "read_file" in harness.tools.tool_names()

        # allow restricts to an explicit subset.
        apply_tool_policy(harness, EvalToolPolicy(allow=["read_file"]))
        assert harness.tools.tool_names() == ["read_file"]
    finally:
        harness.close()


def test_build_variant_session_injects_system_prompt(tmp_path):
    client = QueueClient()
    harness = build_variant_session(
        EvalVariant(
            name="baseline",
            provider="deepseek",
            model="m-baseline",
            system_prompt="这是 variant 的额外系统指令。",
        ),
        home=tmp_path,
        client=client,
        stream_fn=client.as_stream_fn(),
    )
    try:
        assert harness.settings.provider == "deepseek"
        assert harness.settings.model == "m-baseline"
        system = harness.session.build_system("", lambda *a, **k: None)
        assert "这是 variant 的额外系统指令。" in system
    finally:
        harness.close()


def test_tool_policy_changes_scenario_outcome():
    """deny 掉 exec 后,calc_bug 场景(脚本模型仍会调用 exec)失败。"""
    scenario = next(s for s in core_scenarios() if s.name == "calc_bug")
    passed = run_scenario(scenario)
    assert passed

    blocked = run_scenario(
        scenario,
        variant=EvalVariant(
            name="no-exec",
            tool_policy=EvalToolPolicy(deny=["exec"]),
        ),
    )
    assert not blocked.passed
    assert any("exec" in f for f in blocked.failures)


def test_comparison_with_explicit_variant_list(tmp_path):
    scenarios = [s for s in core_scenarios() if s.name == "read_and_answer"]
    variants = [
        EvalVariant(name="baseline", model="m-baseline"),
        EvalVariant(name="candidate", model="m-candidate"),
    ]
    reports = run_comparison(
        scenarios,
        variants=variants,
        repetitions=2,
        artifacts_dir=tmp_path,
    )
    assert len(reports) == 1
    report = reports[0]
    assert [v.name for v in report.variants] == ["baseline", "candidate"]
    for variant in report.variants:
        assert variant.total == 2
        assert variant.passed == 2

    # artifacts are nested per scenario / variant / rep
    for name in ("baseline", "candidate"):
        for rep in ("rep00", "rep01"):
            bundle = tmp_path / "read_and_answer" / name / rep
            assert bundle.is_dir()
            data = json.loads(
                (bundle / "result.json").read_text(encoding="utf-8")
            )
            assert data["name"] == "read_and_answer"


def test_comparison_legacy_dict_still_works(tmp_path):
    """旧 dict[str, dict] 形式(extra kwargs)保持向后兼容。"""
    scenarios = [s for s in core_scenarios() if s.name == "read_and_answer"]
    reports = run_comparison(
        scenarios,
        variants={"baseline": {}, "candidate": {}},
        repetitions=1,
        artifacts_dir=tmp_path,
    )
    report = reports[0]
    assert [v.name for v in report.variants] == ["baseline", "candidate"]
    for name in ("baseline", "candidate"):
        assert (tmp_path / "read_and_answer" / name / "rep00" / "result.json").exists()
