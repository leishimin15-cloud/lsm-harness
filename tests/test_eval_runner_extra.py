"""Eval 2.0 step-3 extras: prompt timeout, artifact provenance, judge scores
in the comparison report, typed tool-call collection, and parallel runs.
"""

from __future__ import annotations

import time

from lsm_harness.ai.api.common import snapshot
from lsm_harness.ai.types import AssistantMessageEvent
from lsm_harness.ops.eval import (
    EvalScenario,
    EvalVariant,
    prompt,
    run_comparison,
    run_scenario,
)
from lsm_harness.ops.eval.suites import core_scenarios

from helpers import QueueClient


def _hanging_stream(_model, _context, options):
    """Block until aborted (mirrors the runner's parked stream)."""
    while not (options.interrupt and options.interrupt.is_set()):
        time.sleep(0.005)
    yield AssistantMessageEvent(
        "error",
        snapshot(text="", thinking="", pending={}, stop_reason="aborted"),
        error_category="aborted",
    )


def test_prompt_step_timeout():
    scenario = EvalScenario(
        name="timeout",
        steps=[prompt("卡住的任务", timeout=0.2)],
        use_real_api=True,
    )
    result = run_scenario(
        scenario,
        client=QueueClient(),
        stream_fn=_hanging_stream,
    )
    assert not result.passed
    assert any("timed out" in f for f in result.failures)
    # 超时后 step 标记 error(不是悄悄挂死)
    assert result.step_results[0].status == "error"


def test_artifacts_carry_provenance():
    scenario = next(s for s in core_scenarios() if s.name == "read_and_answer")
    result = run_scenario(
        scenario,
        variant=EvalVariant(
            name="baseline", provider="kimi", model="k3",
            settings_overrides={"max_iterations": 9},
        ),
    )
    artifacts = result.artifacts
    assert artifacts is not None
    assert artifacts.model == "k3"
    assert artifacts.provider == "kimi"
    assert artifacts.config["model"] == "k3"
    assert artifacts.config["max_iterations"] == 9
    # git revision 是 40 位 hex(在仓库内运行)
    assert len(artifacts.git_revision) == 40
    int(artifacts.git_revision, 16)

    d = artifacts.as_dict()
    assert d["model"] == "k3"
    assert d["provider"] == "kimi"
    assert d["git_revision"] == artifacts.git_revision


def test_typed_tool_calls_carry_args_and_output():
    scenario = next(s for s in core_scenarios() if s.name == "read_and_answer")
    result = run_scenario(scenario)
    calls = [c for c in result.tool_calls if c["tool"] == "read_file"]
    assert calls, "typed collector should capture tool calls"
    call = calls[0]
    assert call["args"] == {"path": "answer.txt"}
    assert "42" in call["output"]


def test_result_carries_iterations_and_tool_error_count():
    """第 0 步 metrics:iterations 从 usage by_model.calls 聚合,
    tool_error_count 从 tool_calls 的 is_error 推导。"""
    scenario = next(s for s in core_scenarios() if s.name == "self_correct")
    result = run_scenario(scenario)
    assert result.passed
    # 脚本:read 错 → read 对 → 回复 = 3 次 LLM 调用
    assert result.iterations == 3
    # 第一次 read_file 路径不存在 → is_error
    assert result.tool_error_count == 1

    reports = run_comparison([scenario], repetitions=1)
    rep = reports[0].variants[0].repetitions[0]
    assert rep.iterations == 3
    assert rep.tool_errors == 1
    # 离线脚本不产生 cache token
    assert rep.cache_hit_rate == 0.0
    d = reports[0].as_dict()
    assert "cache_hit_rate" in d["variants"][0]
    assert "p95_duration_ms" in d["variants"][0]


def test_comparison_report_includes_judge_score():
    scenario = next(s for s in core_scenarios() if s.name == "read_and_answer")
    reports = run_comparison([scenario], repetitions=2)
    variant = reports[0].variants[0]
    # DeterministicJudge: 全过 = score 1.0
    assert variant.avg_judge_score == 1.0
    assert "judge 1.00" in reports[0].render()
    assert reports[0].as_dict()["variants"][0]["avg_judge_score"] == 1.0


def test_parallel_comparison_matches_serial():
    scenarios = [
        s for s in core_scenarios() if s.name in ("read_and_answer", "calc_bug")
    ]
    variants = [EvalVariant(name="a"), EvalVariant(name="b")]

    serial = run_comparison(scenarios, variants=variants, repetitions=2, parallel=False)
    parallel = run_comparison(scenarios, variants=variants, repetitions=2, parallel=True)

    def stability(reports):
        # 只比较确定性指标(耗时因并行而不同,不纳入比较)
        return {
            r.scenario: {
                v.name: (v.passed, v.total, v.avg_judge_score)
                for v in r.variants
            }
            for r in reports
        }

    assert stability(serial) == stability(parallel)
    for report in parallel:
        for variant in report.variants:
            assert variant.passed == variant.total == 2


# ── 第 1-3 步收尾:真 A/B 接线、成对指标、实现隐患 ──────


def test_pair_summary_lift_and_wins():
    """两个 variant 一成一败:lift/wins/losses/ties 与指标 delta 正确。"""
    from lsm_harness.ops.eval import EvalToolPolicy

    scenario = next(s for s in core_scenarios() if s.name == "calc_bug")
    reports = run_comparison(
        [scenario],
        variants=[
            EvalVariant(name="baseline"),  # 正常通过
            # candidate 禁 exec → 验证命令失败
            EvalVariant(
                name="candidate",
                tool_policy=EvalToolPolicy(deny=["exec"]),
            ),
        ],
        repetitions=2,
    )
    report = reports[0]
    pair = report.pair_summary()
    assert pair is not None
    assert pair.baseline == "baseline" and pair.candidate == "candidate"
    assert pair.baseline_pass_rate == 1.0
    assert pair.candidate_pass_rate == 0.0
    assert pair.lift == -1.0
    assert pair.wins == 0 and pair.losses == 2 and pair.ties == 0
    # tokens/duration 成对 delta 都有值
    assert pair.tokens.baseline_mean is not None
    assert pair.tokens.delta is not None
    d = report.as_dict()
    assert d["pair"]["losses"] == 2
    assert "lift -100pp" in report.render()


def test_pair_summary_none_for_non_pair():
    """单 variant 或三 variant 不出成对摘要。"""
    scenario = next(s for s in core_scenarios() if s.name == "read_and_answer")
    single = run_comparison([scenario], repetitions=1)[0]
    assert single.pair_summary() is None
    three = run_comparison(
        [scenario],
        variants=[EvalVariant(name=n) for n in ("a", "b", "c")],
        repetitions=1,
    )[0]
    assert three.pair_summary() is None


def test_real_api_setup_failure_is_failed_run_not_crash():
    """真实场景 + 不存在的 provider:返回 failed ScenarioResult,不抛异常。"""
    scenario = EvalScenario(
        name="real-missing-provider",
        steps=[prompt("随便做点啥")],
        use_real_api=True,
        assertions=[],
    )
    result = run_scenario(
        scenario,
        variant=EvalVariant(name="v", provider="no-such-provider", model="x"),
    )
    assert result.passed is False
    assert any("setup" in f for f in result.failures)


def test_real_api_falls_back_to_real_home_credentials(monkeypatch, tmp_path):
    """eval 用临时 home(没有 auth.json):必须按 variant 的 provider 从
    真实 home 重新解析凭证并注入 settings——不能沿用 Settings 默认从
    .env 解析到的其他 provider 的 key(实测:DEEPSEEK_API_KEY 被发给
    kimi 端点 → 401)。"""
    from lsm_harness.ai.types import ModelResponse
    from lsm_harness.coding_agent import startup
    from lsm_harness.config import Settings

    calls = {}

    def fake_resolve(provider, *, home, explicit="", catalog=None):
        calls["provider"] = provider
        calls["home"] = home
        calls["explicit"] = explicit
        return "stored-key"

    monkeypatch.setattr(startup, "resolve_product_api_key", fake_resolve)

    client = QueueClient()
    client.responses.append(ModelResponse(text="ok"))
    scenario = EvalScenario(
        name="real-auth-fallback",
        steps=[prompt("hi")],
        use_real_api=True,
        assertions=[],
    )
    result = run_scenario(
        scenario,
        # 只注入 stream_fn(免网络);client=None 走真实 key 解析路径
        stream_fn=client.as_stream_fn(),
        variant=EvalVariant(name="v", provider="kimi-coding", model="k3"),
    )
    # 即使 env 里有 DEEPSEEK_API_KEY(Settings 默认解析),也要按 variant
    # 的 provider 重新解析,且不能把旧 key 当 explicit 传进去
    assert calls["provider"] == "kimi-coding"
    assert not calls["explicit"]
    # 必须读真实 home,不是 eval 的临时 home
    assert calls["home"] == Settings().home
    assert "lsm-eval-home-" not in str(calls["home"])
    assert result.passed, result.failures


def test_zero_tool_variant_not_misdetected_as_summarizer():
    """P2 隐患:variant 禁用全部工具后,普通调用 tools=[]——
    不能因此被 ScenarioClient 误判为 compaction summarizer(那样会
    返回 summary_text 而不是脚本响应,场景必败)。"""
    from lsm_harness.ops.eval import EvalToolPolicy

    scenario = next(
        s for s in core_scenarios() if s.name == "compaction_remembers_goal"
    )
    result = run_scenario(
        scenario,
        variant=EvalVariant(
            name="no-tools",
            tool_policy=EvalToolPolicy(allow=["nonexistent_tool"]),
        ),
    )
    # 正常调用 system 非空 → 弹脚本响应;compaction summarizer system="" → 摘要
    assert result.passed, result.failures


def test_git_revision_prefers_module_repo_over_cwd(tmp_path, monkeypatch):
    """git revision 记录被评测代码所在仓库,与调用者 cwd 无关。"""
    from pathlib import Path

    from lsm_harness.ops.eval import runner as eval_runner

    module_root = eval_runner._find_git_root(Path(eval_runner.__file__).resolve())
    assert module_root is not None and (module_root / "pyproject.toml").exists()

    # 从无关目录跑已安装的 lsm:revision 仍是本仓库,不是 cwd
    monkeypatch.chdir(tmp_path)
    scenario = next(s for s in core_scenarios() if s.name == "read_and_answer")
    result = run_scenario(scenario)
    assert result.artifacts is not None
    assert len(result.artifacts.git_revision) == 40


def test_cli_rejects_malformed_variant_spec(capsys):
    """--compare 的 spec 缺 /model 直接报错(exit 2),不进入跑批。"""
    from types import SimpleNamespace

    from lsm_harness.__main__ import _run_core_evals

    args = SimpleNamespace(
        suite="core", compare=True, repetitions=1, artifacts_dir="",
        judge="deterministic", parallel=False,
        baseline="openai", candidate="",  # 缺 /model
        baseline_prompt="", candidate_prompt="",
        baseline_deny_tools="", candidate_deny_tools="",
        baseline_override=[], candidate_override=[],
    )
    assert _run_core_evals(args) == 2
    assert "provider/model" in capsys.readouterr().out


def test_cli_warns_when_variants_identical(capsys, tmp_path):
    """baseline == candidate 时明确警告:只是稳定性测量,不构成 A/B。"""
    from types import SimpleNamespace

    from lsm_harness.__main__ import _run_core_evals

    args = SimpleNamespace(
        suite="core", compare=True, repetitions=1,
        artifacts_dir=str(tmp_path),
        judge="deterministic", parallel=False,
        baseline="kimi/k3", candidate="kimi/k3",
        baseline_prompt="", candidate_prompt="",
        baseline_deny_tools="", candidate_deny_tools="",
        baseline_override=[], candidate_override=[],
    )
    assert _run_core_evals(args) == 0
    out = capsys.readouterr().out
    assert "mode=offline" in out          # 模式横幅:core 是脚本回归
    assert "配置完全相同" in out          # 同配置警告
    assert "provenance only" in out      # provenance 提示


def test_cli_tasks_banner_marks_real_model_eval(capsys):
    """tasks 套件横幅明确标出真实模型评测。"""
    from types import SimpleNamespace

    from lsm_harness.__main__ import _run_core_evals

    args = SimpleNamespace(
        suite="tasks", compare=False, repetitions=1, artifacts_dir="",
        judge="deterministic", parallel=False,
        baseline="", candidate="no-such-provider/x",
        baseline_prompt="", candidate_prompt="",
        baseline_deny_tools="", candidate_deny_tools="",
        baseline_override=[], candidate_override=[],
    )
    # 未知 provider 在真实套件下直接报错(exit 2),不打真实 API
    assert _run_core_evals(args) == 2
    assert "未知 provider" in capsys.readouterr().out


def test_cli_requires_an_explicit_offline_or_real_mode(capsys):
    """Bare `lsm eval` must not silently spend money or run a legacy suite."""
    from types import SimpleNamespace

    from lsm_harness.__main__ import _run_core_evals

    args = SimpleNamespace(
        suite="", offline=False, provider="", model="", compare=False,
        repetitions=1, artifacts_dir="", judge="deterministic", parallel=False,
        baseline="", candidate="", baseline_prompt="", candidate_prompt="",
        baseline_deny_tools="", candidate_deny_tools="",
        baseline_override=[], candidate_override=[],
    )
    assert _run_core_evals(args) == 2
    assert "--offline" in capsys.readouterr().out


def test_cli_comparison_requires_both_variants(capsys):
    from types import SimpleNamespace

    from lsm_harness.__main__ import _run_core_evals

    args = SimpleNamespace(
        suite="core", offline=False, provider="", model="", compare=True,
        repetitions=1, artifacts_dir="", judge="deterministic", parallel=False,
        baseline="kimi/k3", candidate="", baseline_prompt="",
        candidate_prompt="", baseline_deny_tools="",
        candidate_deny_tools="",
        baseline_override=[], candidate_override=[],
    )
    assert _run_core_evals(args) == 2
    assert "--baseline" in capsys.readouterr().out


def test_variant_provider_switch_clears_stale_small_model():
    """variant 切 provider 但没给 small_model:必须清空 settings 里的
    默认小模型(可能属于另一个 provider,如 .env 的 deepseek-v4-flash),
    让 ModelRuntime 回填新 provider 的默认——否则压缩摘要会把
    deepseek-v4-flash 发给 kimi 端点(与 api_key 同类陷阱)。"""
    from lsm_harness.config import Settings
    from lsm_harness.ops.eval.variant import apply_variant_settings

    settings = Settings(small_model="deepseek-v4-flash")
    apply_variant_settings(
        settings, EvalVariant(name="v", provider="kimi-coding", model="k3")
    )
    assert settings.small_model == ""

    # 显式给的 small_model 必须保留
    settings = Settings(small_model="deepseek-v4-flash")
    apply_variant_settings(
        settings,
        EvalVariant(name="v", provider="kimi-coding", model="k3",
                    small_model="kimi-for-coding"),
    )
    assert settings.small_model == "kimi-for-coding"

    # 不切 provider/model 时不动
    settings = Settings(small_model="deepseek-v4-flash")
    apply_variant_settings(settings, EvalVariant(name="v"))
    assert settings.small_model == "deepseek-v4-flash"
