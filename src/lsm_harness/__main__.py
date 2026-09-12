"""`lsm` CLI entry: chat (default), tui, rpc, -p print, diagnostics."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="lsm", description="LSM 的个人 Agent Harness")
    parser.add_argument(
        "-p", "--print",
        dest="print_prompt",
        metavar="PROMPT",
        help="一次性问答：回复流式打到 stdout，工具活动打到 stderr",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="环境自检")
    sub.add_parser("smoke", help="确定性冒烟测试")
    sub.add_parser("traces", help="查看最近的 trace 文件")
    tui_p = sub.add_parser("tui", help="启动 Textual TUI 界面")
    tui_p.add_argument(
        "--mode",
        choices=["regular", "fullscreen"],
        default="",
        help="fullscreen(默认,稳定的 Textual 全屏模式);"
        "regular 目前作兼容别名，等待独立 main-screen renderer",
    )
    sub.add_parser("rpc", help="JSONL RPC 模式（stdin/stdout，供编辑器集成）")
    eval_p = sub.add_parser("eval", help="运行 eval 测试套件")
    eval_p.add_argument("--suite", default="", help="指定套件 (tools/safety/core/tasks/all)")
    eval_p.add_argument("--record", action="store_true", help="记录 golden traces")
    eval_p.add_argument("--compare", action="store_true", help="比较模式（多次重复，统计通过率）")
    eval_p.add_argument("--repetitions", type=int, default=3, help="比较模式的重复次数")
    eval_p.add_argument("--artifacts-dir", default="", help="产物输出目录（仅 core）")
    eval_p.add_argument(
        "--judge", choices=["deterministic", "model"], default="deterministic",
        help="评分方式（model 需要真实 API key）",
    )
    eval_p.add_argument("--parallel", action="store_true", help="比较模式并行跑 repetition")
    eval_p.add_argument("--baseline", default="", help="baseline variant 'provider/model'")
    eval_p.add_argument("--candidate", default="", help="candidate variant 'provider/model'")
    eval_p.add_argument("--baseline-prompt", default="", help="baseline 附加 system prompt")
    eval_p.add_argument("--candidate-prompt", default="", help="candidate 附加 system prompt")
    eval_p.add_argument("--baseline-deny-tools", default="", help="baseline 禁用工具(逗号分隔)")
    eval_p.add_argument("--candidate-deny-tools", default="", help="candidate 禁用工具(逗号分隔)")
    args = parser.parse_args()

    if args.print_prompt is not None:
        from lsm_harness.gateway.print_mode import run_print
        raise SystemExit(run_print(args.print_prompt))
    if args.command == "doctor":
        from lsm_harness.doctor import run
        raise SystemExit(run())
    if args.command == "smoke":
        from lsm_harness.smoke import run
        raise SystemExit(run())
    if args.command == "tui":
        from lsm_harness.gateway.tui import run_tui
        run_tui(mode=args.mode or None)
        return
    if args.command == "rpc":
        from lsm_harness.gateway.rpc import run_rpc
        raise SystemExit(run_rpc())
    if args.command == "traces":
        from lsm_harness.ops.tracing import list_recent_traces
        list_recent_traces()
        return
    if args.command == "eval":
        if args.suite in ("core", "tasks", "all"):
            raise SystemExit(_run_core_evals(args))
        from lsm_harness.ops.eval import run_evals
        raise SystemExit(run_evals(suite_name=args.suite, record=args.record))
    from lsm_harness.coding_agent.cli import run_chat
    raise SystemExit(run_chat())


def _run_core_evals(args) -> int:
    """Run the Eval 2.0 scenarios/tasks (optionally as a real A/B comparison).

    ``--suite core`` = 确定性脚本场景；``--suite tasks`` = 非脚本任务
    (真实模型自主执行，ModelRuntime 从 variant 的 provider/model 解析)；
    ``--suite all`` = 两者。``--compare`` + ``--baseline/--candidate`` 构成
    真 A/B：两个不同的 EvalVariant 分别构建自己的 CodingSession。
    """
    from pathlib import Path

    from lsm_harness.ops.eval.compare import run_comparison
    from lsm_harness.ops.eval.runner import run_scenario
    from lsm_harness.ops.eval.suites import core_scenarios, core_tasks
    from lsm_harness.ops.eval.task import task_to_scenario
    from lsm_harness.ops.eval.variant import EvalToolPolicy, EvalVariant

    def _variant(name: str, spec: str, prompt: str, deny: str) -> EvalVariant:
        provider, _, model = (spec or "").partition("/")
        deny_list = [t.strip() for t in deny.split(",") if t.strip()]
        return EvalVariant(
            name=name,
            provider=provider.strip(),
            model=model.strip(),
            system_prompt=prompt or "",
            tool_policy=EvalToolPolicy(deny=deny_list) if deny_list else None,
        )

    def _validate_spec(flag: str, spec: str, *, real: bool) -> str | None:
        """校验 provider/model 写法;非法返回错误文案。"""
        if not spec:
            return None
        provider, _, model = spec.partition("/")
        if not provider or not model:
            return f"{flag} 格式应为 provider/model(如 kimi/k3),收到: {spec!r}"
        if real:
            from lsm_harness.coding_agent.model_config import load_model_catalog
            from lsm_harness.config import Settings
            providers = load_model_catalog(Settings().home).providers
            if provider not in providers:
                return (f"{flag} 未知 provider {provider!r},"
                        f"可选: {', '.join(providers)}")
        return None

    suite = args.suite or "core"
    has_real = suite in ("tasks", "all")
    for flag, spec in (("--baseline", args.baseline), ("--candidate", args.candidate)):
        err = _validate_spec(flag, spec, real=has_real)
        if err:
            print(f"[error] {err}")
            return 2

    scenarios = []
    if suite in ("core", "all"):
        scenarios.extend(core_scenarios())
    if suite in ("tasks", "all"):
        scenarios.extend(task_to_scenario(t) for t in core_tasks())

    # 模式区分:core 是脚本回归(不调用真实 API,variant 的 provider/model
    # 只记入 provenance);tasks 才是模型行为评测。避免误读结果。
    if suite == "core":
        print("[mode] 确定性回归(脚本模型,不调用真实 API)")
        if args.baseline or args.candidate:
            print("[note] core 为脚本场景:variant 的 provider/model 只记入 "
                  "provenance;真实模型对比请用 --suite tasks")
    else:
        print("[mode] 真实模型行为评测(调用真实 API)")

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else None

    judge_client = None
    if args.judge == "model":
        from lsm_harness.ai.providers import get_client
        judge_client = get_client()

    if args.compare:
        variants = [
            _variant("baseline", args.baseline, args.baseline_prompt,
                     args.baseline_deny_tools),
            _variant("candidate", args.candidate, args.candidate_prompt,
                     args.candidate_deny_tools),
        ]
        if (
            (args.baseline, args.baseline_prompt, args.baseline_deny_tools)
            == (args.candidate, args.candidate_prompt, args.candidate_deny_tools)
        ):
            print("[warn] baseline 与 candidate 配置完全相同——"
                  "比较结果仅衡量重复稳定性,不构成 A/B")
        reports = run_comparison(
            scenarios,
            variants=variants,
            repetitions=args.repetitions,
            artifacts_dir=artifacts_dir,
            judge_client=judge_client,
            parallel=args.parallel,
        )
        for report in reports:
            print(report.render())
        total_passed = sum(v.passed for r in reports for v in r.variants)
        total_runs = sum(v.total for r in reports for v in r.variants)
        print(f"\n总计 {total_passed}/{total_runs} 通过")
        return 0 if total_passed == total_runs else 1

    # 单次模式:用 candidate 侧 CLI 配置(若有)作为唯一 variant。
    candidate = _variant("candidate", args.candidate, args.candidate_prompt,
                         args.candidate_deny_tools)
    variant = candidate if any(
        (args.candidate, args.candidate_prompt, args.candidate_deny_tools)
    ) else None
    results = []
    for scenario in scenarios:
        out_dir = artifacts_dir / scenario.name if artifacts_dir else None
        result = run_scenario(
            scenario,
            artifacts_dir=out_dir,
            judge_client=judge_client,
            variant=variant,
        )
        results.append(result)
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}"
              f"  ({result.duration_ms:.0f}ms)")
        for failure in result.failures:
            print(f"      - {failure}")

    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    main()
