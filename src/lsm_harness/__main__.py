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
    eval_p = sub.add_parser("eval", help="运行 Agent Harness 评测")
    eval_p.add_argument(
        "--offline", action="store_true",
        help="运行确定性离线回归（不调用真实 API）",
    )
    eval_p.add_argument("--provider", default="", help="真实评测 Provider")
    eval_p.add_argument("--model", default="", help="真实评测 Model")
    eval_p.add_argument("--suite", default="", help="指定套件 (tools/safety/core/tasks/all)")
    eval_p.add_argument("--record", action="store_true", help="记录 golden traces")
    eval_p.add_argument("--compare", action="store_true", help="比较模式（多次重复，统计通过率）")
    eval_p.add_argument("--repetitions", type=int, default=3, help="比较模式的重复次数")
    eval_p.add_argument(
        "--artifacts-dir", default="",
        help="产物输出目录（默认 .lsm/evals/<run-id>）",
    )
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
    eval_p.add_argument(
        "--baseline-override", action="append", default=[],
        metavar="KEY=VALUE",
        help="baseline 的 Settings 覆盖(可重复,如 context_keep_recent_tokens=1000000)",
    )
    eval_p.add_argument(
        "--candidate-override", action="append", default=[],
        metavar="KEY=VALUE",
        help="candidate 的 Settings 覆盖(可重复)",
    )
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
        if (
            args.offline
            or args.provider
            or args.model
            or args.compare
            or args.suite in ("core", "tasks", "all", "")
        ):
            raise SystemExit(_run_core_evals(args))
        from lsm_harness.ops.eval import run_evals
        raise SystemExit(run_evals(suite_name=args.suite, record=args.record))
    from lsm_harness.coding_agent.cli import run_chat
    raise SystemExit(run_chat())


def _parse_overrides(items: list[str]) -> dict:
    """把 CLI 的 key=value 列表解析成 settings_overrides。

    值做最小类型推断:纯数字转 int,带小数点转 float,其余保持字符串。
    """
    overrides: dict = {}
    for item in items:
        key, separator, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key:
            raise ValueError(f"override 格式应为 key=value,收到: {item!r}")
        if value.lstrip("-").isdigit():
            overrides[key] = int(value)
        else:
            try:
                overrides[key] = float(value)
            except ValueError:
                overrides[key] = value
    return overrides


def _run_core_evals(args) -> int:
    """Run deterministic regressions or real-model harness experiments.

    The CLI keeps offline correctness checks separate from paid model runs.
    Every invocation gets a durable artifact directory and run ledger.
    """

    from lsm_harness.ops.eval.compare import run_comparison
    from lsm_harness.ops.eval.reporter import (
        create_eval_artifact_dir,
        render_comparison_report,
        render_results,
        write_comparison_run_files,
        write_single_run_files,
    )
    from lsm_harness.ops.eval.runner import run_scenario
    from lsm_harness.ops.eval.suites import (
        core_real_scenarios,
        core_scenarios,
        core_tasks,
    )
    from lsm_harness.ops.eval.task import task_to_scenario
    from lsm_harness.ops.eval.variant import EvalToolPolicy, EvalVariant

    def _variant(
        name: str, spec: str, prompt: str, deny: str,
        overrides: list[str] | None = None,
    ) -> EvalVariant:
        provider, _, model = (spec or "").partition("/")
        deny_list = [t.strip() for t in deny.split(",") if t.strip()]
        return EvalVariant(
            name=name,
            provider=provider.strip(),
            model=model.strip(),
            system_prompt=prompt or "",
            settings_overrides=_parse_overrides(overrides or []),
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

    offline = bool(getattr(args, "offline", False))
    provider = str(getattr(args, "provider", "") or "").strip()
    model = str(getattr(args, "model", "") or "").strip()
    if bool(provider) != bool(model):
        print("[error] 真实评测必须同时提供 --provider 和 --model")
        return 2
    if args.compare and (provider or model):
        print(
            "[error] --compare 请使用 --baseline provider/model 和 "
            "--candidate provider/model"
        )
        return 2
    requested_suite = str(getattr(args, "suite", "") or "")
    if offline and requested_suite in ("tasks", "all"):
        print("[error] --offline 不能与 --suite tasks/all 同时使用")
        return 2
    if not requested_suite:
        if offline:
            suite = "core"
        elif provider and model:
            suite = "tasks"
        elif args.compare:
            suite = "tasks"
        else:
            print(
                "[error] 请选择 --offline，或同时提供 "
                "--provider <id> --model <id>"
            )
            return 2
    else:
        suite = requested_suite

    if provider and model and not args.compare and not args.candidate:
        args.candidate = f"{provider}/{model}"
    has_real = suite in ("tasks", "all")
    for flag, spec in (("--baseline", args.baseline), ("--candidate", args.candidate)):
        err = _validate_spec(flag, spec, real=has_real)
        if err:
            print(f"[error] {err}")
            return 2
    if args.compare and (not args.baseline or not args.candidate):
        print("[error] --compare 必须同时提供 --baseline 和 --candidate")
        return 2

    scenarios = []
    if suite in ("core", "all"):
        scenarios.extend(core_scenarios())
    if suite in ("tasks", "all"):
        scenarios.extend(task_to_scenario(t) for t in core_tasks())
        scenarios.extend(core_real_scenarios())

    from lsm_harness.config import Settings
    artifacts_dir = create_eval_artifact_dir(
        Settings().home,
        args.artifacts_dir or None,
    )
    mode = "real" if has_real else "offline"
    selected_model = (
        "scripted"
        if mode == "offline"
        else args.candidate or f"{provider}/{model}"
    )
    mode_detail = (
        "deterministic, no real API"
        if mode == "offline"
        else "real model API"
    )
    print(f"[eval] mode={mode} ({mode_detail})")
    print(f"[eval] model={selected_model}")
    print(f"[eval] artifacts={artifacts_dir}")
    if suite == "core" and (args.baseline or args.candidate):
        print(
            "[note] offline/core uses a scripted model; provider/model "
            "values are provenance only. Use a real task suite for model A/B."
        )

    judge_client = None
    if args.judge == "model":
        from lsm_harness.ai.providers import get_client
        judge_client = get_client()

    if args.compare:
        variants = [
            _variant("baseline", args.baseline, args.baseline_prompt,
                     args.baseline_deny_tools, args.baseline_override),
            _variant("candidate", args.candidate, args.candidate_prompt,
                     args.candidate_deny_tools, args.candidate_override),
        ]
        if (
            (args.baseline, args.baseline_prompt, args.baseline_deny_tools,
             args.baseline_override)
            == (args.candidate, args.candidate_prompt, args.candidate_deny_tools,
                args.candidate_override)
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
        write_comparison_run_files(
            artifacts_dir,
            reports,
            eval_set=f"{suite}-suite",
            mode=mode,
            variants=variants,
        )
        print(render_comparison_report(
            reports,
            eval_set=f"{suite}-suite",
            variant_labels={
                "baseline": args.baseline,
                "candidate": args.candidate,
            },
        ))
        total_passed = sum(v.passed for r in reports for v in r.variants)
        total_runs = sum(v.total for r in reports for v in r.variants)
        return 0 if total_passed == total_runs else 1

    # 单次模式:用 candidate 侧 CLI 配置(若有)作为唯一 variant。
    candidate = _variant("candidate", args.candidate, args.candidate_prompt,
                         args.candidate_deny_tools, args.candidate_override)
    variant = candidate if any(
        (args.candidate, args.candidate_prompt, args.candidate_deny_tools,
         args.candidate_override)
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

    write_single_run_files(
        artifacts_dir,
        results,
        mode=mode,
        variant=variant,
    )
    print(render_results(results, mode=mode))
    passed = sum(1 for result in results if result.passed)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    main()
