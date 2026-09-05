"""`lsm`, `lsm web`, diagnostics, and compatibility entrypoints."""

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
    sub.add_parser("tui", help="启动 Textual TUI 界面")
    eval_p = sub.add_parser("eval", help="运行 eval 测试套件")
    eval_p.add_argument("--suite", default="", help="指定套件 (tools/retrieval/safety)")
    eval_p.add_argument("--record", action="store_true", help="记录 golden traces")
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
        from lsm_harness.gateway.tui import LSMTui
        app = LSMTui()
        app.run()
        return
    if args.command == "traces":
        from lsm_harness.ops.tracing import list_recent_traces
        list_recent_traces()
        return
    if args.command == "eval":
        from lsm_harness.ops.eval import run_evals
        raise SystemExit(run_evals(suite_name=args.suite, record=args.record))
    from lsm_harness.coding_agent.cli import run_chat
    raise SystemExit(run_chat())


if __name__ == "__main__":
    main()
