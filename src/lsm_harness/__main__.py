"""`lsm`, `lsm doctor`, `lsm smoke`, `lsm serve`, `lsm traces` entrypoints."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="lsm", description="LSM 的个人 Agent Harness")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="环境自检")
    sub.add_parser("smoke", help="确定性冒烟测试")
    serve_p = sub.add_parser("serve", help="启动 HTTP API 服务")
    serve_p.add_argument("--port", type=int, default=8910, help="监听端口 (默认 8910)")
    serve_p.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    sub.add_parser("traces", help="查看最近的 trace 文件")
    eval_p = sub.add_parser("eval", help="运行 eval 测试套件")
    eval_p.add_argument("--suite", default="", help="指定套件 (tools/retrieval/safety)")
    eval_p.add_argument("--record", action="store_true", help="记录 golden traces")
    args = parser.parse_args()

    if args.command == "doctor":
        from lsm_harness.doctor import run
        raise SystemExit(run())
    if args.command == "smoke":
        from lsm_harness.smoke import run
        raise SystemExit(run())
    if args.command == "serve":
        from lsm_harness.gateway.http_api import run_server
        run_server(port=args.port, host=args.host)
        return
    if args.command == "traces":
        from lsm_harness.ops.tracing import list_recent_traces
        list_recent_traces()
        return
    if args.command == "eval":
        from lsm_harness.ops.eval import run_evals
        raise SystemExit(run_evals(suite_name=args.suite, record=args.record))
    from lsm_harness.gateway.cli import run_chat
    raise SystemExit(run_chat())


if __name__ == "__main__":
    main()
