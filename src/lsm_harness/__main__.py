"""`lsm`, `lsm doctor`, and `lsm smoke` entrypoints."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="lsm", description="LSM 的个人 Agent Harness")
    parser.add_argument("command", nargs="?", choices=["doctor", "smoke"])
    args = parser.parse_args()
    if args.command == "doctor":
        from lsm_harness.doctor import run

        raise SystemExit(run())
    if args.command == "smoke":
        from lsm_harness.smoke import run

        raise SystemExit(run())
    from lsm_harness.gateway.cli import run_chat

    raise SystemExit(run_chat())


if __name__ == "__main__":
    main()
