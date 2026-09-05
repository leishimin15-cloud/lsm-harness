"""Print mode: ``lsm -p "prompt"`` — one-shot answer on stdout.

Pi/tau-style non-interactive entry point: the prompt goes in as an
argument, the assistant's text streams to stdout, tool activity goes to
stderr, and the process exit code mirrors the run's status (0 = completed,
1 = anything else).  No REPL, no prompt toolkit — safe to pipe and script.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lsm_harness.coding_agent.app import Harness


def run_print(prompt: str, *, app: "Harness | None" = None) -> int:
    """Run one prompt non-interactively; return the process exit code."""
    owns_app = app is None
    if owns_app:
        from lsm_harness.coding_agent.app import Harness
        from lsm_harness.config import Settings

        app = Harness(settings=Settings())

    def observe(event) -> None:
        kind, data = event.type, event.data
        if kind == "llm.text.delta":
            sys.stdout.write(data.get("text", ""))
            sys.stdout.flush()
        elif kind == "llm.text.end":
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif kind == "tool.requested":
            label = data.get("label") or data.get("tool", "?")
            print(f"⚙ {label}", file=sys.stderr)
        elif kind == "tool.completed":
            label = data.get("label") or data.get("tool", "?")
            icon = "✓" if data.get("status", "ok") == "ok" else "✗"
            print(f"{icon} {label}", file=sys.stderr)

    try:
        result = app.respond(prompt, observer=observe, source="print")
    finally:
        if owns_app:
            app.close()

    if result.status == "completed":
        return 0
    if result.error:
        print(result.error, file=sys.stderr)
    return 1
