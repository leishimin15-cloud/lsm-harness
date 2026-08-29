"""Non-billable local environment checks."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys

from rich.console import Console
from rich.table import Table

from lsm_harness.config import Settings


def checks(settings: Settings | None = None) -> list[tuple[str, bool, str]]:
    settings = settings or Settings()
    results: list[tuple[str, bool, str]] = []
    results.append(("Python", sys.version_info >= (3, 11), sys.version.split()[0]))
    results.append(("openai", bool(importlib.util.find_spec("openai")), "installed"))
    if settings.provider == "anthropic":
        results.append((
            "anthropic",
            bool(importlib.util.find_spec("anthropic")),
            "installed",
        ))
    results.append(("rich", bool(importlib.util.find_spec("rich")), "installed"))
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE probe USING fts5(text, tokenize='trigram')")
        conn.execute("INSERT INTO probe(text) VALUES('中文 Harness 检索')")
        matched = bool(conn.execute("SELECT 1 FROM probe WHERE probe MATCH 'Harness'").fetchone())
        conn.close()
        results.append(("SQLite FTS5 trigram", matched, sqlite3.sqlite_version))
    except sqlite3.Error as exc:
        results.append(("SQLite FTS5 trigram", False, str(exc)))
    provider = settings.provider or "auto-detected"
    results.append((
        "Model API Key",
        bool(settings.api_key),
        f"{provider}: configured" if settings.api_key else f"{provider}: missing",
    ))
    results.append((
        "Thinking",
        settings.thinking in {
            "disabled", "auto", "enabled",
            "off", "minimal", "low", "medium", "high", "xhigh",
        },
        settings.thinking,
    ))
    results.append((
        "Cache retention",
        settings.cache_retention in {"none", "short", "long"},
        settings.cache_retention,
    ))
    context_ok = (
        settings.context_budget_tokens >= 256
        and 0 < settings.context_compression_tokens <= settings.context_budget_tokens
        and settings.context_recent_turns > 0
        and settings.summary_max_tokens > 0
    )
    results.append(
        (
            "Context budget",
            context_ok,
            f"compress {settings.context_compression_tokens} / budget {settings.context_budget_tokens}",
        )
    )
    return results


def run() -> int:
    table = Table(title="LSM Harness doctor（不会调用模型）")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    results = checks()
    for name, ok, detail in results:
        table.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", detail)
    Console().print(table)
    return 0 if all(ok for _, ok, _ in results) else 1
