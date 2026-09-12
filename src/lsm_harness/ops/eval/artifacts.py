"""Workspace snapshots and run artifacts for eval.

Artifacts capture everything needed to investigate a run after the fact:
the final answer, tool-call records, a workspace diff, the session JSONL
(tree of truth), the raw trace, usage (token/cache/cost), and duration.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Directories that are tool/agent internals or build detritus, never part of
# the *user-visible* workspace state an eval should diff.
_SKIP_DIRS = frozenset({".git", ".lsm", "__pycache__", ".pytest_cache", "node_modules", ".mypy_cache"})

_MAX_SNAPSHOT_FILE = 1024 * 1024  # skip files larger than 1 MiB


def snapshot_workspace(root: Path) -> dict[str, str]:
    """Map workspace-relative path → text content for all text files.

    Skips internals (``.lsm``, VCS, caches) and undecodable / oversized files.
    """
    snapshot: dict[str, str] = {}
    if not root.exists():
        return snapshot
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        try:
            if path.stat().st_size > _MAX_SNAPSHOT_FILE:
                continue
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        snapshot[str(rel)] = text
    return snapshot


def diff_snapshots(before: dict[str, str], after: dict[str, str]) -> list[tuple[str, str]]:
    """Return ``(status, path)`` pairs: status ∈ {added, deleted, modified}."""
    changed: list[tuple[str, str]] = []
    for path in sorted(set(before) | set(after)):
        if path not in before:
            changed.append(("added", path))
        elif path not in after:
            changed.append(("deleted", path))
        elif before[path] != after[path]:
            changed.append(("modified", path))
    return changed


def unified_patch(before: dict[str, str], after: dict[str, str]) -> str:
    """One unified-diff patch across all changed files."""
    parts: list[str] = []
    for status, path in sorted(diff_snapshots(before, after), key=lambda p: p[1]):
        old = before.get(path, "")
        new = after.get(path, "")
        if old == new:
            continue
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        if status == "added":
            diff = list(difflib.unified_diff([], new_lines, fromfile="/dev/null", tofile=f"b/{path}"))
        elif status == "deleted":
            diff = list(difflib.unified_diff(old_lines, [], fromfile=f"a/{path}", tofile="/dev/null"))
        else:
            diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}"))
        if diff:
            parts.append("".join(diff))
    return "".join(parts)


@dataclass
class EvalRunArtifacts:
    """Serializable record of one scenario run."""

    name: str
    status: str = "ok"
    duration_ms: float = 0.0
    final_answer: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    changed_files: list[tuple[str, str]] = field(default_factory=list)
    workspace_diff: str = ""
    session_jsonl: str = ""
    trace_jsonl: str = ""
    usage: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    # ── run provenance(阶段 2 步 3) ───────────────────────
    model: str = ""
    provider: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    git_revision: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "final_answer": self.final_answer,
            "tool_calls": self.tool_calls,
            "changed_files": [{"status": s, "path": p} for s, p in self.changed_files],
            "failures": self.failures,
            "usage": self.usage,
            "model": self.model,
            "provider": self.provider,
            "config": self.config,
            "git_revision": self.git_revision,
        }

    def write(self, output_dir: Path) -> Path:
        """Write the artifact bundle into ``output_dir``; return it."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "reply.txt").write_text(self.final_answer, encoding="utf-8")
        (output_dir / "tool_calls.json").write_text(
            json.dumps(self.tool_calls, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "diff.patch").write_text(self.workspace_diff, encoding="utf-8")
        if self.session_jsonl:
            (output_dir / "session.jsonl").write_text(self.session_jsonl, encoding="utf-8")
        if self.trace_jsonl:
            (output_dir / "trace.jsonl").write_text(self.trace_jsonl, encoding="utf-8")
        if self.usage:
            (output_dir / "usage.json").write_text(
                json.dumps(self.usage, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return output_dir
