"""File state tracking: diff recording, undo, and change summaries.

Every ``write_file`` call records a snapshot of the previous content,
enabling:
  - Per-turn change summaries ("修改了 3 个文件, +45 -12")
  - Undo for individual files or all files in a turn
  - Diff generation for display
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FileChange:
    """One recorded file modification."""

    path: str
    old_content: str
    new_content: str
    existed_before: bool = True
    added_lines: int = 0
    removed_lines: int = 0

    def __post_init__(self):
        old_lines = self.old_content.splitlines()
        new_lines = self.new_content.splitlines()
        if self.old_content:  # not a new file
            sm = difflib.SequenceMatcher(None, old_lines, new_lines)
            ops = sm.get_opcodes()
            self.added_lines = sum(
                len(new_lines[j1:j2])
                for tag, i1, i2, j1, j2 in ops if tag in ("insert", "replace")
            )
            self.removed_lines = sum(
                len(old_lines[i1:i2])
                for tag, i1, i2, j1, j2 in ops if tag in ("delete", "replace")
            )
        else:
            self.added_lines = len(new_lines)
            self.removed_lines = 0

    @property
    def is_new_file(self) -> bool:
        return not self.existed_before

    @property
    def diff(self) -> str:
        """Generate a unified diff."""
        old_lines = self.old_content.splitlines(keepends=True)
        new_lines = self.new_content.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                old_lines if old_lines else [],
                new_lines,
                fromfile=f"a/{self.path}" if not self.is_new_file else "/dev/null",
                tofile=f"b/{self.path}",
            )
        )
        return "".join(diff_lines)

    @property
    def summary(self) -> str:
        """One-line change summary."""
        if self.is_new_file:
            return f"  + {self.path} (new, {self.added_lines} lines)"
        return (
            f"  ~ {self.path} (+{self.added_lines} -{self.removed_lines})"
        )


@dataclass
class TurnChanges:
    """All file changes in one turn."""

    changes: list[FileChange] = field(default_factory=list)

    @property
    def total_added(self) -> int:
        return sum(c.added_lines for c in self.changes)

    @property
    def total_removed(self) -> int:
        return sum(c.removed_lines for c in self.changes)

    @property
    def modified_files(self) -> list[str]:
        return [c.path for c in self.changes]

    @property
    def summary(self) -> str:
        """Multi-line summary of all changes."""
        if not self.changes:
            return "（无文件变更）"

        lines = [
            f"修改了 {len(self.changes)} 个文件，"
            f"+{self.total_added} 行，-{self.total_removed} 行：",
        ]
        for c in self.changes:
            lines.append(c.summary)
        return "\n".join(lines)


class FileState:
    """Tracks file modifications across turns.

    Usage::

        state = FileState()
        state.snapshot("src/app.py")          # before editing
        # ... write_file modifies src/app.py ...
        state.record("src/app.py", new_content)
        state.summary()                       # → "修改了 2 个文件, +15 -3"
        state.undo("src/app.py")              # → restores old content
        state.undo_all()                      # → restores everything
    """

    def __init__(self, home: Path | None = None):
        self._snapshots: dict[str, tuple[bool, str]] = {}
        self._current_turn: TurnChanges = TurnChanges()
        self._history: list[TurnChanges] = []
        self._home = home

    def _normalise(self, path: str) -> str:
        filepath = Path(path).expanduser()
        if not filepath.is_absolute() and self._home:
            filepath = self._home / filepath
        return str(filepath.resolve(strict=False))

    # ── snapshot / record ──────────────────────────────────────

    def snapshot(self, path: str) -> None:
        """Take a snapshot of a file before editing.

        Call this BEFORE write_file.  If the file doesn't exist,
        stores an empty string (indicating a new file).
        """
        filepath = Path(self._normalise(path))

        if filepath.exists():
            try:
                self._snapshots[str(filepath)] = (
                    True,
                    filepath.read_text(encoding="utf-8"),
                )
            except Exception:
                self._snapshots[str(filepath)] = (True, "")
        else:
            self._snapshots[str(filepath)] = (False, "")

    def record(self, path: str, new_content: str) -> FileChange | None:
        """Record a file modification after writing.

        Call this AFTER write_file.  Returns the FileChange, or None
        if no snapshot was taken.
        """
        path = self._normalise(path)
        snapshot = self._snapshots.pop(path, None)
        if snapshot is None:
            return None  # no snapshot — wasn't tracked
        existed_before, old = snapshot

        change = FileChange(
            path=path,
            old_content=old,
            new_content=new_content,
            existed_before=existed_before,
        )
        self._current_turn.changes.append(change)
        return change

    # ── undo ───────────────────────────────────────────────────

    def undo(self, path: str) -> bool:
        """Restore a file to its pre-edit state.

        Only works for files modified in the current turn.
        Returns True if the file was restored.
        """
        normalised = self._normalise(path)
        for change in self._current_turn.changes:
            if change.path == normalised:
                target = Path(change.path)
                if change.is_new_file:
                    target.unlink(missing_ok=True)
                else:
                    target.write_text(change.old_content, encoding="utf-8")
                self._current_turn.changes.remove(change)
                return True
        return False

    def undo_all(self) -> int:
        """Restore all files modified in the current turn.

        Returns the number of files restored.
        """
        count = 0
        for change in list(self._current_turn.changes):
            try:
                target = Path(change.path)
                if change.is_new_file:
                    target.unlink(missing_ok=True)
                else:
                    target.write_text(change.old_content, encoding="utf-8")
                count += 1
            except Exception:
                pass
        self._current_turn.changes.clear()
        return count

    # ── query ──────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a summary of this turn's file changes."""
        return self._current_turn.summary

    def diff(self, path: str) -> str | None:
        """Return the diff for a specific file, or None."""
        normalised = self._normalise(path)
        for change in self._current_turn.changes:
            if change.path == normalised:
                return change.diff
        return None

    def all_diffs(self) -> str:
        """Return all diffs for this turn, concatenated."""
        parts = []
        for change in self._current_turn.changes:
            parts.append(f"--- {change.path}\n{change.diff}")
        return "\n".join(parts)

    @property
    def modified_files(self) -> list[str]:
        return self._current_turn.modified_files

    # ── turn lifecycle ─────────────────────────────────────────

    def end_turn(self) -> None:
        """Archive this turn's changes and start a fresh turn."""
        if self._current_turn.changes:
            self._history.append(self._current_turn)
        self._current_turn = TurnChanges()
        self._snapshots.clear()

    @property
    def last_turn_summary(self) -> str:
        """Summary of the most recently completed turn."""
        if self._history:
            return self._history[-1].summary
        return "（无文件变更）"
