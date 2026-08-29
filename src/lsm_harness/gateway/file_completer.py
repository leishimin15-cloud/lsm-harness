"""@ 文件引用 — fuzzy file search triggered by @ in the input.

Uses prompt_toolkit's ``WordCompleter`` with a dynamic word list that
refreshes when the user types ``@``.  Because prompt_toolkit completers
are static, we rebuild the list on every keypress in a background thread.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.document import Document


class AtFileCompleter(Completer):
    """Fuzzy file search triggered by ``@`` prefix.

    Usage::

        session = PromptSession(completer=AtFileCompleter(root="."))
    """

    def __init__(self, root: str | Path = ".", max_files: int = 2000):
        self.root = Path(root).resolve()
        self.max_files = max_files
        self._file_list: list[str] = []
        self._last_refresh = 0.0
        self._refresh_interval = 5.0  # seconds
        self._lock = threading.Lock()

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ):
        text = document.text_before_cursor

        # Find the @ position
        at_idx = text.rfind("@")
        if at_idx == -1:
            return

        # Extract the query after @
        query = text[at_idx + 1:]

        # Refresh file list periodically
        self._maybe_refresh()

        # Search
        with self._lock:
            files = list(self._file_list)

        matches = self._fuzzy_match(query, files)

        # Build completions
        prefix = text[:at_idx + 1]  # includes @
        for match in matches[:20]:
            yield Completion(
                match,
                start_position=-len(query),
                display=match,
            )

    def _maybe_refresh(self) -> None:
        """Refresh the file list if stale."""
        now = time.monotonic()
        if now - self._last_refresh < self._refresh_interval and self._file_list:
            return
        self._last_refresh = now

        def _refresh():
            files = self._collect_files(self.root)
            with self._lock:
                self._file_list = files

        t = threading.Thread(target=_refresh, daemon=True)
        t.start()

    def _collect_files(self, root: Path) -> list[str]:
        """Collect all non-hidden files under root."""
        files: list[str] = []
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                # Skip hidden dirs
                dirnames[:] = [
                    d for d in dirnames
                    if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".git")
                ]
                for fn in filenames:
                    if fn.startswith("."):
                        continue
                    fp = Path(dirpath) / fn
                    try:
                        rel = fp.relative_to(root)
                        files.append(str(rel))
                    except ValueError:
                        continue
                    if len(files) >= self.max_files:
                        return files
        except OSError:
            pass
        return files

    @staticmethod
    def _fuzzy_match(query: str, candidates: list[str]) -> list[str]:
        """Score candidates by how well they match the query.

        Prefers:
          - Exact substring match
          - Starts-with match
          - Character sequence match (fuzzy)
        """
        if not query:
            return sorted(candidates)[:20]

        qlower = query.lower()
        scored: list[tuple[int, str]] = []

        for cand in candidates:
            clower = cand.lower()
            score = 0

            # Exact substring match (highest priority)
            if qlower in clower:
                score += 1000
                # Prefer shorter paths
                score -= len(cand)
                # Prefer matches closer to start
                idx = clower.index(qlower)
                score -= idx
            else:
                # Fuzzy: check if all chars appear in order
                pos = 0
                all_found = True
                for ch in qlower:
                    pos = clower.find(ch, pos)
                    if pos == -1:
                        all_found = False
                        break
                    pos += 1
                if all_found:
                    score += 500 - pos

            if score > 0:
                scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]


# ── path completer (Tab) ──────────────────────────────────────────


class PathCompleter(Completer):
    """Tab-complete filesystem paths."""

    def __init__(self, root: str | Path = "."):
        self.root = Path(root).resolve()

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ):
        text = document.text_before_cursor

        # Find the last path-like token
        import re
        tokens = re.split(r"[\s]+", text)
        if not tokens:
            return
        last = tokens[-1]

        # Only complete if it looks like a path
        if not last or (not last.startswith("./") and not last.startswith("../")
                        and not last.startswith("/") and not last.startswith("~/")):
            return

        expanded = os.path.expanduser(last)
        parent = os.path.dirname(expanded) or "."
        prefix = os.path.basename(expanded)

        try:
            full_parent = (self.root / parent).resolve()
            if not full_parent.is_dir():
                return
        except OSError:
            return

        try:
            entries = sorted(full_parent.iterdir())
        except OSError:
            return

        for entry in entries:
            name = entry.name
            if name.startswith("."):
                continue
            if not name.startswith(prefix):
                continue

            display = name + ("/" if entry.is_dir() else "")
            # Calculate how much of the path to replace
            yield Completion(
                display,
                start_position=-len(prefix),
                display=display,
            )
