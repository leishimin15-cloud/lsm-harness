"""Project context files for the system prompt (pi chapter 8, mechanism ②).

Collects ``CLAUDE.md`` / ``AGENTS.md`` from three scopes and merges them
outermost-first, so the model reads general rules before specific ones:

1. ``agent_dir``  — global, user-level (analogous to ``~/.pi/CLAUDE.md``)
2. ancestors of ``cwd`` — from the filesystem root down to the parent
3. ``cwd`` itself — the current project, most specific, comes last

The merged result is wrapped in XML: the ``</project_instructions>`` tag is
an unambiguous boundary and the ``path`` attribute lets the model tell
org-level rules apart from project-level ones.
"""

from __future__ import annotations

from pathlib import Path

CONTEXT_FILE_NAMES = ("CLAUDE.md", "AGENTS.md", "claude.md", "agents.md")

_MAX_FILE_BYTES = 64 * 1024


def _find_in(directory: Path) -> Path | None:
    for name in CONTEXT_FILE_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _read(path: Path) -> str | None:
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return text or None


def load_project_context_files(
    cwd: Path,
    agent_dir: Path | None = None,
) -> list[tuple[Path, str]]:
    """Return ``(path, content)`` pairs ordered global → ancestors → cwd."""
    cwd = cwd.resolve()
    found: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def collect(directory: Path) -> None:
        marker = _find_in(directory)
        if marker is None or marker in seen:
            return
        content = _read(marker)
        if content is not None:
            seen.add(marker)
            found.append((marker, content))

    if agent_dir is not None:
        collect(agent_dir.resolve())
    # Ancestors root-first, then cwd itself last (most specific).
    for ancestor in reversed(list(cwd.parents)):
        collect(ancestor)
    collect(cwd)
    return found


def format_project_context(files: list[tuple[Path, str]]) -> str:
    """Wrap collected files in the ``<project_context>`` XML envelope."""
    if not files:
        return ""
    blocks = [
        "<project_context>",
        "",
        "Project-specific instructions and guidelines:",
        "",
    ]
    for path, content in files:
        blocks.append(f'<project_instructions path="{path}">')
        blocks.append(content)
        blocks.append("</project_instructions>")
        blocks.append("")
    blocks.append("</project_context>")
    return "\n".join(blocks)


__all__ = [
    "CONTEXT_FILE_NAMES",
    "format_project_context",
    "load_project_context_files",
]
