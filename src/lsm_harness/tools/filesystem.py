"""Workspace-bounded filesystem ToolDefinitions backed by FileOperations."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from lsm_harness.coding_agent.operations import (
    FileOperations,
    LocalFileOperations,
    TrackingFileOperations,
)
from lsm_harness.coding_agent.tools import ToolDefinition
from lsm_harness.tools.truncate import truncate_head, truncate_line


def _size_fmt(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _ops(
    home: Path | None,
    operations: FileOperations | None,
) -> FileOperations:
    return operations or LocalFileOperations(home or Path.cwd())


def _read_file(
    path: str,
    offset: int = 1,
    limit: int = 200,
    home: Path | None = None,
    operations: FileOperations | None = None,
) -> str:
    fs = _ops(home, operations)
    try:
        target = fs.resolve(path)
    except PermissionError as exc:
        return f"Error: {exc}"
    if not fs.exists(target):
        return f"Error: file not found: {path}"
    if not fs.is_file(target):
        return f"Error: not a regular file: {path}"
    size = fs.size(target)
    if size > 10 * 1024 * 1024:
        return f"Error: file too large ({_size_fmt(size)}, max 10 MB)."
    try:
        text = fs.read_text(target)
    except UnicodeDecodeError:
        try:
            text = fs.read_text(target, errors="replace")
        except Exception:
            return "Error: cannot decode file as text (binary file?)."

    lines = text.splitlines()
    total = len(lines)
    line_limit = max(1, min(int(limit), 1000))
    start = max(0, int(offset) - 1)
    if total > 0 and start >= total:
        return (
            f"Error: offset {int(offset)} is beyond end of file "
            f"({total} lines total)."
        )
    end = min(total, start + line_limit)
    # Byte cap via truncate_head: a handful of minified lines could
    # otherwise blow straight past the context budget.
    selected = truncate_head(
        "\n".join(lines[start:end]), max_lines=line_limit
    )
    shown = selected.content.split("\n") if selected.content else []
    snippet = "\n".join(
        f"{index + 1:>6}│{line}"
        for index, line in enumerate(shown, start=start)
    )
    header = (
        f"File: {path}  ({_size_fmt(size)} · {total} lines · "
        f"lines {start + 1}–{start + len(shown)})"
    )
    if selected.truncated:
        header += (
            f"\n[Truncated by {selected.truncated_by}: showing "
            f"{selected.output_lines} of {end - start} requested lines. "
            "Use offset/limit to read further.]"
        )
    return f"{header}\n{snippet}"


def _write_file(
    path: str,
    content: str,
    home: Path | None = None,
    operations: FileOperations | None = None,
) -> str:
    fs = _ops(home, operations)
    try:
        write = fs.write_text(path, content)
    except PermissionError as exc:
        return f"Error: {exc}"
    action = "Updated" if write.existed_before else "Created"
    return f"{action} {path} ({_size_fmt(len(content.encode('utf-8')))})."


def _write_file_safe(
    path: str,
    content: str,
    home: Path | None = None,
    file_state=None,
    operations: FileOperations | None = None,
) -> str:
    fs = _ops(home, operations)
    if file_state is not None and not isinstance(fs, TrackingFileOperations):
        fs = TrackingFileOperations(fs, file_state)
    try:
        write = fs.write_text(path, content)
    except PermissionError as exc:
        return f"Error: {exc}"
    action = "Updated" if write.existed_before else "Created"
    result = f"{action} {path} ({_size_fmt(len(content.encode('utf-8')))})."
    if write.change_summary:
        result += f"\n{write.change_summary}"
    return result


def _list_dir(
    path: str = ".",
    pattern: str = "*",
    home: Path | None = None,
    operations: FileOperations | None = None,
) -> str:
    fs = _ops(home, operations)
    try:
        target = fs.resolve(path)
    except PermissionError as exc:
        return f"Error: {exc}"
    if not fs.exists(target):
        return f"Error: directory not found: {path}"
    if not fs.is_dir(target):
        return f"Error: not a directory: {path}"
    try:
        entries = [
            entry for entry in fs.list_dir(target)
            if fnmatch.fnmatch(entry.name, pattern)
        ]
    except PermissionError:
        return f"Error: permission denied reading directory: {path}"
    if not entries:
        return f"Directory {path} is empty (or no entries match '{pattern}')."

    lines = [f"Directory: {target}  ({len(entries)} entries)"]
    max_name = max(len(entry.name) for entry in entries)
    for entry in entries[:200]:
        suffix = "/" if entry.kind == "dir" else "@" if entry.kind == "link" else ""
        kind_mark = {"dir": "[DIR]", "link": "[LNK]", "file": "     "}[entry.kind]
        size = _size_fmt(entry.size) if entry.kind == "file" else ""
        size_column = f"  {size:>8}" if size else " " * 10
        lines.append(
            f"  {kind_mark}  {entry.name + suffix:<{max_name + 3}}{size_column}"
        )
    if len(entries) > 200:
        lines.append(f"  ... and {len(entries) - 200} more entries")
    return "\n".join(lines)


def _grep(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    ignore_case: bool = True,
    max_results: int = 40,
    home: Path | None = None,
    operations: FileOperations | None = None,
) -> str:
    fs = _ops(home, operations)
    try:
        target = fs.resolve(path)
    except PermissionError as exc:
        return f"Error: {exc}"
    if not fs.exists(target):
        return f"Error: path not found: {path}"
    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return f"Error: invalid regex pattern: {exc}"

    files = fs.walk_files(target, recursive=recursive, max_size=2 * 1024 * 1024)
    if not files:
        return f"No text files found in {target}."
    results: list[str] = []
    result_limit = max(1, int(max_results))
    for file_path in files:
        if len(results) >= result_limit:
            break
        try:
            lines = fs.read_text(file_path, errors="replace").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(lines, 1):
            if regex.search(line):
                results.append(
                    f"{fs.relative(file_path)}:{line_number}: "
                    f"{truncate_line(line.strip())}"
                )
                if len(results) >= result_limit:
                    break
    if not results:
        return f"No matches for '{pattern}' in {target}."
    return f"grep: {len(results)} matches for '{pattern}' in {target}\n" + "\n".join(results)


def make_tools(
    home: Path,
    file_state=None,
    *,
    operations: FileOperations | None = None,
    readable_roots: list[Path] | None = None,
) -> list[ToolDefinition]:
    """Build filesystem definitions around an injected operations object."""
    fs: FileOperations = operations or LocalFileOperations(
        home, readable_roots=readable_roots or ()
    )
    if file_state is not None:
        fs = TrackingFileOperations(fs, file_state)
    return [
        ToolDefinition(
            name="read_file",
            label="读取文件",
            description=(
                "读取文件内容。支持指定起始行 (offset) 和读取行数 (limit)。"
                "单次读取上限 1000 行或 50KB（先到为准），超出会注明截断。"
                "对于大文件，使用 offset/limit 分段读取。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对或绝对）"},
                    "offset": {"type": "integer", "description": "起始行号，1-indexed，默认 1"},
                    "limit": {"type": "integer", "description": "最大行数，默认 200，上限 1000"},
                },
                "required": ["path"],
            },
            execute=lambda path, offset=1, limit=200: _read_file(
                path, offset, limit, operations=fs
            ),
            effect="read",
            execution_mode="parallel",
            prompt_snippet="读取大文件时使用 read_file 的 offset/limit 分段读取。",
        ),
        ToolDefinition(
            name="write_file",
            label="写入文件",
            description="创建或覆盖文件。会自动创建父目录。用于保存代码、笔记、配置等。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["path", "content"],
            },
            execute=lambda path, content: _write_file_safe(
                path, content, operations=fs
            ),
            effect="local_write",
            execution_mode="sequential",
            prompt_snippet="写文件使用 write_file；路径必须位于当前工作区。",
        ),
        ToolDefinition(
            name="list_dir",
            label="列出目录",
            description="列出目录内容，可按 glob pattern 过滤（如 '*.py'）。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，默认当前目录"},
                    "pattern": {"type": "string", "description": "Glob 过滤模式，默认 '*'"},
                },
            },
            execute=lambda path=".", pattern="*": _list_dir(
                path, pattern, operations=fs
            ),
            effect="read",
            execution_mode="parallel",
        ),
        ToolDefinition(
            name="grep",
            label="搜索文件",
            description="在文件中搜索正则表达式。支持递归搜索目录。",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python 正则表达式"},
                    "path": {"type": "string", "description": "文件或目录路径，默认当前目录"},
                    "recursive": {"type": "boolean", "description": "目录中递归搜索，默认 true"},
                    "ignore_case": {"type": "boolean", "description": "忽略大小写，默认 true"},
                    "max_results": {"type": "integer", "description": "最大结果数，默认 40"},
                },
                "required": ["pattern"],
            },
            execute=lambda pattern, path=".", recursive=True, ignore_case=True, max_results=40: _grep(
                pattern,
                path,
                recursive,
                ignore_case,
                max_results,
                operations=fs,
            ),
            effect="read",
            execution_mode="parallel",
        ),
    ]


__all__ = [
    "_grep",
    "_list_dir",
    "_read_file",
    "_write_file",
    "_write_file_safe",
    "make_tools",
]
