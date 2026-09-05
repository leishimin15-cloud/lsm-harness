"""Product operations for filesystem and shell tool definitions.

Tools own argument policy and result formatting. Operations own the concrete
host/sandbox I/O mechanism and can be replaced with in-memory test doubles.
"""

from __future__ import annotations

import os
import posixpath
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


@dataclass(frozen=True)
class FileEntry:
    path: str
    name: str
    kind: str
    size: int = 0


@dataclass(frozen=True)
class FileWrite:
    path: str
    existed_before: bool
    change_summary: str = ""


class FileOperations(Protocol):
    root: str

    def resolve(self, path: str) -> str: ...
    def exists(self, path: str) -> bool: ...
    def is_file(self, path: str) -> bool: ...
    def is_dir(self, path: str) -> bool: ...
    def size(self, path: str) -> int: ...
    def read_text(self, path: str, *, errors: str = "strict") -> str: ...
    def write_text(self, path: str, content: str) -> FileWrite: ...
    def list_dir(self, path: str) -> list[FileEntry]: ...
    def walk_files(self, path: str, *, recursive: bool, max_size: int) -> list[str]: ...
    def relative(self, path: str) -> str: ...


class LocalFileOperations:
    """Workspace-bounded local filesystem implementation."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self.root = str(self._root)

    def resolve(self, path: str) -> str:
        target = Path(os.path.expanduser(path))
        target = (
            target.resolve()
            if target.is_absolute()
            else (self._root / target).resolve()
        )
        try:
            target.relative_to(self._root)
        except ValueError as exc:
            raise PermissionError(
                f"Path '{path}' is outside the allowed workspace ({self._root})."
            ) from exc
        return str(target)

    def exists(self, path: str) -> bool:
        return Path(self.resolve(path)).exists()

    def is_file(self, path: str) -> bool:
        return Path(self.resolve(path)).is_file()

    def is_dir(self, path: str) -> bool:
        return Path(self.resolve(path)).is_dir()

    def size(self, path: str) -> int:
        return Path(self.resolve(path)).stat().st_size

    def read_text(self, path: str, *, errors: str = "strict") -> str:
        return Path(self.resolve(path)).read_text(encoding="utf-8", errors=errors)

    def write_text(self, path: str, content: str) -> FileWrite:
        target = Path(self.resolve(path))
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return FileWrite(path=str(target), existed_before=existed)

    def list_dir(self, path: str) -> list[FileEntry]:
        target = Path(self.resolve(path))
        result: list[FileEntry] = []
        for entry in sorted(target.iterdir()):
            if entry.is_dir():
                kind = "dir"
            elif entry.is_symlink():
                kind = "link"
            else:
                kind = "file"
            result.append(FileEntry(
                path=str(entry),
                name=entry.name,
                kind=kind,
                size=entry.stat().st_size if kind == "file" else 0,
            ))
        return result

    def walk_files(self, path: str, *, recursive: bool, max_size: int) -> list[str]:
        target = Path(self.resolve(path))
        if target.is_file():
            return [str(target)] if target.stat().st_size <= max_size else []
        if not target.is_dir():
            return []
        if not recursive:
            return [
                str(item)
                for item in target.iterdir()
                if item.is_file() and item.stat().st_size <= max_size
            ]
        files: list[str] = []
        for directory, dirnames, filenames in os.walk(target):
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            for filename in filenames:
                candidate = Path(directory) / filename
                if candidate.stat().st_size <= max_size:
                    files.append(str(candidate))
        return files

    def relative(self, path: str) -> str:
        target = Path(self.resolve(path))
        return str(target.relative_to(self._root))


class TrackingFileOperations:
    """Writer decorator that adds FileState without coupling tools to it."""

    def __init__(self, inner: FileOperations, file_state: Any) -> None:
        self.inner = inner
        self.file_state = file_state
        self.root = inner.root

    def resolve(self, path: str) -> str:
        return self.inner.resolve(path)

    def exists(self, path: str) -> bool:
        return self.inner.exists(path)

    def is_file(self, path: str) -> bool:
        return self.inner.is_file(path)

    def is_dir(self, path: str) -> bool:
        return self.inner.is_dir(path)

    def size(self, path: str) -> int:
        return self.inner.size(path)

    def read_text(self, path: str, *, errors: str = "strict") -> str:
        return self.inner.read_text(path, errors=errors)

    def write_text(self, path: str, content: str) -> FileWrite:
        resolved = self.inner.resolve(path)
        self.file_state.snapshot(resolved)
        written = self.inner.write_text(path, content)
        change = self.file_state.record(written.path, content)
        return FileWrite(
            path=written.path,
            existed_before=written.existed_before,
            change_summary=change.summary if change is not None else "",
        )

    def list_dir(self, path: str) -> list[FileEntry]:
        return self.inner.list_dir(path)

    def walk_files(self, path: str, *, recursive: bool, max_size: int) -> list[str]:
        return self.inner.walk_files(path, recursive=recursive, max_size=max_size)

    def relative(self, path: str) -> str:
        return self.inner.relative(path)


class MockFileOperations:
    """In-memory filesystem used by unit tests; never touches the host."""

    def __init__(self, files: dict[str, str] | None = None, root: str = "/workspace") -> None:
        self.root = str(PurePosixPath(root))
        self.files: dict[str, str] = {}
        for path, content in (files or {}).items():
            self.files[self.resolve(path)] = content

    def resolve(self, path: str) -> str:
        raw = str(PurePosixPath(path))
        target = raw if raw.startswith("/") else posixpath.join(self.root, raw)
        normalized = posixpath.normpath(target)
        if normalized != self.root and not normalized.startswith(self.root + "/"):
            raise PermissionError(
                f"Path '{path}' is outside the allowed workspace ({self.root})."
            )
        return normalized

    def exists(self, path: str) -> bool:
        target = self.resolve(path)
        return target in self.files or self.is_dir(target)

    def is_file(self, path: str) -> bool:
        return self.resolve(path) in self.files

    def is_dir(self, path: str) -> bool:
        target = self.resolve(path).rstrip("/")
        return target == self.root or any(
            item.startswith(target + "/") for item in self.files
        )

    def size(self, path: str) -> int:
        return len(self.files[self.resolve(path)].encode("utf-8"))

    def read_text(self, path: str, *, errors: str = "strict") -> str:
        del errors
        return self.files[self.resolve(path)]

    def write_text(self, path: str, content: str) -> FileWrite:
        target = self.resolve(path)
        existed = target in self.files
        self.files[target] = content
        return FileWrite(path=target, existed_before=existed)

    def list_dir(self, path: str) -> list[FileEntry]:
        target = self.resolve(path).rstrip("/")
        entries: dict[str, FileEntry] = {}
        for file_path, content in self.files.items():
            if not file_path.startswith(target + "/"):
                continue
            suffix = file_path[len(target) + 1:]
            first = suffix.split("/", 1)[0]
            child = target + "/" + first
            if "/" in suffix:
                entries[first] = FileEntry(child, first, "dir")
            else:
                entries[first] = FileEntry(
                    child,
                    first,
                    "file",
                    len(content.encode("utf-8")),
                )
        return [entries[name] for name in sorted(entries)]

    def walk_files(self, path: str, *, recursive: bool, max_size: int) -> list[str]:
        target = self.resolve(path)
        if target in self.files:
            return [target] if self.size(target) <= max_size else []
        prefix = target.rstrip("/") + "/"
        return [
            file_path
            for file_path in sorted(self.files)
            if file_path.startswith(prefix)
            and (recursive or "/" not in file_path[len(prefix):])
            and self.size(file_path) <= max_size
        ]

    def relative(self, path: str) -> str:
        return self.resolve(path).removeprefix(self.root + "/")


@dataclass(frozen=True)
class ShellResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class ShellOperations(Protocol):
    def run(self, command: list[str], *, cwd: str, timeout: int) -> ShellResult: ...


def _sanitized_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in list(environment):
        if any(marker in key.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            environment[key] = "[REDACTED]"
    environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    environment["LSM_SHELL"] = "1"
    return environment


class LocalShellOperations:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def run(self, command: list[str], *, cwd: str, timeout: int) -> ShellResult:
        working_dir = Path(cwd).expanduser() if cwd else self.root
        if not working_dir.is_absolute():
            working_dir = self.root / working_dir
        working_dir = working_dir.resolve()
        try:
            working_dir.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(
                f"cwd '{cwd}' is outside the allowed workspace ({self.root})."
            ) from exc
        if not working_dir.is_dir():
            raise FileNotFoundError(f"working directory does not exist: {working_dir}")
        completed = subprocess.run(
            command,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_sanitized_environment(),
        )
        return ShellResult(completed.returncode, completed.stdout or "", completed.stderr or "")


class SandboxShellOperations:
    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox

    def run(self, command: list[str], *, cwd: str, timeout: int) -> ShellResult:
        session_id = self.sandbox.current_session
        if not session_id:
            raise RuntimeError("Docker sandbox has no active session")
        exit_code, stdout, stderr = self.sandbox.exec(
            session_id,
            command,
            timeout=timeout,
            cwd=cwd or "/workspace",
        )
        return ShellResult(exit_code, stdout, stderr)


class UnavailableShellOperations:
    def run(self, command: list[str], *, cwd: str, timeout: int) -> ShellResult:
        del command, cwd, timeout
        raise RuntimeError(
            "Docker sandbox was requested but is unavailable. "
            "Host shell execution is disabled to preserve isolation."
        )


class MockShellOperations:
    """Recording shell double with a preconfigured result."""

    def __init__(self, result: ShellResult | None = None) -> None:
        self.result = result or ShellResult(0, "ok", "")
        self.calls: list[tuple[list[str], str, int]] = []

    def run(self, command: list[str], *, cwd: str, timeout: int) -> ShellResult:
        self.calls.append((list(command), cwd, timeout))
        return self.result


__all__ = [
    "FileEntry",
    "FileOperations",
    "FileWrite",
    "LocalFileOperations",
    "LocalShellOperations",
    "MockFileOperations",
    "MockShellOperations",
    "SandboxShellOperations",
    "ShellOperations",
    "ShellResult",
    "TrackingFileOperations",
    "UnavailableShellOperations",
]
