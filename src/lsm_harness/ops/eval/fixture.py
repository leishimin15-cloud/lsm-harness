"""Isolated fixture workspaces for eval scenarios.

A fixture is either an inline ``files`` dict (relative path → content) or a
``source_dir`` on disk.  ``materialize`` copies it into a fresh temporary
workspace so every eval run starts from a clean copy — no pollution, no
mutating the checked-in fixture tree.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalFixture:
    """A starting filesystem state for one scenario.

    Exactly one of ``files`` or ``source_dir`` should be set.  ``files`` maps
    a workspace-relative path (``/``-separated) to its text content;
    ``source_dir`` is a directory whose whole tree is copied in.
    """

    files: dict[str, str] = field(default_factory=dict)
    source_dir: Path | None = None

    @classmethod
    def inline(cls, files: dict[str, str]) -> "EvalFixture":
        return cls(files=files)

    @classmethod
    def from_dir(cls, source_dir: str | Path) -> "EvalFixture":
        return cls(source_dir=Path(source_dir))


def materialize(fixture: EvalFixture, target_dir: Path) -> Path:
    """Copy a fixture into ``target_dir`` (created fresh), return it."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if fixture.source_dir is not None:
        _copy_tree(fixture.source_dir, target_dir)
    for rel_path, content in fixture.files.items():
        dest = target_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    return target_dir


def new_workspace(fixture: EvalFixture | None) -> Path:
    """Create a fresh temporary workspace and materialize ``fixture`` into it."""
    workspace = Path(tempfile.mkdtemp(prefix="lsm-eval-"))
    if fixture is not None:
        materialize(fixture, workspace)
    return workspace


def _copy_tree(source: Path, target: Path) -> None:
    source = source.resolve()
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        dest = target / rel
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


class FixtureRegistry:
    """Named fixtures loaded from a directory (``evals/fixtures/<name>/``).

    Each immediate subdirectory is one fixture keyed by its name.  Files
    inside are copied verbatim, so a fixture can be any project tree.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._fixtures: dict[str, EvalFixture] = {}

    def load(self) -> None:
        if not self.root.exists():
            return
        for child in sorted(self.root.iterdir()):
            if child.is_dir():
                self._fixtures[child.name] = EvalFixture.from_dir(child)

    def get(self, name: str) -> EvalFixture:
        if name not in self._fixtures:
            raise KeyError(f"unknown fixture '{name}' (have {sorted(self._fixtures)})")
        return self._fixtures[name]

    def names(self) -> list[str]:
        return sorted(self._fixtures)


def load_dir_fixtures(root: str | Path) -> FixtureRegistry:
    """Load the fixture registry rooted at ``root``."""
    registry = FixtureRegistry(Path(root))
    registry.load()
    return registry
