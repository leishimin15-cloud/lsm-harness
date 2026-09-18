"""Skill discovery and Pi-style lazy loading (metadata-only listing).

Pi parity additions over the minimal version:

- 多目录发现:项目级优先(同名冲突 first-wins),用户级补充;
- 诊断:格式错误的 SKILL.md 不再静默跳过,记录 SkillDiagnostic;
- listing 仍只注入元数据,正文由模型 read_file 按需加载。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path
    source: str = "project"


@dataclass(frozen=True)
class SkillDiagnostic:
    """Pi resource diagnostics: 加载问题可见,不中断启动。"""

    path: Path
    code: str  # "parse_failed" | "invalid_metadata" | "collision"
    message: str


def parse_skill(text: str, path: Path) -> Skill | None:
    skill, _diagnostic = _parse(text, path)
    return skill


def _parse(
    text: str, path: Path, source: str = "project"
) -> tuple[Skill | None, SkillDiagnostic | None]:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return None, SkillDiagnostic(
            path, "parse_failed", "missing or malformed --- frontmatter ---"
        )
    frontmatter, body = match.groups()
    fields = {
        key.strip(): value.strip().strip("'\"")
        for key, separator, value in (
            line.partition(":") for line in frontmatter.splitlines()
        )
        if separator
    }
    if not fields.get("name"):
        return None, SkillDiagnostic(
            path, "invalid_metadata", "frontmatter is missing 'name'"
        )
    if not fields.get("description"):
        return None, SkillDiagnostic(
            path, "invalid_metadata", "frontmatter is missing 'description'"
        )
    if not body.strip():
        return None, SkillDiagnostic(path, "invalid_metadata", "body is empty")
    return (
        Skill(fields["name"], fields["description"], body.strip(), path, source),
        None,
    )


def default_skill_directories(home: Path) -> list[tuple[Path, str]]:
    """Pi loadSkills 的默认目录:项目级(home/skills)在前 —— 同名冲突
    first-wins,项目赢;用户级(~/.lsm/skills)补充全局 skill。
    两目录解析后相同(如 cwd 恰好是 home)时去重。"""
    candidates = [
        (home / "skills", "project"),
        (Path.home() / ".lsm" / "skills", "user"),
    ]
    seen: set[Path] = set()
    directories: list[tuple[Path, str]] = []
    for path, scope in candidates:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            directories.append((resolved, scope))
    return directories


class SkillLoader:
    """directories 每项是 Path 或 (Path, scope);纯 Path 默认 "project"。"""

    def __init__(self, directories: list):
        self.directories: list[tuple[Path, str]] = [
            entry
            if isinstance(entry, tuple)
            else (Path(entry), "project")
            for entry in directories
        ]
        self.skills: list[Skill] = []
        self.diagnostics: list[SkillDiagnostic] = []
        self._signature: tuple = ()
        self.refresh()

    def _scan(self) -> tuple:
        return tuple(
            (str(path), path.stat().st_mtime_ns)
            for directory, _scope in self.directories
            if directory.is_dir()
            for path in sorted(directory.rglob("SKILL.md"))
        )

    def refresh(self) -> None:
        self.skills = []
        self.diagnostics = []
        winners: dict[str, Path] = {}
        for directory, scope in self.directories:
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("SKILL.md")):
                skill, diagnostic = _parse(
                    path.read_text(encoding="utf-8"), path, source=scope
                )
                if diagnostic is not None:
                    self.diagnostics.append(diagnostic)
                if skill is None:
                    continue
                winner = winners.get(skill.name)
                if winner is not None:
                    self.diagnostics.append(SkillDiagnostic(
                        path,
                        "collision",
                        f"skill '{skill.name}' already loaded from {winner}",
                    ))
                    continue
                winners[skill.name] = path
                self.skills.append(skill)
        self._signature = self._scan()

    def listing(self, workspace: Path) -> str:
        """Pi-mode lazy loading: metadata ONLY, never the skill body.

        The model sees what skills exist and where they live; the contract
        line tells it to ``read_file`` the SKILL.md before acting.  Paths
        are rendered relative to ``workspace`` (read_file's root) when
        possible so the model gets the exact spelling read_file expects.
        """
        if self._scan() != self._signature:
            self.refresh()
        if not self.skills:
            return ""
        root = workspace.resolve()
        entries = []
        for skill in self.skills:
            try:
                location = str(skill.path.resolve().relative_to(root))
            except ValueError:
                location = str(skill.path)
            entries.append(
                "  <skill>\n"
                f"    <name>{skill.name}</name>\n"
                f"    <description>{skill.description}</description>\n"
                f"    <location>{location}</location>\n"
                "  </skill>"
            )
        return (
            "When a skill matches the task, use read_file to load its "
            "SKILL.md before acting. Relative paths referenced inside a "
            "skill file resolve against that skill's directory.\n\n"
            "<available_skills>\n" + "\n".join(entries) + "\n</available_skills>"
        )
