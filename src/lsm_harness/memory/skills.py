"""File-backed procedural memory with transparent Chinese/English matching."""

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


def parse_skill(text: str, path: Path) -> Skill | None:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return None
    frontmatter, body = match.groups()
    fields = {
        key.strip(): value.strip().strip("'\"")
        for key, separator, value in (
            line.partition(":") for line in frontmatter.splitlines()
        )
        if separator
    }
    if not fields.get("name") or not fields.get("description") or not body.strip():
        return None
    return Skill(fields["name"], fields["description"], body.strip(), path)


def _tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9_-]{2,}", text.lower()))
    for chunk in re.findall(r"[\u3400-\u9fff]+", text):
        if len(chunk) == 1:
            tokens.add(chunk)
        else:
            tokens.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return tokens


class SkillLoader:
    def __init__(self, directories: list[Path]):
        self.directories = directories
        self.skills: list[Skill] = []
        self._signature: tuple = ()
        self.refresh()

    def _scan(self) -> tuple:
        return tuple(
            (str(path), path.stat().st_mtime_ns)
            for directory in self.directories
            if directory.is_dir()
            for path in sorted(directory.rglob("SKILL.md"))
        )

    def refresh(self) -> None:
        self.skills = []
        for directory in self.directories:
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("SKILL.md")):
                skill = parse_skill(path.read_text(encoding="utf-8"), path)
                if skill:
                    self.skills.append(skill)
        self._signature = self._scan()

    def match(self, message: str, max_skills: int = 2) -> list[Skill]:
        if self._scan() != self._signature:
            self.refresh()
        message_tokens = _tokens(message)
        scored = []
        for skill in self.skills:
            overlap = len(message_tokens & _tokens(f"{skill.name} {skill.description}"))
            if overlap:
                scored.append((overlap, skill))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [skill for _, skill in scored[:max_skills]]

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
            "SKILL.md before acting.\n\n"
            "<available_skills>\n" + "\n".join(entries) + "\n</available_skills>"
        )

