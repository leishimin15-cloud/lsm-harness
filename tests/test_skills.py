"""Skill loader tests: Pi-style lazy listing (metadata only)."""

from pathlib import Path

import pytest

from lsm_harness.coding_agent.operations import LocalFileOperations
from lsm_harness.coding_agent.skills import (
    SkillLoader,
    default_skill_directories,
    parse_skill,
)
from lsm_harness.tools.filesystem import _read_file


def _write_skill(root: Path, name: str, description: str, body: str = "正文。") -> Path:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_parse_skill_requires_frontmatter_and_body(tmp_path):
    assert parse_skill("no frontmatter", tmp_path / "SKILL.md") is None
    skill = parse_skill(
        "---\nname: a\ndescription: b\n---\n\nbody\n", tmp_path / "SKILL.md"
    )
    assert skill is not None and skill.name == "a" and skill.body == "body"


def test_listing_is_metadata_only_and_workspace_relative(tmp_path):
    root = tmp_path / "skills"
    path = _write_skill(root, "weekly-review", "每周复盘项目进度")
    loader = SkillLoader([root])

    listing = loader.listing(tmp_path)
    assert "<available_skills>" in listing
    assert "weekly-review" in listing
    assert "每周复盘项目进度" in listing
    assert "read_file" in listing  # the load contract line
    assert "正文" not in listing  # body is NEVER inlined in lazy mode
    assert str(path.relative_to(tmp_path)) in listing


def test_listing_empty_when_no_skills(tmp_path):
    assert SkillLoader([tmp_path / "skills"]).listing(tmp_path) == ""


def test_refresh_picks_up_new_skill_on_next_listing(tmp_path):
    root = tmp_path / "skills"
    loader = SkillLoader([root])
    assert loader.listing(tmp_path) == ""
    _write_skill(root, "late-skill", "后加的")
    assert "late-skill" in loader.listing(tmp_path)


# ── Pi parity: diagnostics instead of silent skips ────────────────


def test_malformed_skills_are_diagnosed_not_silent(tmp_path):
    root = tmp_path / "skills"
    (root / "broken").mkdir(parents=True)
    (root / "broken" / "SKILL.md").write_text("no frontmatter at all\n")
    (root / "no-desc").mkdir()
    (root / "no-desc" / "SKILL.md").write_text(
        "---\nname: no-desc\n---\nbody\n"
    )
    (root / "empty-body").mkdir()
    (root / "empty-body" / "SKILL.md").write_text(
        "---\nname: empty-body\ndescription: d\n---\n\n   \n"
    )
    _write_skill(root, "good", "正常的")

    loader = SkillLoader([root])
    assert [s.name for s in loader.skills] == ["good"]
    codes = [d.code for d in loader.diagnostics]
    assert codes.count("parse_failed") == 1
    assert codes.count("invalid_metadata") == 2


# ── Pi parity: multi-directory discovery + collision precedence ───


def test_project_skill_wins_collision_over_user(tmp_path):
    project = tmp_path / "project" / "skills"
    user = tmp_path / "user" / "skills"
    _write_skill(project, "code-review", "项目版", body="项目正文")
    _write_skill(user, "code-review", "用户版", body="用户正文")
    _write_skill(user, "user-only", "只在用户级")

    loader = SkillLoader([(project, "project"), (user, "user")])
    by_name = {s.name: s for s in loader.skills}
    assert by_name["code-review"].description == "项目版"  # first-wins
    assert by_name["code-review"].source == "project"
    assert by_name["user-only"].source == "user"
    collisions = [d for d in loader.diagnostics if d.code == "collision"]
    assert len(collisions) == 1
    assert "code-review" in collisions[0].message


def test_default_skill_directories_order_and_dedupe(tmp_path, monkeypatch):
    fake_home_dir = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home_dir))

    directories = default_skill_directories(tmp_path / "proj")
    assert [scope for _path, scope in directories] == ["project", "user"]
    assert directories[0][0] == (tmp_path / "proj" / "skills").resolve()
    assert directories[1][0] == (fake_home_dir / ".lsm" / "skills").resolve()

    # home/skills 与用户级解析后相同(如 home 指向 ~/.lsm)时去重
    deduped = default_skill_directories(fake_home_dir / ".lsm")
    assert len(deduped) == 1


# ── readable_roots: 工作区外的 skill 正文可读、不可写 ─────────────


def test_user_skill_outside_workspace_is_readable_via_read_file(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    user_skills = tmp_path / "user-home" / ".lsm" / "skills"
    path = _write_skill(user_skills, "global-skill", "全局的", body="全局正文")

    loader = SkillLoader([(user_skills, "user")])
    listing = loader.listing(workspace)
    location = listing.split("<location>", 1)[1].split("</location>", 1)[0]
    assert location == str(path)  # 工作区外 → 绝对路径

    bounded = LocalFileOperations(workspace)
    assert "Error" in _read_file(location, operations=bounded)

    opened = LocalFileOperations(workspace, readable_roots=[user_skills])
    assert "全局正文" in _read_file(location, operations=opened)


def test_readable_roots_are_read_only(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    fs = LocalFileOperations(workspace, readable_roots=[skills])

    with pytest.raises(PermissionError):
        fs.write_text(str(skills / "SKILL.md"), "nope")
    fs.write_text("inside.md", "ok")  # workspace 内仍可写
    assert (workspace / "inside.md").read_text() == "ok"
