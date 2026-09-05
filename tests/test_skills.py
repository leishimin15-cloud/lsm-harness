"""Skill loader tests: Pi-style lazy listing (metadata only)."""

from pathlib import Path

from lsm_harness.coding_agent.skills import SkillLoader, parse_skill


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
