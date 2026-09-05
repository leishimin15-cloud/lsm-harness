"""Chapter 8: context engineering — tool-output truncation + project context
files in the system prompt."""

from __future__ import annotations

from pathlib import Path

from lsm_harness.coding_agent.operations import LocalFileOperations
from lsm_harness.coding_agent.resources import (
    format_project_context,
    load_project_context_files,
)
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.loop.agent import run_loop
from lsm_harness.loop.governance import ContextGovernor, GovernanceConfig
from lsm_harness.coding_agent.skills import SkillLoader
from lsm_harness.runtime import Session
from lsm_harness.tools.registry import Tool, ToolRegistry
from lsm_harness.tools.filesystem import _grep, _read_file
from lsm_harness.tools.shell import _format_output
from lsm_harness.tools.truncate import (
    GREP_MAX_LINE_LENGTH,
    truncate_head,
    truncate_line,
    truncate_tail,
)
from lsm_harness.types import ModelResponse, ToolCall

from helpers import QueueClient


# ── truncate_tail / truncate_head ────────────────────────────────


def test_truncate_tail_keeps_end_within_line_limit():
    content = "\n".join(f"line {i}" for i in range(1, 11))
    result = truncate_tail(content, max_lines=3)
    assert result.content == "line 8\nline 9\nline 10"
    assert result.truncated and result.truncated_by == "lines"
    assert result.total_lines == 10 and result.output_lines == 3


def test_truncate_tail_respects_byte_limit():
    # 100 lines × ~100 bytes; 250-byte budget must stop by bytes, not lines.
    content = "\n".join(f"row {i:03d} " + "x" * 90 for i in range(100))
    result = truncate_tail(content, max_lines=1000, max_bytes=250)
    assert result.truncated and result.truncated_by == "bytes"
    assert result.output_bytes <= 250
    assert result.content.endswith("x" * 90)  # tail preserved


def test_truncate_tail_single_oversized_line_keeps_tail_bytes():
    content = "A" * 100 + "B" * 100
    result = truncate_tail(content, max_bytes=50)
    assert result.last_line_partial
    assert result.content == "B" * 50
    assert result.truncated_by == "bytes"


def test_truncate_head_keeps_beginning():
    content = "\n".join(f"line {i}" for i in range(1, 11))
    result = truncate_head(content, max_lines=3)
    assert result.content == "line 1\nline 2\nline 3"
    assert result.truncated and result.truncated_by == "lines"


def test_truncate_head_single_oversized_line_keeps_head_bytes():
    content = "A" * 100 + "B" * 100
    result = truncate_head(content, max_bytes=50)
    assert result.first_line_partial
    assert result.content == "A" * 50


def test_truncation_never_splits_multibyte_characters():
    # 😀 is 4 UTF-8 bytes; a budget of N*4+2 must keep whole emoji only.
    content = "😀" * 10  # 40 bytes, single line
    result = truncate_tail(content, max_bytes=14)
    assert result.last_line_partial
    assert result.content == "😀" * 3  # 12 bytes; a 4th would exceed
    assert result.content.encode("utf-8")  # decodes cleanly


def test_no_truncation_when_content_fits():
    result = truncate_tail("a\nb\nc")
    assert not result.truncated and result.truncated_by is None
    assert result.content == "a\nb\nc"


def test_truncate_line_marks_long_lines():
    short = "x" * 100
    assert truncate_line(short) == short
    long = "y" * (GREP_MAX_LINE_LENGTH + 100)
    result = truncate_line(long)
    assert result.endswith("... [truncated]")
    assert len(result) == GREP_MAX_LINE_LENGTH + len("... [truncated]")


# ── shell wiring ─────────────────────────────────────────────────


def test_format_output_keeps_tail_and_spills_full_output(tmp_path):
    stdout = "\n".join(f"line {i}" for i in range(1, 3001))
    result = _format_output(stdout, "", workspace=tmp_path)
    assert "[Showing lines 1001–3000 of 3000." in result
    assert "Full output:" in result
    assert "line 3000" in result and "line 1\n" not in result
    # §7.2: the spill lands INSIDE the workspace, addressed by a
    # workspace-relative path that read_file can actually open.
    path = result.split("Full output: ", 1)[1].rstrip("]")
    assert not Path(path).is_absolute()
    assert path.startswith(".lsm/tool-results/exec-")
    assert (tmp_path / path).read_text() == stdout
    # §7.4: the spilled file is readable through the read_file tool itself
    read_back = _read_file(path, operations=LocalFileOperations(tmp_path))
    assert "line 1" in read_back


def test_format_output_small_output_untouched():
    assert _format_output("hello", "") == "hello"
    assert _format_output("", "") == "(no output)"
    err = _format_output("out", "boom", exit_code=2)
    assert err.startswith("Error: command exited with code 2.")
    assert "[stderr]\nboom" in err


def test_format_output_single_huge_line_notice(tmp_path):
    stdout = "z" * (60 * 1024)
    result = _format_output(stdout, "", workspace=tmp_path)
    assert "of line 1 (line is 60.0 KB)" in result
    assert "Full output:" in result
    path = result.split("Full output: ", 1)[1].rstrip("]")
    assert (tmp_path / path).read_text() == stdout


# ── read_file / grep wiring ──────────────────────────────────────


def _ops(tmp_path) -> LocalFileOperations:
    return LocalFileOperations(tmp_path)


def test_read_file_offset_beyond_eof_is_an_error(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\nthree")
    result = _read_file("a.txt", offset=99, operations=_ops(tmp_path))
    assert result == "Error: offset 99 is beyond end of file (3 lines total)."


def test_read_file_byte_cap_protects_against_huge_lines(tmp_path):
    (tmp_path / "big.txt").write_text("x" * (60 * 1024) + "\ntail line")
    result = _read_file("big.txt", operations=_ops(tmp_path))
    assert "[Truncated by bytes:" in result
    assert "tail line" not in result  # byte budget hit on line 1


def test_read_file_normal_output_keeps_numbered_lines(tmp_path):
    (tmp_path / "b.txt").write_text("alpha\nbeta")
    result = _read_file("b.txt", operations=_ops(tmp_path))
    assert "2 lines" in result
    assert "     1│alpha" in result and "     2│beta" in result


def test_grep_truncates_minified_lines(tmp_path):
    (tmp_path / "min.js").write_text("var needle = '" + "x" * 5000 + "';")
    result = _grep("needle", path=".", operations=_ops(tmp_path))
    assert "... [truncated]" in result
    body_line = result.splitlines()[1]
    assert len(body_line) < 600


# ── project context files ────────────────────────────────────────


def test_context_files_collected_global_then_ancestors_then_cwd(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("org rules")
    team = tmp_path / "team"
    team.mkdir()
    (team / "AGENTS.md").write_text("team rules")
    project = team / "app"
    project.mkdir()
    (project / "CLAUDE.md").write_text("project rules")

    files = load_project_context_files(project, agent_dir=tmp_path)
    assert [content for _, content in files] == [
        "org rules",      # global (agent_dir == tmp_path, deduped)
        "team rules",     # ancestor
        "project rules",  # cwd, most specific, last
    ]


def test_context_files_skip_missing_and_unreadable(tmp_path):
    cwd = tmp_path / "app"
    cwd.mkdir()
    assert load_project_context_files(cwd, agent_dir=tmp_path) == []
    (cwd / "CLAUDE.md").write_text("   ")
    assert load_project_context_files(cwd, agent_dir=tmp_path) == []


def test_format_project_context_wraps_xml_with_paths(tmp_path):
    files = [(tmp_path / "CLAUDE.md", "rule one"), (tmp_path / "sub" / "AGENTS.md", "rule two")]
    xml = format_project_context(files)
    assert xml.startswith("<project_context>")
    assert xml.endswith("</project_context>")
    assert f'<project_instructions path="{tmp_path / "CLAUDE.md"}">' in xml
    assert "rule one" in xml and "rule two" in xml
    assert format_project_context([]) == ""


def test_build_system_includes_global_claude_md(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("always use pnpm in this org")
    settings = Settings(api_key="k", home=tmp_path)
    session = Session(
        settings, conn=connect(tmp_path), client=QueueClient(), session_id="s-ctx"
    )
    system = session.build_system("hello", lambda *a: None)
    assert "<project_context>" in system
    assert "always use pnpm in this org" in system
    # project context sits before the trailing time/model metadata
    assert system.index("<project_context>") < system.index("当前时间：")


# ── batch D: skills lazy loading (§7.1 / §7.4) ───────────────────


def _write_skill(home: Path, name: str = "code-review", body: str = "审查步骤正文：先读 diff") -> Path:
    skill_dir = home / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: Review code for correctness.\n---\n{body}",
        encoding="utf-8",
    )
    return path


def _skill_session(tmp_path, client=None, **overrides):
    settings = Settings(api_key="k", home=tmp_path, **overrides)
    return Session(
        settings,
        conn=connect(tmp_path),
        client=client or QueueClient(),
        session_id="s-skills",
    )


def test_system_prompt_lists_skill_metadata_without_bodies(tmp_path):
    """§7.4: the system prompt carries ONLY the skill listing — name,
    description, location, and the read_file contract — never the body."""
    _write_skill(tmp_path)
    session = _skill_session(tmp_path)  # lazy is the default
    system = session.build_system("please review my code", lambda *a: None)
    assert "<available_skills>" in system
    assert "<name>code-review</name>" in system
    assert "<description>Review code for correctness.</description>" in system
    assert "<location>" in system and "SKILL.md" in system
    assert "use read_file to load its SKILL.md before acting" in system
    assert "审查步骤正文" not in system  # the body is NOT inlined


def test_listed_skill_location_is_readable_via_read_file(tmp_path):
    """§7.4: the <location> in the listing resolves through the read_file
    tool's workspace root, so the model can pull the body on demand."""
    _write_skill(tmp_path)
    loader = SkillLoader([tmp_path / "skills"])
    listing = loader.listing(tmp_path)
    location = listing.split("<location>", 1)[1].split("</location>", 1)[0]
    assert location == "skills/code-review/SKILL.md"
    read_back = _read_file(location, operations=LocalFileOperations(tmp_path))
    assert "审查步骤正文" in read_back


# ── batch D: one primary truncation per tool result (§7.3 / §7.4) ──


def _one_tool_registry(name: str, output: str) -> "ToolRegistry":
    tools = ToolRegistry()
    tools.register(
        Tool(
            name,
            f"fake {name}",
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            lambda value: output,
        )
    )
    return tools


def _run_one_tool(tmp_path, tool_name: str, output: str):
    client = QueueClient(
        ModelResponse(tool_calls=[ToolCall("1", tool_name, {"value": "x"})]),
        ModelResponse(text="done"),
    )
    governor = ContextGovernor(
        config=GovernanceConfig(max_result_chars=1000, offload_threshold_chars=4000),
        home=tmp_path,
    )
    return run_loop(
        client=client,
        model="scripted",
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        tools=_one_tool_registry(tool_name, output),
        max_iterations=4,
        max_tokens=100,
        emit=lambda *_: None,
        governor=governor,
    )


def test_self_truncating_tool_is_marked_not_cut_again(tmp_path):
    """read/exec/grep truncate inside the tool (Pi-style); the governor
    must NOT apply its own truncation on top — one primary truncation."""
    huge = "\n".join(f"row {i} " + "x" * 200 for i in range(500))
    result = _run_one_tool(tmp_path, "read_file", huge)
    record = result.tool_calls[0]
    assert record["output"] == huge  # governor left it untouched
    assert "已截断" not in record["output"]  # governor's note never appears
    assert record["details"]["governed"] == "tool"


def test_unknown_tool_result_falls_back_to_governor(tmp_path):
    """Third-party/unknown tools without internal truncation get the
    ContextGovernor fallback, marked as governor-managed."""
    big = "y" * 3000  # over max_result_chars, under the offload threshold
    result = _run_one_tool(tmp_path, "mystery_tool", big)
    record = result.tool_calls[0]
    assert len(record["output"]) < len(big)
    assert "已截断" in record["output"]
    assert record["details"]["governed"] == "governor"
