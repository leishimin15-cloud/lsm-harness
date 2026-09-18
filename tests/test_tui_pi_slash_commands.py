"""Pi-facing slash command parity for the Textual TUI."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace

from textual.app import App, ComposeResult
from textual.widgets import Input, OptionList

from lsm_harness.gateway.tui.autocomplete import _COMMANDS
from lsm_harness.gateway.tui.app import LSMTui
from lsm_harness.gateway.tui.commands import CommandsMixin
from lsm_harness.gateway.tui.screens import ResumeSessionScreen, _relative_age


class _Harness:
    def __init__(self) -> None:
        self.session = SimpleNamespace(session_id="current-session")
        self.switched_to = ""

    def switch_session(self, reference: str) -> str | None:
        self.switched_to = reference
        return "restored-session"


class _Commands(CommandsMixin):
    def __init__(self) -> None:
        self.harness = _Harness()
        self.state = SimpleNamespace(running=False, is_compacting=False)
        self.notes: list[str] = []
        self.resume_opened = False
        self.compact_started = False
        self.status_refreshes = 0

    def _note(self, text: str) -> None:
        self.notes.append(text)

    def _open_session_picker(self) -> None:
        self.resume_opened = True

    def _start_manual_compaction(self) -> None:
        self.compact_started = True

    def _refresh_status(self) -> None:
        self.status_refreshes += 1


def _run(command: str) -> _Commands:
    target = _Commands()
    asyncio.run(target._handle_command(command))
    return target


def test_pi_resume_and_compact_appear_in_completion_catalog() -> None:
    inserted = {value for _label, value in _COMMANDS}
    assert "/resume" in inserted
    assert "/compact" in inserted
    assert "/hotkeys" in inserted
    assert "/skills" in inserted


def test_skills_command_lists_skills_and_diagnostics() -> None:
    target = _Commands()
    target.harness.skill_loader = SimpleNamespace(
        skills=[
            SimpleNamespace(
                name="code-review",
                source="project",
                description="Review code for correctness.",
                path="/proj/.lsm/skills/code-review/SKILL.md",
            )
        ],
        diagnostics=[
            SimpleNamespace(
                code="collision",
                path="/home/.lsm/skills/code-review/SKILL.md",
                message="skill 'code-review' already loaded from /proj/...",
            )
        ],
    )
    asyncio.run(target._handle_command("/skills"))
    text = "\n".join(target.notes)
    assert "code-review" in text
    assert "Review code for correctness." in text
    assert "collision" in text


def test_skills_command_reports_empty() -> None:
    target = _Commands()
    target.harness.skill_loader = SimpleNamespace(skills=[], diagnostics=[])
    asyncio.run(target._handle_command("/skills"))
    assert any("No skills loaded" in note for note in target.notes)


def test_resume_without_id_opens_picker() -> None:
    target = _run("/resume")
    assert target.resume_opened is True


def test_resume_with_id_switches_directly() -> None:
    target = _run("/resume abc123")
    assert target.harness.switched_to == "abc123"
    assert target.status_refreshes == 1


def test_compact_starts_manual_compaction() -> None:
    target = _run("/compact")
    assert target.compact_started is True


class _ResumeHost(App):
    def __init__(self, rows: list[dict]) -> None:
        super().__init__()
        self.rows = rows
        self.selected: str | None = None

    def compose(self) -> ComposeResult:
        yield Input(id="host-input")

    def on_mount(self) -> None:
        self.push_screen(
            ResumeSessionScreen(
                self.rows,
                current_session_id="current",
                current_cwd="/workspace/a",
            ),
            self._picked,
        )

    def _picked(self, value: str | None) -> None:
        self.selected = value


def _session_row(
    session_id: str,
    title: str,
    *,
    cwd: str,
    updated_at: str,
) -> dict:
    return {
        "id": session_id,
        "title": title,
        "cwd": cwd,
        "path": f"/sessions/{session_id}.jsonl",
        "updated_at": updated_at,
        "message_count": 2,
    }


def test_resume_screen_filters_scope_search_and_selects() -> None:
    rows = [
        _session_row(
            "current",
            "Current task",
            cwd="/workspace/a",
            updated_at="2026-01-02T00:00:00",
        ),
        _session_row(
            "other",
            "Needle task",
            cwd="/workspace/b",
            updated_at="2026-01-01T00:00:00",
        ),
    ]
    app = _ResumeHost(rows)

    async def main() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ResumeSessionScreen)
            assert screen.query_one("#resume-list", OptionList).option_count == 1

            await pilot.press("tab")
            assert screen.query_one("#resume-list", OptionList).option_count == 2

            search = screen.query_one("#resume-search", Input)
            search.value = "needle"
            await pilot.pause()
            assert screen.query_one("#resume-list", OptionList).option_count == 1

            search.value = "nedle"
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert screen.query_one("#resume-list", OptionList).option_count == 1
            await pilot.press("enter")
            await pilot.pause()
            assert app.selected == "other"

    asyncio.run(main())


def test_resume_screen_rejects_invalid_regex_without_crashing() -> None:
    rows = [
        _session_row(
            "current",
            "Current task",
            cwd="/workspace/a",
            updated_at="2026-01-02T00:00:00",
        )
    ]
    app = _ResumeHost(rows)

    async def main() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ResumeSessionScreen)
            screen.query_one("#resume-search", Input).value = "re:["
            await pilot.pause()
            assert screen.query_one("#resume-list", OptionList).option_count == 0
            assert screen._filter_error

    asyncio.run(main())


def test_relative_age_is_compact() -> None:
    timestamp = (datetime.now() - timedelta(minutes=5)).isoformat()
    assert _relative_age(timestamp) == "5m"


class _BlockingCompactor:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def compact(self) -> bool:
        self.started.set()
        self.release.wait(timeout=2)
        return False


class _Input:
    def __init__(self) -> None:
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _CompactionHost:
    _start_manual_compaction = LSMTui._start_manual_compaction
    _finish_manual_compaction = LSMTui._finish_manual_compaction

    def __init__(self) -> None:
        self.harness = _BlockingCompactor()
        self.state = SimpleNamespace(running=False, is_compacting=False)
        self._compaction_worker = None
        self.input = _Input()
        self.notes: list[str] = []

    def _refresh_status(self) -> None:
        pass

    def _note(self, text: str) -> None:
        self.notes.append(text)

    def _call_ui(self, callback, *args) -> None:
        callback(*args)

    def query_one(self, _selector, _widget_type):
        return self.input


def test_manual_compaction_runs_off_the_ui_thread_and_restores_focus() -> None:
    target = _CompactionHost()
    target._start_manual_compaction()
    assert target.harness.started.wait(timeout=1)
    worker = target._compaction_worker
    assert worker is not None and worker.is_alive()
    assert target.state.is_compacting is True

    target.harness.release.set()
    worker.join(timeout=2)

    assert worker.is_alive() is False
    assert target.state.is_compacting is False
    assert target.input.focused is True
    assert any("Nothing to compact" in note for note in target.notes)
