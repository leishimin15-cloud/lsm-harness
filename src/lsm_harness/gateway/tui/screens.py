"""TUI 选择器界面。

对齐 Tau 的 *PickerScreen 模式,只留最小骨架:/model、/sessions、
/tree 三个选择器共用。Enter 选中 → dismiss(value);Escape → dismiss(None)。

``ResumeSessionScreen`` 则是 Pi 式的专用会话工作台：它需要搜索、
范围、排序与路径视图，不能再塞进通用小弹窗。
"""

from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList
from textual.widgets.option_list import Option


class PickerScreen(ModalScreen[str | None]):
    """通用单选弹窗。"""

    CSS = """
    PickerScreen {
        align: center middle;
    }
    #picker-box {
        width: 72;
        max-height: 70%;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }
    #picker-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str, options: list[tuple[str, str]]):
        super().__init__()
        self._title = title
        self._values = [value for _label, value in options]
        self._labels = [label for label, _value in options]

    def compose(self) -> ComposeResult:
        with Container(id="picker-box"):
            yield Label(self._title, id="picker-title")
            yield OptionList(*[Option(label) for label in self._labels])

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self._values[event.option_index])

    def action_cancel(self) -> None:
        self.dismiss(None)


class ResumeSessionScreen(ModalScreen[str | None]):
    """Pi-style searchable session selector used by ``/resume``."""

    CSS = """
    ResumeSessionScreen {
        align: center middle;
        background: $background 70%;
    }
    #resume-box {
        width: 96%;
        height: 90%;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }
    #resume-title {
        height: 1;
        text-style: bold;
    }
    #resume-controls {
        height: 1;
        color: $text-muted;
    }
    #resume-search {
        height: 3;
        margin: 1 0;
    }
    #resume-list {
        height: 1fr;
    }
    #resume-help {
        height: 2;
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("tab", "toggle_scope", "Scope", show=False, priority=True),
        Binding("ctrl+s", "cycle_sort", "Sort", show=False, priority=True),
        Binding("ctrl+p", "toggle_path", "Path", show=False, priority=True),
        Binding("up", "previous", "Previous", show=False, priority=True),
        Binding("down", "next", "Next", show=False, priority=True),
    ]

    def __init__(
        self,
        sessions: list[dict],
        *,
        current_session_id: str,
        current_cwd: str,
    ) -> None:
        super().__init__()
        self._sessions = sessions
        self._current_session_id = current_session_id
        self._current_cwd = Path(current_cwd).resolve()
        self._current_folder_only = True
        self._sort_mode = "recent"
        self._show_path = False
        self._visible: list[dict] = []
        self._filter_error = ""

    def compose(self) -> ComposeResult:
        with Container(id="resume-box"):
            yield Label("", id="resume-title")
            yield Label("", id="resume-controls")
            yield Input(
                placeholder='Search sessions (text, re:<pattern>, or "exact phrase")',
                id="resume-search",
            )
            yield OptionList(id="resume-list")
            yield Label("", id="resume-help")

    def on_mount(self) -> None:
        self._refresh()
        self.query_one("#resume-search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "resume-search":
            self._refresh()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "resume-search":
            return
        event.prevent_default()
        event.stop()
        self._choose_highlighted()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if 0 <= event.option_index < len(self._visible):
            self.dismiss(str(self._visible[event.option_index]["id"]))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_toggle_scope(self) -> None:
        self._current_folder_only = not self._current_folder_only
        self._refresh()

    def action_cycle_sort(self) -> None:
        self._sort_mode = "fuzzy" if self._sort_mode == "recent" else "recent"
        self._refresh()

    def action_toggle_path(self) -> None:
        self._show_path = not self._show_path
        self._refresh()

    def action_previous(self) -> None:
        option_list = self.query_one("#resume-list", OptionList)
        if option_list.option_count:
            current = option_list.highlighted or 0
            option_list.highlighted = max(0, current - 1)

    def action_next(self) -> None:
        option_list = self.query_one("#resume-list", OptionList)
        if option_list.option_count:
            current = option_list.highlighted
            option_list.highlighted = min(
                option_list.option_count - 1,
                0 if current is None else current + 1,
            )

    def _choose_highlighted(self) -> None:
        option_list = self.query_one("#resume-list", OptionList)
        index = option_list.highlighted
        if index is not None and 0 <= index < len(self._visible):
            self.dismiss(str(self._visible[index]["id"]))

    def _refresh(self) -> None:
        query_widget = self.query_one_optional("#resume-search", Input)
        query = query_widget.value.strip() if query_widget is not None else ""
        rows = [
            row for row in self._sessions
            if not self._current_folder_only or self._same_cwd(row.get("cwd", ""))
        ]
        self._filter_error = ""
        rows = self._filter_rows(rows, query)
        if self._sort_mode == "fuzzy" and query:
            rows.sort(key=lambda row: self._fuzzy_score(row, query), reverse=True)
        else:
            rows.sort(key=lambda row: str(row.get("updated_at", "")), reverse=True)
        self._visible = rows

        scope = "Current Folder" if self._current_folder_only else "All"
        sort = "Fuzzy" if self._sort_mode == "fuzzy" else "Recent"
        title = self.query_one_optional("#resume-title", Label)
        controls = self.query_one_optional("#resume-controls", Label)
        help_label = self.query_one_optional("#resume-help", Label)
        option_list = self.query_one_optional("#resume-list", OptionList)
        if title is not None:
            title.update(f"Resume Session ({scope})")
        if controls is not None:
            controls.update(
                f"Tab scope  •  Ctrl+S sort  •  Ctrl+P path    "
                f"Scope: {scope}  |  Sort: {sort}"
            )
        if option_list is not None:
            option_list.set_options(
                [Option(self._format_row(row)) for row in self._visible]
            )
            option_list.highlighted = 0 if self._visible else None
        if help_label is not None:
            if self._filter_error:
                help_label.update(f"Invalid regex: {self._filter_error}")
            elif self._visible:
                help_label.update(
                    f"{len(self._visible)} session(s)  •  Enter select  •  Esc cancel"
                )
            else:
                help_label.update("No matching sessions  •  Esc cancel")

    def _same_cwd(self, cwd: str) -> bool:
        if not cwd:
            return False
        try:
            return Path(cwd).resolve() == self._current_cwd
        except OSError:
            return False

    def _filter_rows(self, rows: list[dict], query: str) -> list[dict]:
        if not query:
            return rows
        if query.startswith("re:"):
            try:
                pattern = re.compile(query[3:], re.IGNORECASE)
            except re.error as exc:
                self._filter_error = str(exc)
                return []
            return [row for row in rows if pattern.search(self._search_text(row))]
        if len(query) >= 2 and query[0] == query[-1] == '"':
            phrase = query[1:-1].casefold()
            return [row for row in rows if phrase in self._search_text(row).casefold()]
        if self._sort_mode == "fuzzy":
            return [row for row in rows if self._fuzzy_score(row, query) >= 0.2]
        words = query.casefold().split()
        return [
            row for row in rows
            if all(word in self._search_text(row).casefold() for word in words)
        ]

    @staticmethod
    def _search_text(row: dict) -> str:
        return " ".join(
            str(row.get(key, "")) for key in ("title", "id", "cwd", "path")
        )

    def _fuzzy_score(self, row: dict, query: str) -> float:
        needle = query.casefold()
        candidates = (
            str(row.get("title", "")).casefold(),
            str(row.get("id", "")).casefold(),
            str(row.get("cwd", "")).casefold(),
        )
        return max(SequenceMatcher(None, needle, value).ratio() for value in candidates)

    def _format_row(self, row: dict) -> str:
        marker = "*" if row.get("id") == self._current_session_id else " "
        title = str(row.get("title") or "untitled").replace("\n", " ")
        count = int(row.get("message_count", 0))
        age = _relative_age(str(row.get("updated_at", "")))
        label = f"{marker} {title}    {count}msgs  {age}"
        if self._show_path:
            path = str(row.get("cwd") or row.get("path") or "unknown path")
            label += f"\n    {row.get('id', '')[:8]}  {path}"
        return label


def _relative_age(value: str) -> str:
    """Compact relative timestamp matching Pi's session-list density."""
    if not value:
        return ""
    try:
        then = datetime.fromisoformat(value.replace("Z", "+00:00"))
        now = datetime.now(then.tzinfo) if then.tzinfo else datetime.now()
        seconds = max(0, int((now - then).total_seconds()))
    except ValueError:
        return ""
    if seconds < 60:
        return "now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    if days < 30:
        return f"{days}d"
    return f"{days // 30}mo"


class ApiKeyScreen(ModalScreen[str | None]):
    """Masked API-key prompt used by ``/login`` and first-run setup."""

    CSS = """
    ApiKeyScreen {
        align: center middle;
    }
    #api-key-box {
        width: 72;
        height: auto;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }
    #api-key-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #api-key-help {
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, provider_name: str, api_key_name: str):
        super().__init__()
        self._provider_name = provider_name
        self._api_key_name = api_key_name

    def compose(self) -> ComposeResult:
        with Container(id="api-key-box"):
            yield Label(
                f"Login to {self._provider_name}", id="api-key-title"
            )
            yield Label(
                f"Enter {self._api_key_name}", id="api-key-help"
            )
            yield Input(
                password=True,
                id="api-key",
            )

    def on_mount(self) -> None:
        self.query_one("#api-key", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        key = event.value.strip()
        # Input.Submitted bubbles. Stop it before dismissing the modal so the
        # same Enter/key can never reach LSMTui.on_input_submitted as chat.
        event.prevent_default()
        event.stop()
        event.input.clear()
        if key:
            self.dismiss(key)

    def action_cancel(self) -> None:
        self.dismiss(None)
