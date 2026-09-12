"""TUI 通用单选弹窗(PickerScreen)。

对齐 Tau 的 *PickerScreen 模式,只留最小骨架:/model、/sessions、
/tree 三个选择器共用。Enter 选中 → dismiss(value);Escape → dismiss(None)。
"""

from __future__ import annotations

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
