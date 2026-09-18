"""输入框补全:`@` 文件模糊搜索 + `/` 斜杠命令(阶段三:autocomplete)。

``SuggestionOverlay`` 是输入框上方的 OptionList 弹层;``ChatInput`` 是
Input 子类,弹层可见时把 Enter/Tab/↑↓/Esc 路由到弹层,其余键走 Input 默认。

文件模糊匹配是子序列匹配(子串是特例),与 `tau autocomplete` 同思路,
但不做评分排序——命中按路径字典序,够小项目用。

按键热路径规则(2026-09-12 实机卡顿修复):

- 普通文本(非 ``/`` ``@`` 开头)立即返回,不碰任何 reactive/overlay 状态;
- ``/login`` 的 provider 目录只加载一次并缓存(禁止每键重载目录),
  登录成功/目录重载后由 ``invalidate_providers()`` 失效;
- ``@`` 文件索引在后台线程用 ``os.walk`` 直接剪枝建立(不遍历
  ``.git/.venv/node_modules``),UI 线程零扫描;
- 文件补全有 ~20ms debounce:新输入作废旧任务,只应用最新 request id
  的结果(Pi 同款节奏)。
"""

from __future__ import annotations

import os
import threading

from pathlib import Path

from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from lsm_harness.coding_agent.model_config import load_model_catalog

# 与 commands.py 的 _handle_command 一一对应(label 显示, id 插入值)。
_COMMANDS: list[tuple[str, str]] = [
    ("/help    Show command list", "/help"),
    ("/hotkeys Show keyboard shortcuts", "/hotkeys"),
    ("/login   Configure a provider", "/login "),
    ("/model   Switch model", "/model"),
    ("/tree    Browse session tree", "/tree"),
    ("/resume  Resume a different session", "/resume"),
    ("/compact Manually compact the session context", "/compact"),
    ("/summary Show context summary", "/summary"),
    ("/skills  List loaded skills and diagnostics", "/skills"),
    ("/new     Start a new session", "/new"),
    ("/usage   Show token usage", "/usage"),
    ("/follow  Queue a follow-up", "/follow "),
    ("/sessions List sessions (compatibility alias)", "/sessions"),
    ("/quit    Exit", "/quit"),
]

_SKIP_DIRS = {
    ".git", ".lsm", "__pycache__", "node_modules",
    ".pytest_cache", ".mypy_cache", ".venv", "venv",
}

_MAX_FILES = 4000
_MAX_SHOWN = 20


def _subsequence(needle: str, haystack: str) -> bool:
    """needle 的每个字符按序出现在 haystack 里(大小写不敏感)。"""
    if not needle:
        return True
    it = iter(haystack.lower())
    return all(ch in it for ch in needle.lower())


def _iter_workspace_files(root: Path) -> list[str]:
    """workspace 相对路径列表(os.walk 直接剪枝——``rglob`` 会先遍历
    进 ``.git``/``.venv`` 再过滤,大项目首次 ``@`` 会卡住 UI)。"""
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            files.append(str(Path(dirpath, name).relative_to(root)))
            if len(files) >= _MAX_FILES:
                return files
    return files


class SuggestionOverlay(OptionList):
    """输入框上方的补全弹层。

    ``update_for(value)`` 按当前输入刷新;无候选则隐藏。
    Tab(经 ChatInput 路由)接受高亮项写入输入框。
    """

    _DEBOUNCE_S = 0.02

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.display = False
        self._workspace: Path | None = None
        self._file_cache: list[str] | None = None
        self._index_token = 0
        # provider 目录缓存:/login 输入期间不得重复 load_model_catalog。
        self._providers_cache: list[tuple[str, str]] | None = None
        # 连续键入只应用最新一次的结果(debounce + 去重渲染)。
        self._request_id = 0
        self._current_items: tuple[tuple[str, str], ...] = ()
        # 刚接受过补全的值:避免接受后立即因 Changed 重新弹出。
        self._just_accepted: str = ""

    def set_workspace(self, workspace: Path) -> None:
        """后台线程建立文件索引,不阻塞 UI;就绪后若输入停在 ``@``
        上,自动补一轮刷新。"""
        self._workspace = workspace
        self._file_cache = None
        self._index_token += 1
        token = self._index_token
        threading.Thread(
            target=self._build_index,
            args=(workspace, token),
            daemon=True,
            name="lsm-file-index",
        ).start()

    def invalidate_providers(self) -> None:
        """登录成功 / models.json 重载后调用:下次 /login 重新读目录。"""
        self._providers_cache = None

    def _build_index(self, workspace: Path, token: int) -> None:
        files = _iter_workspace_files(workspace)
        try:
            self.app.call_from_thread(self._apply_index, files, token)
        except Exception:
            pass  # 应用退出中,索引丢弃

    def _apply_index(self, files: list[str], token: int) -> None:
        if token != self._index_token:
            return  # workspace 已换,旧索引作废
        self._file_cache = files
        try:
            value = self.app.query_one("#input", Input).value
        except Exception:
            return
        if value.startswith("@"):
            self._apply_file_suggestions(self._request_id, value)

    def update_for(self, value: str) -> None:
        if value == self._just_accepted:
            if self.display:
                self.hide_overlay()
            return
        self._just_accepted = ""
        self._request_id += 1  # 任何新输入都作废旧补全任务
        if value.startswith("@"):
            # ~20ms debounce:连续键入只应用最后一次的匹配结果
            request_id = self._request_id
            self.set_timer(
                self._DEBOUNCE_S,
                lambda: self._apply_file_suggestions(request_id, value),
            )
            return
        if value.startswith("/login "):
            prefix = value[7:].strip().lower()
            self._set_items([
                (label, text)
                for label, text in self._provider_items()
                if not prefix or prefix in text[7:].lower() or prefix in label.lower()
            ])
            return
        if value.startswith("/") and " " not in value:
            self._set_items([
                (label, text) for label, text in _COMMANDS
                if text.strip().startswith(value)
            ])
            return
        # 普通文本:快速返回,不碰 overlay 状态
        if self.display:
            self.hide_overlay()

    def _apply_file_suggestions(self, request_id: int, value: str) -> None:
        if request_id != self._request_id or not value.startswith("@"):
            return  # 已被更新的输入作废
        self._set_items(self._match_files(value[1:]))

    def _provider_items(self) -> list[tuple[str, str]]:
        """provider 候选(缓存;只有 invalidate_providers 后才重读目录)。"""
        if self._providers_cache is None:
            app = self.app
            harness = getattr(app, "harness", None)
            settings = (
                harness.settings
                if harness is not None
                else getattr(app, "_startup_settings", None)
            )
            if settings is None:
                return []
            providers = load_model_catalog(settings.home).providers
            self._providers_cache = [
                (f"{provider.name} · API key", f"/login {provider_id}")
                for provider_id, provider in sorted(
                    providers.items(),
                    key=lambda pair: (pair[1].name.lower(), pair[0]),
                )
            ]
        return self._providers_cache

    def _set_items(self, items: list[tuple[str, str]]) -> None:
        """候选没变化时不重建 OptionList(避免每键 render/layout 抖动)。"""
        if not items:
            self._current_items = ()
            if self.display:
                self.hide_overlay()
            return
        keyed = tuple(items)
        if keyed != self._current_items:
            self.clear_options()
            self.add_options([Option(label, id=text) for label, text in items])
            self.highlighted = 0
            self._current_items = keyed
        if not self.display:
            self.display = True

    def hide_overlay(self) -> None:
        self.display = False

    def accept_highlighted(self, input_widget: Input) -> bool:
        """把高亮项写进输入框(光标到末尾);无高亮返回 False。"""
        if not self.display or self.highlighted is None:
            return False
        option = self.get_option_at_index(self.highlighted)
        if option is None or option.id is None:
            return False
        input_widget.value = option.id
        input_widget.action_end()
        self._just_accepted = option.id
        self.hide_overlay()
        return True

    def _match_files(self, needle: str) -> list[tuple[str, str]]:
        if self._workspace is None or self._file_cache is None:
            return []  # 索引尚在后台构建,就绪后 _apply_index 会补一轮
        hits = [p for p in self._file_cache if _subsequence(needle, p)]
        return [(f"@{p}", f"@{p}") for p in hits[:_MAX_SHOWN]]


class ChatInput(Input):
    """输入框:补全弹层可见时把 Enter/Tab/↑↓/Esc 路由到弹层。

    Tab = 只接受高亮项;Enter = 接受斜杠命令并立即提交,文件补全则
    只接受不提交;↑/↓ = 移动高亮;Esc = 关弹层(不放行给 abort)。
    """

    async def _on_key(self, event) -> None:
        overlay = getattr(self.app, "_suggestion_overlay", None)
        if overlay is not None and overlay.display:
            if event.key == "escape":
                overlay.hide_overlay()
                event.prevent_default()
                event.stop()
                return
            if event.key == "tab":
                if overlay.accept_highlighted(self):
                    event.prevent_default()
                    event.stop()
                    return
            if event.key == "enter":
                is_command = self.value.startswith("/")
                if overlay.accept_highlighted(self) and not is_command:
                    event.prevent_default()
                    event.stop()
                    return
            if event.key in ("up", "down"):
                if event.key == "up":
                    overlay.action_cursor_up()
                else:
                    overlay.action_cursor_down()
                event.prevent_default()
                event.stop()
                return
        await super()._on_key(event)
