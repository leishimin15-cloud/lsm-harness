"""TUI 斜杠命令与选择器回调(阶段二批 ⑦:从 app.py 拆出)。

``CommandsMixin`` 只依赖 App 提供的骨架方法/属性(``harness`` /
``state`` / ``_note`` / ``_refresh_status`` / ``_rebuild_from_session`` /
``push_screen`` / ``exit``),本身不接触 Textual 控件或线程模型。
"""

from __future__ import annotations

from lsm_harness.agent.messages import message_preview
from lsm_harness.ai.providers import (
    ProviderConfigurationError,
    available_models,
)
from lsm_harness.coding_agent.app import RunBusyError
from lsm_harness.coding_agent.auth_storage import AuthStorageError, save_api_key
from lsm_harness.coding_agent.model_config import load_model_catalog
from lsm_harness.coding_agent.startup import (
    provider_auth_status,
    provider_is_configured,
)

from .screens import ApiKeyScreen, PickerScreen


class CommandsMixin:
    """斜杠命令 + /model /sessions /tree 选择器回调。"""

    async def _handle_command(self, text: str) -> None:
        h = self.harness
        cmd = text.strip()

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()

        elif cmd == "/help":
            self._note("[bold]Commands:[/bold]")
            for c, desc in [
                ("/login [provider]", "Configure an API key"),
                ("/model", "Switch model"),
                ("/tree", "Session history tree"),
                ("/sessions", "List all sessions"),
                ("/summary", "View context summary"),
                ("/new", "Start new session"),
                ("/usage", "Token usage stats"),
                ("/follow <文本>", "运行中排队 follow-up"),
                ("/quit", "Exit"),
                ("Esc", "中断当前轮"),
                ("Shift+Tab", "Cycle thinking level"),
                ("@filename", "Fuzzy file search"),
            ]:
                self._note(f"  {c:15s} {desc}")

        elif cmd == "/login" or cmd.startswith("/login "):
            providers = self._model_catalog().providers
            provider_ref = cmd[len("/login"):].strip()
            provider = self._resolve_login_provider(provider_ref)
            if provider_ref and provider is None:
                self._note(
                    f"[yellow]Unknown provider: {provider_ref}. "
                    f"Choose from {', '.join(providers)}[/yellow]"
                )
                return
            self._open_login_picker(provider)

        elif not h:
            self._note("[yellow]No model is configured. Use /login first.[/yellow]")

        elif cmd == "/model":
            self._open_model_picker()

        elif cmd.startswith("/model:"):
            try:
                catalog = self._model_catalog()
                providers = catalog.providers
                idx = int(cmd.split(":")[1]) - 1
                configured = tuple(
                    name
                    for name in providers
                    if provider_is_configured(
                        name,
                        home=h.settings.home,
                        explicit=(
                            h.settings.api_key
                            if name == h.settings.provider
                            else ""
                        ),
                        catalog=catalog,
                    )
                )
                models = available_models(
                    providers=configured, catalog=providers
                )
                if 0 <= idx < len(models):
                    selected = models[idx]
                    p = providers[selected.provider]
                    try:
                        h.switch_model(
                            selected.provider,
                            model=selected.id,
                            small_model=p.small_model,
                        )
                    except RunBusyError:
                        self._note("[yellow]运行中不可切换模型（等本轮结束）[/yellow]")
                        return
                    except ProviderConfigurationError as exc:
                        self._note(f"[red]模型切换失败: {exc}[/red]")
                        return
                    self._refresh_status()
                    self._note(
                        f"[green]→ {selected.provider}/{selected.id}[/green]"
                    )
            except (ValueError, IndexError):
                self._note("[yellow]Invalid model number[/yellow]")

        elif cmd == "/tree":
            self._open_tree_picker()

        elif cmd == "/sessions":
            rows = h.list_sessions(15)
            if not rows:
                self._note("[dim]No sessions[/dim]")
                return
            options = [
                (f"{'*' if item['id'] == h.session.session_id else ' '} "
                 f"{item['id'][:8]}  {item['message_count']}msgs  "
                 f"{item['title'] or 'untitled'}", item["id"])
                for item in rows
            ]
            self.push_screen(
                PickerScreen("选择会话(Enter 切换,Esc 取消)", options),
                self._on_session_picked,
            )

        elif cmd == "/summary":
            info = h.summary_info()
            if info:
                self._note(f"[bold]Summary v{info['version']}:[/bold]")
                self._note(info["summary"])
            else:
                self._note("[dim]No summary yet[/dim]")

        elif cmd == "/new":
            try:
                sid = h.new_session()
            except RunBusyError:
                self._note("[yellow]运行中不可新建会话（等本轮结束）[/yellow]")
                return
            self._refresh_status()
            self._note(f"[dim]New session: {sid[:8]}[/dim]")

        elif cmd.startswith("/follow"):
            # 运行中排队 follow-up:本轮结束后接着跑(Tau Alt+Enter 的
            # 单行输入替代)。
            payload = cmd[len("/follow"):].strip()
            if not payload:
                self._note("[dim]用法: /follow <文本>[/dim]")
                return
            if not self.state.running:
                self._note("[dim]当前没有运行中的轮次,直接发送即可[/dim]")
                return
            if not h.follow_up(payload):
                self._note("[dim]本轮已结束,消息未排队——请重新发送[/dim]")
                return
            self._note(f"[magenta]follow-up ›[/magenta] {payload}")

        elif cmd == "/usage":
            s = h.usage_summary()
            if s["total_input"] == 0:
                self._note("[dim]No usage recorded[/dim]")
            else:
                self._note(
                    f"Total: ↑{s['total_input']:,} ↓{s['total_output']:,} tokens"
                )
                for model, stats in s.get("by_model", {}).items():
                    self._note(
                        f"  {model}: {stats['calls']} calls, "
                        f"↑{stats['input']:,} ↓{stats['output']:,}"
                    )

        else:
            self._note(f"[yellow]Unknown: {cmd}[/yellow]")

    # ── 选择器 ─────────────────────────────────────────────

    def _resolve_login_provider(self, provider_ref: str) -> str | None:
        """Match Pi's `/login <id-or-display-name>` behavior."""
        normalized = provider_ref.strip().lower()
        if not normalized:
            return None
        for provider_id, provider in self._model_catalog().providers.items():
            if normalized in {provider_id.lower(), provider.name.lower()}:
                return provider_id
        return None

    def _open_login_picker(self, provider: str | None = None) -> None:
        """Open provider selection, then a masked API-key prompt."""
        if provider is not None:
            self._open_api_key_prompt(provider)
            return
        home = (
            self.harness.settings.home
            if self.harness is not None
            else self._startup_settings.home
        )
        catalog = load_model_catalog(home)
        providers = catalog.providers
        options = [
            (
                f"{item.name}"
                + self._provider_auth_label(
                    name, home=home, catalog=catalog
                ),
                name,
            )
            for name, item in sorted(
                providers.items(), key=lambda pair: (pair[1].name.lower(), pair[0])
            )
        ]
        self.push_screen(
            PickerScreen("Select provider to configure:", options),
            self._on_login_provider_picked,
        )

    def _on_login_provider_picked(self, provider: str | None) -> None:
        if provider is not None:
            self._open_api_key_prompt(provider)

    def _provider_auth_label(self, provider: str, *, home, catalog) -> str:
        status = provider_auth_status(
            provider,
            home=home,
            catalog=catalog,
        )
        if not status.configured:
            return " • unconfigured"
        source = status.source or "configured"
        return f" ✓ {source}"

    def _open_api_key_prompt(self, provider: str) -> None:
        item = self._model_catalog().providers[provider]
        self.push_screen(
            ApiKeyScreen(item.name, item.api_key_name),
            lambda key: self._on_api_key_entered(provider, key),
        )

    def _on_api_key_entered(self, provider: str, key: str | None) -> None:
        if key is None:
            return
        home = (
            self.harness.settings.home
            if self.harness is not None
            else self._startup_settings.home
        )
        try:
            path = save_api_key(home, provider, key)
            item = load_model_catalog(home).providers[provider]
            if self.harness is None:
                settings = self._startup_settings
                settings.provider = provider
                settings.api_key = key.strip()
                settings.model = item.model
                settings.small_model = item.small_model
                if not self._initialize_harness(settings):
                    self._note(
                        f"[yellow]Saved credentials for {provider} to {path}, "
                        "but the model runtime could not start.[/yellow]"
                    )
                    return
            else:
                try:
                    self.harness.switch_model(
                        provider,
                        model=item.model,
                        small_model=item.small_model,
                    )
                except RunBusyError:
                    self._note(
                        f"[yellow]Saved credentials for {provider} to {path}. "
                        "Wait for the current run, then select it with /model.[/yellow]"
                    )
                    return
                self._refresh_status()
            self._note(
                f"[green]Saved credentials for {provider} to {path}. "
                f"Selected {provider}/{item.model}.[/green]"
            )
        except (AuthStorageError, ProviderConfigurationError, ValueError) as exc:
            self._note(f"[red]Login failed: {exc}[/red]")

    def _open_model_picker(self) -> None:
        h = self.harness
        if not h:
            return
        catalog = load_model_catalog(h.settings.home)
        providers = catalog.providers
        configured = tuple(
            name
            for name in providers
            if provider_is_configured(
                name,
                home=h.settings.home,
                explicit=(
                    h.settings.api_key
                    if name == h.settings.provider
                    else ""
                ),
                catalog=catalog,
            )
        )
        models = available_models(providers=configured, catalog=providers)
        if not models:
            self._note("[yellow]No configured models. Use /login first.[/yellow]")
            return
        options = [
            (
                f"{'*' if model.provider == h.settings.provider and model.id == h.settings.model else ' '} "
                f"{model.provider:15s} {model.id}",
                f"{model.provider}/{model.id}",
            )
            for model in models
        ]
        self.push_screen(
            PickerScreen(f"选择模型(当前: {h.settings.model})", options),
            self._on_model_picked,
        )

    def _open_tree_picker(self) -> None:
        if self.state.running:
            self._note("[yellow]运行中不可切换节点（先 Esc 中断或等本轮结束）[/yellow]")
            return
        options = self._tree_options()
        if not options:
            self._note("[dim]No history[/dim]")
            return
        self.push_screen(
            PickerScreen("会话树——选节点分支(Enter 选中,Esc 取消)", options),
            self._on_tree_node_picked,
        )

    def _on_model_picked(self, reference: str | None) -> None:
        if reference is None or not self.harness:
            return
        provider, separator, model = reference.partition("/")
        providers = self._model_catalog().providers
        if not separator or provider not in providers:
            self._note(f"[red]Invalid model reference: {reference}[/red]")
            return
        p = providers[provider]
        try:
            self.harness.switch_model(
                provider,
                model=model,
                small_model=p.small_model,
            )
        except RunBusyError:
            self._note("[yellow]运行中不可切换模型（等本轮结束）[/yellow]")
            return
        except (ProviderConfigurationError, ValueError) as exc:
            self._note(f"[red]模型切换失败: {exc}[/red]")
            return
        self._refresh_status()
        self._note(f"[green]→ {provider}/{model}[/green]")

    def _on_session_picked(self, session_id: str | None) -> None:
        if session_id is None or not self.harness:
            return
        try:
            switched = self.harness.switch_session(session_id)
        except RunBusyError:
            self._note("[yellow]运行中不可切换会话（等本轮结束）[/yellow]")
            return
        if switched is None:
            self._note("[yellow]会话匹配失败[/yellow]")
            return
        self._refresh_status()
        self._note(f"[dim]→ session {switched[:8]}[/dim]")

    def _tree_options(self) -> list[tuple[str, str]]:
        """当前路径(到 live leaf)的条目列表:序号 + 预览。"""
        h = self.harness
        if h is None:
            return []
        path, leaf_id = h.current_path_entries()
        if not path:
            return []
        options = []
        for i, entry in enumerate(path, 1):
            marker = "▸" if entry.id == leaf_id else " "
            if entry.type == "message":
                role = getattr(entry.message, "role", "?")
                text = message_preview(entry.message, limit=60).replace("\n", " ")
                label = f"{marker} #{i} [{role}] {text}"
            else:
                label = f"{marker} #{i} ({entry.type})"
            options.append((label, entry.id))
        return options

    def _on_tree_node_picked(self, entry_id: str | None) -> None:
        if entry_id is None or not self.harness:
            return
        if self.state.running:
            self._note("[yellow]运行中不可切换节点（等本轮结束）[/yellow]")
            return
        branched = self.harness.branch_to(entry_id)
        if branched is None:
            self._note("[yellow]分支失败:节点不存在[/yellow]")
            return
        self._rebuild_from_session()
        self._note(f"[dim]↩ 已回到节点 {branched[:8]},后续消息将创建新分支[/dim]")

    def _model_catalog(self):
        home = (
            self.harness.settings.home
            if self.harness is not None
            else self._startup_settings.home
        )
        return load_model_catalog(home)
