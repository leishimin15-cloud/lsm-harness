"""Pi 风格的底部状态快照:Coding Agent 层的统一 footer 数据源。

TUI/RPC 等前端的 footer 不再各自拼装:身份(provider/model/thinking/
session)、环境(cwd/git branch)、累计用量、context 占用全部由
``footer_snapshot(harness)`` 一次给出。

累计用量从**当前路径的历史消息**现算(Pi 的 footer.ts 同款算法)——
恢复/切换会话天然带全量数据,前端不需要自己维护计数器,也不存在
「重启后累计清零」的问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FooterSnapshot:
    """一次 footer 渲染所需的全部数据(不可变快照,可跨线程传递)。"""

    # 身份
    provider: str
    model: str
    thinking: str  # Pi 七档原样:off/minimal/low/medium/high/xhigh/max
    session: str  # session_id 前 8 位
    # 环境
    cwd: str  # home 缩写成 ~
    git_branch: str | None
    # context
    context_tokens: int  # 最近一次主模型调用的 prompt 大小(input+cache r/w)
    context_window: int
    auto_compact: bool
    # 最近一次主模型 turn
    last_input: int
    last_output: int
    last_cache_read: int
    last_cache_write: int
    # 全程累计(当前路径)
    total_input: int
    total_output: int
    total_cache_read: int
    total_cache_write: int
    total_cost: float

    @property
    def cache_hit_rate(self) -> float | None:
        """最近一次调用的 cache 命中率(cache_read 占整个 prompt 的比例)。"""
        prompt = self.last_input + self.last_cache_read + self.last_cache_write
        if prompt <= 0 or (not self.last_cache_read and not self.last_cache_write):
            return None
        return self.last_cache_read / prompt * 100

    @property
    def context_percent(self) -> float | None:
        if self.context_window <= 0:
            return None
        return self.context_tokens / self.context_window * 100


def format_tokens(count: int) -> str:
    """紧凑 token 计数(Pi footer.ts 的 formatTokens 同款分档)。"""
    if count < 1000:
        return str(count)
    if count < 10000:
        return f"{count / 1000:.1f}k"
    if count < 1000000:
        return f"{round(count / 1000)}k"
    if count < 10000000:
        return f"{count / 1000000:.1f}M"
    return f"{round(count / 1000000)}M"


def _display_cwd(cwd: Path) -> str:
    """home 下的路径缩成 ~/…(Pi 的 formatCwdForFooter 等价物)。"""
    try:
        rel = cwd.resolve().relative_to(Path.home())
    except (ValueError, OSError):
        return str(cwd)
    return "~" if str(rel) == "." else f"~/{rel}"


def _git_branch(root: Path) -> str | None:
    """从 .git/HEAD 读当前分支(不起子进程;worktree 跟 gitdir 指针)。"""
    try:
        git = root / ".git"
        head = git / "HEAD"
        if git.is_file():
            # worktree / submodule:.git 是 "gitdir: <path>" 指针文件
            for line in git.read_text(encoding="utf-8").splitlines():
                if line.startswith("gitdir:"):
                    head = Path(line.removeprefix("gitdir:").strip()) / "HEAD"
                    break
        if not head.is_file():
            return None
        text = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text.startswith("ref: refs/heads/"):
        return text.removeprefix("ref: refs/heads/")
    return text[:7] if text else None  # detached HEAD:短哈希


def footer_snapshot(harness) -> FooterSnapshot:
    """从 CodingSession 现算 footer 快照。

    数据源全是 CodingSession 的公开面:settings / model / session /
    workspace_root / current_path_messages();前端不读 JSONL。
    """
    settings = harness.settings
    last: dict = {}
    total_input = total_output = 0
    total_cache_read = total_cache_write = 0
    total_cost = 0.0
    for message in harness.current_path_messages():
        if getattr(message, "role", None) != "assistant":
            continue
        usage = getattr(message, "usage", None)
        if not usage:
            continue
        last = usage
        total_input += int(usage.get("input_tokens", 0) or 0)
        total_output += int(usage.get("output_tokens", 0) or 0)
        total_cache_read += int(usage.get("cache_read_tokens", 0) or 0)
        total_cache_write += int(usage.get("cache_write_tokens", 0) or 0)
        total_cost += float(usage.get("cost_total", 0.0) or 0.0)

    last_input = int(last.get("input_tokens", 0) or 0)
    last_cache_read = int(last.get("cache_read_tokens", 0) or 0)
    last_cache_write = int(last.get("cache_write_tokens", 0) or 0)
    # context 窗口与压缩红线同源(effective:override > 模型窗口 > 兜底),
    # footer 百分比和自动压缩判断不会出现两个分母。
    context_window = harness.session.effective_context_window()
    return FooterSnapshot(
        provider=settings.provider,
        model=settings.model,
        thinking=settings.thinking,
        session=harness.session.session_id[:8],
        cwd=_display_cwd(harness.workspace_root),
        git_branch=_git_branch(harness.workspace_root),
        context_tokens=last_input + last_cache_read + last_cache_write,
        context_window=context_window,
        # 自动压缩目前始终开启(红线走 settings 的 context budget),
        # 将来加了开关在这里接线。
        auto_compact=True,
        last_input=last_input,
        last_output=int(last.get("output_tokens", 0) or 0),
        last_cache_read=last_cache_read,
        last_cache_write=last_cache_write,
        total_input=total_input,
        total_output=total_output,
        total_cache_read=total_cache_read,
        total_cache_write=total_cache_write,
        total_cost=total_cost,
    )
