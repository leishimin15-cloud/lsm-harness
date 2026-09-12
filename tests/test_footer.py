"""footer_snapshot / format_tokens / git branch / cwd 缩写的纯单元测试。

harness 用 SimpleNamespace 假对象——footer.py 只触它的公开面
(settings / model / session / workspace_root / current_path_messages)。
"""

from __future__ import annotations

from types import SimpleNamespace

from lsm_harness.ai.types import Model
from lsm_harness.coding_agent.footer import (
    _display_cwd,
    _git_branch,
    footer_snapshot,
    format_tokens,
)


def _assistant(**usage) -> SimpleNamespace:
    return SimpleNamespace(role="assistant", usage=usage)


def _harness(tmp_path, messages, window=128_000, **over) -> SimpleNamespace:
    harness = SimpleNamespace(
        settings=SimpleNamespace(
            provider="deepseek", model="k3", thinking="medium"
        ),
        model=Model(
            id="k3", api="x", provider="deepseek", context_window=128_000
        ),
        session=SimpleNamespace(
            session_id="abcdef1234567890",
            # footer 与压缩红线同源的 effective 窗口(真身由 Session 提供)
            effective_context_window=lambda: window,
        ),
        workspace_root=tmp_path,
        current_path_messages=lambda: messages,
    )
    for key, value in over.items():
        setattr(harness, key, value)
    return harness


def test_format_tokens_tiers():
    assert format_tokens(0) == "0"
    assert format_tokens(999) == "999"
    assert format_tokens(1200) == "1.2k"
    assert format_tokens(12_345) == "12k"
    assert format_tokens(1_234_567) == "1.2M"
    assert format_tokens(12_345_678) == "12M"


def test_display_cwd_home_collapse(tmp_path):
    home = tmp_path / "home"
    project = home / "proj"
    project.mkdir(parents=True)
    import pathlib
    from unittest.mock import patch

    with patch.object(pathlib.Path, "home", staticmethod(lambda: home)):
        assert _display_cwd(project) == "~/proj"
        assert _display_cwd(home) == "~"
    assert _display_cwd(tmp_path / "elsewhere") == str(tmp_path / "elsewhere")


def test_git_branch_from_head(tmp_path):
    assert _git_branch(tmp_path) is None  # 非 git 目录

    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    assert _git_branch(tmp_path) == "main"

    # detached HEAD:短哈希
    (git / "HEAD").write_text("abcdef1234567890\n", encoding="utf-8")
    assert _git_branch(tmp_path) == "abcdef1"


def test_git_branch_worktree_pointer(tmp_path):
    real = tmp_path / "real-git"
    real.mkdir()
    (real / "HEAD").write_text("ref: refs/heads/feature\n", encoding="utf-8")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {real}\n", encoding="utf-8")
    assert _git_branch(wt) == "feature"


def test_footer_snapshot_aggregates_history(tmp_path):
    messages = [
        SimpleNamespace(role="user", content="q1"),
        _assistant(input_tokens=1000, output_tokens=100,
                   cache_read_tokens=800, cache_write_tokens=0,
                   cost_total=0.001),
        SimpleNamespace(role="user", content="q2"),
        _assistant(input_tokens=2000, output_tokens=200,
                   cache_read_tokens=1500, cache_write_tokens=100,
                   cost_total=0.002),
    ]
    snap = footer_snapshot(_harness(tmp_path, messages))

    assert snap.provider == "deepseek"
    assert snap.model == "k3"
    assert snap.thinking == "medium"  # 七档原样显示,不再映射三档
    assert snap.session == "abcdef12"
    assert snap.auto_compact is True

    # 累计与最近 turn
    assert snap.total_input == 3000 and snap.total_output == 300
    assert snap.total_cache_read == 2300 and snap.total_cache_write == 100
    assert abs(snap.total_cost - 0.003) < 1e-9
    assert snap.last_input == 2000 and snap.last_output == 200

    # context:最近一次调用的 prompt 大小(input + cache r/w)
    assert snap.context_tokens == 2000 + 1500 + 100
    assert snap.context_window == 128_000
    assert abs(snap.context_percent - 3600 / 128_000 * 100) < 1e-6

    # cache 命中率:最近一次调用 cache_read / 整个 prompt
    assert abs(snap.cache_hit_rate - 1500 / 3600 * 100) < 1e-6


def test_footer_snapshot_empty_session(tmp_path):
    snap = footer_snapshot(_harness(tmp_path, []))
    assert snap.total_input == 0 and snap.total_cost == 0.0
    assert snap.context_tokens == 0
    assert snap.cache_hit_rate is None
    # 窗口为 0 时 percent 为 None(不除零)
    snap0 = footer_snapshot(_harness(tmp_path, [], window=0))
    assert snap0.context_percent is None
