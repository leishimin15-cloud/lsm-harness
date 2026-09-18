"""Eval variants: named, self-contained CodingSession configurations.

Eval 2.0's first-class A/B unit.  A variant bundles everything needed to
build one *independent* :class:`CodingSession` — provider/model, an extra
system-prompt line, settings overrides and a tool policy — instead of the
old ad-hoc ``dict[str, dict]`` of extra ``run_scenario`` kwargs.

``build_variant_session`` is the public seam for "build two different
CodingSessions and run the same scenario against each": pass a real client
+ stream_fn for integration runs, or omit them to resolve from
``settings.provider`` / the environment (as ``ModelRuntime`` does).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsm_harness.ai.types import ModelClient, StreamFunction
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect


@dataclass
class EvalToolPolicy:
    """Which tools a variant exposes to the model.

    ``allow`` empty means "all tools" (the native registry); otherwise only
    the listed names are kept.  ``deny`` removes names regardless of
    ``allow``.  Filtering is applied after the registry is built, so the
    policy cannot add tools that don't exist.
    """

    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)


@dataclass
class EvalVariant:
    """One named configuration for an eval run.

    ``system_prompt`` is an *extra* directive appended to the built-in
    persona (not a wholesale replacement) — the minimal surface needed to
    A/B prompt style without forking ``Session.build_system``.
    """

    name: str
    provider: str = ""
    model: str = ""
    small_model: str = ""
    system_prompt: str = ""
    settings_overrides: dict[str, Any] = field(default_factory=dict)
    tool_policy: EvalToolPolicy | None = None


def apply_variant_settings(settings: Settings, variant: EvalVariant) -> None:
    """Write a variant's fields onto ``settings`` (overrides earlier values).

    Applied after scenario-level ``settings_overrides`` so a variant always
    wins where both are set.
    """
    if variant.provider:
        settings.provider = variant.provider
    if variant.model:
        settings.model = variant.model
    if variant.small_model:
        settings.small_model = variant.small_model
    elif variant.provider or variant.model:
        # 与 api_key 同类陷阱:provider 变了但 small_model 还停在 .env
        # 的默认(如 deepseek-v4-flash),压缩摘要会把别家模型名发给
        # 当前 provider。清空让 ModelRuntime 回填 provider 默认小模型。
        settings.small_model = ""
    if variant.system_prompt:
        settings.system_prompt = variant.system_prompt
    for key, value in variant.settings_overrides.items():
        setattr(settings, key, value)


def apply_tool_policy(harness: Harness, policy: EvalToolPolicy) -> None:
    """Filter ``harness.tools`` down to the policy's allow/deny set.

    The registry and the agent's ``AgentState.tools`` both point at the
    filtered copy afterwards, so the loop and ``harness.tools`` agree.
    """
    names = harness.tools.tool_names()
    if policy.allow:
        names = [n for n in names if n in policy.allow]
    if policy.deny:
        names = [n for n in names if n not in policy.deny]
    filtered = harness.tools.filter(names)
    harness.tools = filtered
    harness.agent.state.tools = filtered


def build_variant_session(
    variant: EvalVariant,
    *,
    home: Path,
    api_key: str | None = None,
    client: ModelClient | None = None,
    stream_fn: StreamFunction | None = None,
) -> Harness:
    """Build one :class:`CodingSession` from a variant.

    With ``client``/``stream_fn`` given, this is a scripted/injected
    session; without them, ``ModelRuntime`` resolves a real client from
    ``variant.provider`` + the environment (``api_key`` defaults to
    ``Settings``' env resolution).
    """
    settings = Settings(home=home)
    if api_key is not None:
        settings.api_key = api_key
    apply_variant_settings(settings, variant)
    settings.ensure_home()
    conn = connect(home, check_same_thread=False)
    harness = Harness(
        settings=settings,
        client=client,
        stream_fn=stream_fn,
        conn=conn,
    )
    if variant.tool_policy is not None:
        apply_tool_policy(harness, variant.tool_policy)
    return harness
