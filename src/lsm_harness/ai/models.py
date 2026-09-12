"""Model capability helpers used by ``stream_simple``."""

from __future__ import annotations

from dataclasses import replace

from lsm_harness.ai.types import (
    CacheRetention,
    Model,
    ThinkingBudgets,
    ThinkingLevel,
)


THINKING_LEVELS: tuple[ThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

# 旧三档 → 七档,仅在读取旧配置 / 旧 JSONL(thinking_level_change
# entry)/ 旧 API 入参时转换;新记录一律写七档名称。
LEGACY_THINKING_LEVELS: dict[str, ThinkingLevel] = {
    "disabled": "off",
    "auto": "medium",
    "enabled": "high",
}


def normalize_thinking_level(value: str) -> ThinkingLevel:
    """把可能的旧三档/未知值归一到七档;未知值回退 ``off``。"""
    if value in THINKING_LEVELS:
        return value  # type: ignore[return-value]
    return LEGACY_THINKING_LEVELS.get(value, "off")


def available_thinking_levels(model: Model) -> tuple[ThinkingLevel, ...]:
    """当前模型实际支持的档位(Shift+Tab 循环范围)。

    显式 ``None`` 映射 = 不支持(如 K3 的 ``off``:null——不能真正
    关闭 thinking);``xhigh``/``max`` 缺省视为不支持,其余档位缺省
    按能力默认支持。非 reasoning 模型只支持 ``off``。
    """
    if not model.reasoning:
        return ("off",)
    missing = object()
    available: list[ThinkingLevel] = []
    for level in THINKING_LEVELS:
        mapped = model.thinking_level_map.get(level, missing)
        if mapped is None:
            continue
        if level in {"xhigh", "max"} and mapped is missing:
            continue
        available.append(level)
    return tuple(available) or ("off",)


def clamp_thinking_level(model: Model, requested: ThinkingLevel) -> ThinkingLevel:
    available = list(available_thinking_levels(model))
    if requested in available:
        return requested
    index = THINKING_LEVELS.index(requested)
    for level in THINKING_LEVELS[index + 1:]:
        if level in available:
            return level
    for level in reversed(THINKING_LEVELS[:index]):
        if level in available:
            return level
    return "off"


def resolve_cache_retention(
    model: Model,
    requested: CacheRetention,
) -> CacheRetention:
    if model.cache_control_format == "none":
        return "none"
    if requested == "long" and not model.supports_long_cache_retention:
        return "short"
    return requested


def resolve_max_tokens(
    model: Model,
    requested: int,
    reasoning: ThinkingLevel,
    thinking_budgets: ThinkingBudgets | None = None,
) -> int:
    """Fit output tokens to model limits and Anthropic thinking budgets."""
    resolved = requested
    if model.thinking_format == "anthropic":
        override = (
            thinking_budgets.get(reasoning)
            if thinking_budgets is not None
            else None
        )
        raw_budget = (
            str(override)
            if override is not None
            else model.thinking_level_map.get(reasoning)
        )
        if raw_budget:
            try:
                budget = int(raw_budget)
            except ValueError:
                pass
            else:
                resolved = max(resolved, budget + 1024)
    if model.max_tokens:
        resolved = min(resolved, model.max_tokens)
    return resolved


def with_model_id(model: Model, model_id: str) -> Model:
    return replace(model, id=model_id)
