"""Model capability helpers used by ``stream_simple``."""

from __future__ import annotations

from dataclasses import replace

from lsm_harness.ai.types import CacheRetention, Model, ThinkingLevel


THINKING_LEVELS: tuple[ThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)


def clamp_thinking_level(model: Model, requested: ThinkingLevel) -> ThinkingLevel:
    available = set(model.thinking_level_map)
    if not available or requested in available:
        return requested if requested in available else "off"
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
) -> int:
    """Fit output tokens to model limits and Anthropic thinking budgets."""
    resolved = requested
    if model.thinking_format == "anthropic":
        raw_budget = model.thinking_level_map.get(reasoning)
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
