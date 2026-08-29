"""Provider-neutral error classification and context-overflow detection."""

from __future__ import annotations

from lsm_harness.ai.types import ErrorCategory, ModelResponse


def categorize_error(exc: Exception) -> ErrorCategory:
    message = str(exc).lower()
    if any(marker in message for marker in (
        "insufficient_quota",
        "billing",
        "quota exceeded",
        "account is not active",
        "payment required",
        "credit balance",
        "free quota",
        "usage limit",
        "exceeded your current quota",
    )):
        return "arrearage"
    if any(marker in message for marker in (
        "rate_limit",
        "rate limit",
        "too many requests",
        "try again later",
        "status code: 429",
        "error code: 429",
    )):
        return "rate_limit"
    if type(exc).__name__ in {
        "Timeout",
        "TimeoutError",
        "ConnectionError",
        "ConnectionResetError",
        "BrokenPipeError",
        "RemoteDisconnected",
        "IncompleteRead",
        "ReadTimeout",
        "ConnectTimeout",
    }:
        return "transient"
    return "permanent"


def is_retryable(category: ErrorCategory) -> bool:
    return category in {"rate_limit", "transient"}


def is_context_overflow(response: ModelResponse) -> bool:
    """Recognize common explicit and silent context-overflow outcomes."""
    message = response.error_message.lower()
    if any(marker in message for marker in (
        "context length",
        "context window",
        "maximum context",
        "too many tokens",
        "prompt is too long",
        "model_context_window_exceeded",
        "context_length_exceeded",
    )):
        return True
    return (
        response.stop_reason == "length"
        and not response.text
        and not response.tool_calls
        and response.usage.output_tokens == 0
    )
