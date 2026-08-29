"""Runtime registry for model API translators."""

from __future__ import annotations

from dataclasses import dataclass

from lsm_harness.ai.types import StreamFunction


@dataclass(frozen=True)
class ApiProvider:
    api: str
    stream: StreamFunction


_API_PROVIDERS: dict[str, ApiProvider] = {}


def register_api_provider(provider: ApiProvider) -> None:
    _API_PROVIDERS[provider.api] = provider


def unregister_api_provider(api: str) -> None:
    _API_PROVIDERS.pop(api, None)


def resolve_api_provider(api: str) -> ApiProvider:
    try:
        return _API_PROVIDERS[api]
    except KeyError as exc:
        available = ", ".join(sorted(_API_PROVIDERS)) or "none"
        raise ValueError(
            f"Unknown model API '{api}'. Registered APIs: {available}"
        ) from exc


def registered_api_providers() -> tuple[str, ...]:
    return tuple(sorted(_API_PROVIDERS))
