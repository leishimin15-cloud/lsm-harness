"""Resolve an authenticated startup model before constructing a Harness."""

from __future__ import annotations

import os
from pathlib import Path

from lsm_harness.ai.providers import (
    ProviderAuthError,
    ProviderConfigurationError,
    canonical_provider_name,
)
from lsm_harness.coding_agent.auth_storage import read_api_key
from lsm_harness.coding_agent.model_config import (
    AuthStatus,
    ModelCatalog,
    load_model_catalog,
)
from lsm_harness.config import Settings


_LEGACY_PROVIDER_IDS = {
    "google": "gemini",
    "zai": "glm",
    # 旧版 /login 以 "kimi" 存键;目录里的正式 id 是 kimi-coding。
    "kimi-coding": "kimi",
}


def resolve_product_api_key(
    provider: str,
    *,
    home: Path,
    explicit: str = "",
    catalog: ModelCatalog | None = None,
) -> str:
    """Resolve explicit, stored, then environment authentication."""
    provider = canonical_provider_name(provider)
    if explicit.strip():
        return explicit.strip()
    catalog = catalog or load_model_catalog(home)
    stored = read_api_key(home, provider)
    if not stored and provider in _LEGACY_PROVIDER_IDS:
        stored = read_api_key(home, _LEGACY_PROVIDER_IDS[provider])
    configured = catalog.configured_key(provider)
    provider_env = catalog.providers[provider].key_env
    environment = os.getenv(provider_env, "").strip() if provider_env else ""
    return stored or configured or environment


def provider_auth_status(
    provider: str,
    *,
    home: Path,
    explicit: str = "",
    catalog: ModelCatalog | None = None,
) -> AuthStatus:
    """Return Pi-style configured state and credential source."""
    provider = canonical_provider_name(provider)
    catalog = catalog or load_model_catalog(home)
    stored = read_api_key(home, provider)
    if not stored and provider in _LEGACY_PROVIDER_IDS:
        stored = read_api_key(home, _LEGACY_PROVIDER_IDS[provider])
    return catalog.auth_status(
        provider, stored_key=stored, explicit=explicit
    )


def provider_is_configured(
    provider: str,
    *,
    home: Path,
    explicit: str = "",
    catalog: ModelCatalog | None = None,
) -> bool:
    """Return whether a provider is usable by this Coding Agent product."""
    status = provider_auth_status(
        provider, home=home, explicit=explicit, catalog=catalog
    )
    return status.configured


def resolve_startup_settings(settings: Settings) -> str | None:
    """Mutate ``settings`` to a usable provider/model and return a notice.

    This is the product-layer equivalent of Pi's startup model fallback.  A
    stale selected provider must not prevent the UI from reaching ``/login``.
    """
    requested = canonical_provider_name(settings.provider or "")
    settings.provider = requested
    catalog = load_model_catalog(settings.home)
    providers = catalog.providers
    config_notice = catalog.error
    if requested and requested not in providers:
        raise ProviderConfigurationError(
            f"Unknown LSM_PROVIDER '{requested}'. Pick one of: "
            f"{', '.join(providers)}"
        )

    candidates = [requested] if requested else []
    candidates.extend(name for name in providers if name not in candidates)

    for name in candidates:
        explicit = settings.api_key if name == requested else ""
        if not provider_is_configured(
            name,
            home=settings.home,
            explicit=explicit,
            catalog=catalog,
        ):
            continue
        provider = providers[name]
        key = resolve_product_api_key(
            name, explicit=explicit, home=settings.home, catalog=catalog
        )
        if name == requested or not requested:
            settings.provider = name
            settings.api_key = key
            settings.model = settings.model or provider.model
            settings.small_model = settings.small_model or provider.small_model
            return config_notice

        old_ref = f"{requested}/{settings.model or providers[requested].model}"
        settings.provider = name
        settings.api_key = key
        settings.model = provider.model
        settings.small_model = provider.small_model
        fallback_notice = (
            f"Could not restore {old_ref} (no auth configured). "
            f"Using {name}/{provider.model}."
        )
        return "\n".join(filter(None, (config_notice, fallback_notice)))

    selected = requested or "deepseek"
    provider = providers[selected]
    raise ProviderAuthError(
        f"No API key for provider '{selected}'. Use /login {selected} "
        f"or set {provider.key_env}."
    )


__all__ = [
    "provider_is_configured",
    "provider_auth_status",
    "resolve_product_api_key",
    "resolve_startup_settings",
]
