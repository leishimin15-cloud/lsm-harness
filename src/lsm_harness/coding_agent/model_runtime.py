"""Provider/model composition owned outside ``CodingSession``."""

from __future__ import annotations

from dataclasses import replace

from lsm_harness.ai import providers
from lsm_harness.ai.providers import canonical_provider_name
from lsm_harness.ai.registry import registered_api_providers
from lsm_harness.ai.stream import stream_simple
from lsm_harness.ai.types import Model, ModelClient, StreamFunction
from lsm_harness.coding_agent.startup import resolve_product_api_key
from lsm_harness.coding_agent.model_config import load_model_catalog
from lsm_harness.config import Settings


class ModelRuntime:
    """Own the coherent client/model/stream triple for a coding session."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: ModelClient | None = None,
        stream_fn: StreamFunction | None = None,
    ) -> None:
        self.settings = settings
        self.catalog = load_model_catalog(settings.home)
        resolved_key = (
            resolve_product_api_key(
                settings.provider,
                home=settings.home,
                explicit=settings.api_key,
                catalog=self.catalog,
            )
            if settings.provider
            else settings.api_key
        )
        self.client = client or providers.get_client(
            provider_name=settings.provider,
            api_key=resolved_key,
            base_url=settings.base_url or None,
            model=settings.model,
            small_model=settings.small_model,
            thinking=settings.thinking,
            catalog=self.catalog.providers,
        )
        if client is None:
            settings.api_key = resolved_key
        if hasattr(self.client, "_resolved_model"):
            if not settings.model:
                settings.model = self.client._resolved_model
            if not settings.small_model:
                settings.small_model = self.client._resolved_small_model
        resolved_model = getattr(self.client, "model", None)
        self.model = (
            resolved_model
            if isinstance(resolved_model, Model)
            else Model(
                id=settings.model or "injected-model",
                api="legacy-client",
                provider=settings.provider or "injected",
            )
        )
        self.stream_fn = stream_fn or self.build_stream_fn(
            self.model, settings.api_key
        )

    @staticmethod
    def build_stream_fn(model: Model, api_key: str) -> StreamFunction:
        if model.api not in registered_api_providers():
            raise ValueError(
                f"model api {model.api!r} is not registered; "
                "pass stream_fn=... for scripted clients"
            )

        def registry_stream(model, context, options):
            return stream_simple(
                model, context, replace(options, api_key=api_key)
            )

        return registry_stream

    def switch(
        self,
        provider_name: str,
        *,
        model: str = "",
        small_model: str = "",
    ) -> None:
        """Build a complete candidate before atomically swapping runtime state."""
        provider_name = canonical_provider_name(provider_name)
        self.catalog = load_model_catalog(self.settings.home)
        provider = self.catalog.providers[provider_name]
        new_model = model or provider.model
        new_small = small_model or provider.small_model
        resolved_key = resolve_product_api_key(
            provider_name,
            home=self.settings.home,
            catalog=self.catalog,
        )
        if not resolved_key and provider_name == self.settings.provider:
            resolved_key = self.settings.api_key
        candidate_client = providers.get_client(
            provider_name=provider_name,
            api_key=resolved_key,
            base_url=self.settings.base_url or None,
            model=new_model,
            small_model=new_small,
            thinking=self.settings.thinking,
            catalog=self.catalog.providers,
        )
        resolved_model = getattr(candidate_client, "model", None)
        candidate_model = (
            resolved_model
            if isinstance(resolved_model, Model)
            else providers.get_model(
                provider_name,
                new_model,
                base_url=self.settings.base_url or None,
                catalog=self.catalog.providers,
            )
        )
        candidate_stream = self.build_stream_fn(candidate_model, resolved_key)

        self.client = candidate_client
        self.model = candidate_model
        self.stream_fn = candidate_stream
        self.settings.provider = provider_name
        self.settings.api_key = resolved_key
        self.settings.model = new_model
        self.settings.small_model = new_small


__all__ = ["ModelRuntime"]
