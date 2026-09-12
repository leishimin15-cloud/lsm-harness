"""Pi-compatible ``models.json`` loading and provider composition.

The AI layer owns built-in provider/model descriptions.  This product layer
loads ``<LSM_HOME>/models.json`` and composes the user overlay without putting
credentials into ``Model`` values.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator

from lsm_harness.ai.providers import PROVIDERS, Provider


AuthSource = Literal[
    "stored", "environment", "models_json_key", "models_json_command"
]


_MODELS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["providers"],
    "properties": {
        "providers": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "baseUrl": {"type": "string", "minLength": 1},
                    "apiKey": {"type": "string", "minLength": 1},
                    "api": {"type": "string", "minLength": 1},
                    "headers": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "authHeader": {"type": "boolean"},
                    "compat": {"type": "object"},
                    "models": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id"],
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "name": {"type": "string", "minLength": 1},
                                "api": {"type": "string", "minLength": 1},
                                "baseUrl": {"type": "string", "minLength": 1},
                                "reasoning": {"type": "boolean"},
                                "thinkingLevelMap": {"type": "object"},
                                "input": {
                                    "type": "array",
                                    "items": {"enum": ["text", "image"]},
                                },
                                "cost": {"type": "object"},
                                "contextWindow": {"type": "number"},
                                "maxTokens": {"type": "number"},
                                "headers": {"type": "object"},
                                "compat": {"type": "object"},
                            },
                        },
                    },
                    "modelOverrides": {"type": "object"},
                },
            },
        }
    },
}


def _strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments without touching quoted strings."""
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and following == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and following == "*":
            index += 2
            while index + 1 < len(text) and text[index:index + 2] != "*/":
                index += 1
            index += 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _error_path(error: Any) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    return path or "root"


@dataclass(frozen=True)
class ModelConfig:
    providers: dict[str, dict[str, Any]]
    error: str | None = None

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls({})
        except OSError as exc:
            return cls({}, f"Failed to load models.json: {exc}\n\nFile: {path}")
        try:
            parsed = json.loads(_strip_json_comments(text))
        except json.JSONDecodeError as exc:
            return cls({}, f"Failed to parse models.json: {exc}\n\nFile: {path}")
        errors = sorted(
            Draft202012Validator(_MODELS_SCHEMA).iter_errors(parsed),
            key=lambda item: list(item.absolute_path),
        )
        if errors:
            details = "\n".join(
                f"  - {_error_path(error)}: {error.message}"
                for error in errors
            )
            return cls(
                {}, f"Invalid models.json schema:\n{details}\n\nFile: {path}"
            )
        return cls(dict(parsed["providers"]))


@dataclass(frozen=True)
class AuthStatus:
    configured: bool
    source: AuthSource | None = None
    label: str | None = None


@dataclass(frozen=True)
class ModelCatalog:
    providers: dict[str, Provider]
    config: ModelConfig

    @property
    def error(self) -> str | None:
        return self.config.error

    def auth_status(
        self,
        provider_id: str,
        *,
        stored_key: str = "",
        explicit: str = "",
    ) -> AuthStatus:
        if explicit.strip() or stored_key.strip():
            return AuthStatus(True, "stored")
        provider_config = self.config.providers.get(provider_id, {})
        raw = provider_config.get("apiKey")
        if isinstance(raw, str):
            if raw.startswith("!"):
                return AuthStatus(True, "models_json_command")
            names = _environment_names(raw)
            if names:
                configured = all(os.getenv(name) is not None for name in names)
                return AuthStatus(
                    configured,
                    "environment" if configured else None,
                    ", ".join(names) if configured else None,
                )
            return AuthStatus(True, "models_json_key")
        provider = self.providers[provider_id]
        if provider.key_env and os.getenv(provider.key_env, "").strip():
            return AuthStatus(True, "environment", provider.key_env)
        return AuthStatus(False)

    def configured_key(self, provider_id: str) -> str:
        raw = self.config.providers.get(provider_id, {}).get("apiKey")
        if not isinstance(raw, str):
            return ""
        return _resolve_config_value(raw)


_ENV_PATTERN = re.compile(
    r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))"
)


def _environment_names(value: str) -> tuple[str, ...]:
    escaped = value.replace("$$", "").replace("$!", "")
    return tuple(
        dict.fromkeys(match.group(1) or match.group(2) for match in _ENV_PATTERN.finditer(escaped))
    )


def _resolve_config_value(value: str) -> str:
    if value.startswith("!"):
        completed = subprocess.run(
            value[1:],
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return completed.stdout.strip()
    literal_dollar = "\0LSM_DOLLAR\0"
    literal_bang = "\0LSM_BANG\0"
    escaped = value.replace("$$", literal_dollar).replace("$!", literal_bang)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name not in os.environ:
            raise ValueError(f"Environment variable {name} is not set")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, escaped).replace(
        literal_dollar, "$"
    ).replace(literal_bang, "!")


def _custom_metadata(
    definition: dict[str, Any],
    provider_config: dict[str, Any],
    inherited: dict[str, Any] | None,
) -> dict[str, Any]:
    inherited = dict(inherited or {})
    compat = {
        **(provider_config.get("compat") or {}),
        **(definition.get("compat") or {}),
    }
    cost = definition.get("cost") or {}
    inherited_cost = inherited.get("cost", (0.0, 0.0, 0.0, 0.0))
    inherited.update({
        "name": definition.get("name", inherited.get("name", definition["id"])),
        "source_api": definition.get(
            "api", provider_config.get("api", inherited.get("source_api", ""))
        ),
        "base_url": definition.get(
            "baseUrl", provider_config.get("baseUrl", inherited.get("base_url"))
        ),
        "reasoning": definition.get("reasoning", inherited.get("reasoning", False)),
        "input_modalities": tuple(
            definition.get("input", inherited.get("input_modalities", ("text",)))
        ),
        "context_window": int(
            definition.get("contextWindow", inherited.get("context_window", 128_000))
        ),
        "max_tokens": int(
            definition.get("maxTokens", inherited.get("max_tokens", 16_384))
        ),
        "cost": (
            float(cost.get("input", inherited_cost[0])),
            float(cost.get("output", inherited_cost[1])),
            float(cost.get("cacheRead", inherited_cost[2])),
            float(cost.get("cacheWrite", inherited_cost[3])),
        ),
        "thinking_level_map": dict(
            definition.get(
                "thinkingLevelMap", inherited.get("thinking_level_map", {})
            )
        ),
        "thinking_format": compat.get(
            "thinkingFormat", inherited.get("thinking_format", "none")
        ),
        "allow_empty_thinking_signature": bool(
            compat.get(
                "allowEmptySignature",
                inherited.get("allow_empty_thinking_signature", False),
            )
        ),
        "force_adaptive_thinking": bool(
            compat.get(
                "forceAdaptiveThinking",
                inherited.get("force_adaptive_thinking", False),
            )
        ),
        "headers": {
            **inherited.get("headers", {}),
            **(definition.get("headers") or {}),
        },
    })
    return inherited


def _apply_override(metadata: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    synthetic = {"id": "override", **override}
    return _custom_metadata(synthetic, {}, metadata)


def _compose_provider(
    provider_id: str,
    base: Provider | None,
    config: dict[str, Any],
) -> Provider:
    metadata = dict(base.model_metadata) if base else {}
    model_ids = list(base.models or (base.model,)) if base else []
    for definition in config.get("models") or []:
        model_id = definition["id"]
        metadata[model_id] = _custom_metadata(
            definition, config, metadata.get(model_id)
        )
        if model_id not in model_ids:
            model_ids.append(model_id)
    for model_id, override in (config.get("modelOverrides") or {}).items():
        if model_id in metadata and isinstance(override, dict):
            metadata[model_id] = _apply_override(metadata[model_id], override)
    if not model_ids:
        raise ValueError(
            f"Provider {provider_id}: at least one model is required"
        )
    first = model_ids[0]
    api = config.get("api") or (base.api if base else metadata[first].get("source_api"))
    base_url = config.get("baseUrl") or (base.base_url if base else None)
    if not api:
        raise ValueError(f'Provider {provider_id}: "api" is required')
    if not base_url:
        raise ValueError(f'Provider {provider_id}: "baseUrl" is required')
    return Provider(
        api=api,
        key_env=base.key_env if base else "",
        base_url=base_url,
        model=base.model if base and base.model in model_ids else first,
        small_model=(
            base.small_model if base and base.small_model in model_ids else first
        ),
        name=config.get("name") or (base.name if base else provider_id),
        api_key_name=base.api_key_name if base else "API key",
        key_url=base.key_url if base else "",
        context_window=base.context_window if base else 128_000,
        max_output_tokens=base.max_output_tokens if base else 16_384,
        input_cost_per_million=base.input_cost_per_million if base else 0.0,
        output_cost_per_million=base.output_cost_per_million if base else 0.0,
        cache_read_cost_per_million=(
            base.cache_read_cost_per_million if base else 0.0
        ),
        cache_write_cost_per_million=(
            base.cache_write_cost_per_million if base else 0.0
        ),
        models=tuple(model_ids),
        auth_mode=(
            "bearer" if config.get("authHeader") else (
                base.auth_mode if base else "api_key"
            )
        ),
        headers={**(base.headers if base else {}), **(config.get("headers") or {})},
        model_metadata=metadata,
    )


def load_model_catalog(home: Path) -> ModelCatalog:
    """Compose built-ins with one immutable ``models.json`` snapshot."""
    config = ModelConfig.load(home / "models.json")
    providers = dict(PROVIDERS)
    # Pi registers llama.cpp from its bundled coding-agent extension rather
    # than from the AI package's built-in provider catalog. It has no models
    # until a router is configured and queried, but it is always selectable
    # from /login.
    providers["llama.cpp"] = Provider(
        api="openai-completions",
        key_env="LLAMA_CPP_URL",
        base_url="http://127.0.0.1:8080/v1",
        model="",
        small_model="",
        name="llama.cpp",
        api_key_name="llama.cpp server URL",
    )
    if config.error:
        return ModelCatalog(providers, config)
    for provider_id, provider_config in config.providers.items():
        try:
            providers[provider_id] = _compose_provider(
                provider_id, providers.get(provider_id), provider_config
            )
        except (KeyError, TypeError, ValueError) as exc:
            return ModelCatalog(
                dict(PROVIDERS),
                ModelConfig({}, f"Invalid models.json provider {provider_id}: {exc}"),
            )
    return ModelCatalog(providers, config)


__all__ = [
    "AuthStatus",
    "ModelCatalog",
    "ModelConfig",
    "load_model_catalog",
]
