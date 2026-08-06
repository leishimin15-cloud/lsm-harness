"""Model-visible tools plus a local side-effect policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


Effect = Literal["read", "local_write", "external_write"]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]
    effect: Effect = "read"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self, allowed_effects: set[Effect] | None = None):
        self._tools: dict[str, Tool] = {}
        self.allowed_effects = allowed_effects or {"read", "local_write"}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: unknown tool '{name}'"
        if tool.effect not in self.allowed_effects:
            return f"Error: tool '{name}' is blocked by the local side-effect policy"
        if "__parse_error__" in arguments:
            return f"Error: invalid JSON arguments ({arguments['__parse_error__']})"
        required = tool.input_schema.get("required", [])
        missing = [key for key in required if arguments.get(key) in (None, "")]
        if missing:
            return f"Error: missing required arguments: {', '.join(missing)}"
        try:
            return str(tool.fn(**arguments))
        except Exception as exc:
            return f"Error running {name}: {exc}"

