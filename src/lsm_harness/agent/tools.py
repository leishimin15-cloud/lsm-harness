"""Agent-layer tools and the Pi-style execution pipeline.

The AI layer owns only model-visible :class:`lsm_harness.ai.types.Tool`
descriptors.  This module owns executable :class:`AgentTool` objects and the
three stable pipeline stages used by the Agent Loop::

    prepare_tool_call
    -> execute_prepared_tool_call
    -> finalize_executed_tool_call

``Tool`` and ``ToolResult`` remain compatibility names for integrations that
used the pre-Chapter-5 API.
"""

from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

from jsonschema import Draft202012Validator

from lsm_harness.agent.events import (
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from lsm_harness.ai.types import Tool as AITool


Effect = Literal["read", "local_write", "external_write"]
ToolExecutionMode = Literal["sequential", "parallel"]


@dataclass
class ToolResultMessage:
    """Internal result envelope for one tool call.

    ``output`` and ``is_error`` cross the provider boundary. ``details`` is
    retained for Trace/UI only.  Call identity is populated by the registry.
    """

    output: str
    is_error: bool = False
    terminate: bool = False
    details: dict[str, Any] | None = None
    tool_call_id: str = ""
    tool_name: str = ""

    def __str__(self) -> str:
        return self.output

    def __bool__(self) -> bool:
        return not self.is_error


# Public compatibility name.  It is deliberately an alias so isinstance and
# type-based field override code keep working without warnings.
ToolResult = ToolResultMessage

BeforeHook = Callable[[str, dict[str, Any]], str | None]
AfterHook = Callable[[str, dict[str, Any], ToolResultMessage], ToolResultMessage]
LoopBeforeHook = Callable[[str, dict[str, Any], dict[str, Any]], str | None]
LoopAfterHook = Callable[
    [str, dict[str, Any], dict[str, Any], ToolResultMessage],
    ToolResultMessage,
]


class AbortedError(Exception):
    """Raised when cooperative tool execution is cancelled."""


class AbortHandle:
    """Thread-safe cancellation handle visible to long-running tools."""

    def __init__(self, external: threading.Event | None = None) -> None:
        self._event = threading.Event()
        self._external = external

    @property
    def aborted(self) -> bool:
        return bool(
            (self._external is not None and self._external.is_set())
            or self._event.is_set()
        )

    def abort(self) -> None:
        self._event.set()

    def check(self) -> None:
        if self.aborted:
            raise AbortedError("tool execution cancelled")


class CombinedAbortHandle(AbortHandle):
    """Per-tool cancellation layered on a shared parent handle.

    Refactor plan §9.2: ``abort()`` only trips THIS tool's local event
    (used for per-tool timeouts), so one tool timing out never pollutes
    its batch siblings.  ``aborted`` still reflects the parent handle —
    the global user interrupt aborts every tool in the batch.
    """

    def __init__(self, parent: AbortHandle) -> None:
        super().__init__()
        self._parent = parent

    @property
    def aborted(self) -> bool:
        return self._parent.aborted or self._event.is_set()


@dataclass
class ExecutionContext:
    """Runtime-only context passed to tools that opt into it."""

    abort: AbortHandle = field(default_factory=AbortHandle)
    on_update: Callable[[str], None] = field(default_factory=lambda: lambda _msg: None)
    timeout: float = 0.0
    turn_id: str = ""
    session_id: str = ""
    interrupt: threading.Event | None = None
    approval_broker: Any = None
    emit: Callable[[str, dict[str, Any]], None] | None = None
    # Chapter 7: when present, tool_execution_* kernel events go through
    # the typed sink; otherwise the legacy emit channel is used.
    event_sink: Any = None


@dataclass(frozen=True)
class AgentTool:
    """Executable Agent-layer tool.

    The default execution mode follows Pi and is parallel. Product-native
    tools still declare their mode explicitly at the ToolDefinition layer.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[..., Any]
    label: str = ""
    prepare_arguments: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    execution_mode: ToolExecutionMode = "parallel"
    effect: Effect = "read"
    timeout: float = 0.0
    before_hook: BeforeHook | None = None
    after_hook: AfterHook | None = None
    terminate_on_success: bool = False

    @property
    def display_label(self) -> str:
        return self.label or self.name

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.parameters

    @property
    def fn(self) -> Callable[..., Any]:
        return self.execute

    @property
    def prepare_args(self) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
        return self.prepare_arguments

    @property
    def is_parallel_safe(self) -> bool:
        return self.execution_mode == "parallel"

    @property
    def parallel_safe(self) -> bool:
        return self.is_parallel_safe

    def descriptor(self) -> AITool:
        return AITool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def schema(self) -> dict[str, Any]:
        """Legacy dictionary descriptor used by token estimation callers."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class Tool(AgentTool):
    """Backward-compatible constructor that maps onto :class:`AgentTool`."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any] | None = None,
        fn: Callable[..., Any] | None = None,
        effect: Effect = "read",
        before_hook: BeforeHook | None = None,
        after_hook: AfterHook | None = None,
        terminate_on_success: bool = False,
        parallel_safe: bool | None = None,
        timeout: float = 0.0,
        prepare_args: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        *,
        parameters: dict[str, Any] | None = None,
        execute: Callable[..., Any] | None = None,
        label: str = "",
        prepare_arguments: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        execution_mode: ToolExecutionMode | None = None,
    ) -> None:
        resolved_parameters = parameters if parameters is not None else input_schema
        resolved_execute = execute if execute is not None else fn
        if resolved_parameters is None:
            raise TypeError("Tool requires input_schema or parameters")
        if resolved_execute is None:
            raise TypeError("Tool requires fn or execute")
        if execution_mode is None:
            if parallel_safe is not None:
                execution_mode = "parallel" if parallel_safe else "sequential"
            else:
                # Preserve the old constructor's safety behaviour while the
                # new AgentTool itself defaults to parallel.
                execution_mode = "parallel" if effect == "read" else "sequential"
        super().__init__(
            name=name,
            description=description,
            parameters=resolved_parameters,
            execute=resolved_execute,
            label=label,
            prepare_arguments=(
                prepare_arguments if prepare_arguments is not None else prepare_args
            ),
            execution_mode=execution_mode,
            effect=effect,
            timeout=timeout,
            before_hook=before_hook,
            after_hook=after_hook,
            terminate_on_success=terminate_on_success,
        )


@dataclass(frozen=True)
class PreparedToolCall:
    call_id: str
    name: str
    tool: AgentTool
    arguments: dict[str, Any]
    raw_tool_call: dict[str, Any]


@dataclass(frozen=True)
class ExecutedToolCall:
    prepared: PreparedToolCall
    result: ToolResultMessage


class _ProgressGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepting_updates = True

    def close(self) -> None:
        with self._lock:
            self._accepting_updates = False

    def accept(self) -> bool:
        with self._lock:
            return self._accepting_updates


class ToolRegistry:
    def __init__(
        self,
        allowed_effects: set[Effect] | None = None,
        before_hook: BeforeHook | None = None,
        after_hook: AfterHook | None = None,
    ) -> None:
        self._tools: dict[str, AgentTool] = {}
        self.allowed_effects = allowed_effects or {"read", "local_write"}
        self.before_hook = before_hook
        self.after_hook = after_hook

    def register(self, tool: AgentTool) -> None:
        if not isinstance(tool, AgentTool):
            raise TypeError("ToolRegistry.register expects an AgentTool")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def filter(self, names: list[str]) -> "ToolRegistry":
        filtered = ToolRegistry(
            allowed_effects=set(self.allowed_effects),
            before_hook=self.before_hook,
            after_hook=self.after_hook,
        )
        for name in names:
            tool = self._tools.get(name)
            if tool is not None:
                filtered._tools[name] = tool
        return filtered

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def ai_tools(self) -> list[AITool]:
        return [tool.descriptor() for tool in self._tools.values()]

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute_batch(
        self,
        calls: list[tuple[str, str, dict[str, Any]]],
        ctx: ExecutionContext | None = None,
        *,
        mode: ToolExecutionMode = "parallel",
        before_tool_call: LoopBeforeHook | None = None,
        after_tool_call: LoopAfterHook | None = None,
        raw_calls: dict[str, dict[str, Any]] | None = None,
    ) -> list[tuple[int, str, ToolResultMessage]]:
        """Run sequential preflight, execute, then ordered finalization.

        Results are always returned in model-call order. In parallel mode only
        the ``execute`` stage runs concurrently; preflight and finalization are
        deterministic and sequential.
        """
        raw_calls = raw_calls or {}
        results: dict[int, tuple[str, ToolResultMessage]] = {}
        prepared: dict[int, PreparedToolCall] = {}

        for index, (call_id, name, arguments) in enumerate(calls):
            item, error = self.prepare_tool_call(
                name,
                arguments,
                call_id,
                ctx,
                tool_call=raw_calls.get(call_id),
                before_tool_call=before_tool_call,
            )
            if error is not None:
                results[index] = (call_id, error)
            elif item is not None:
                prepared[index] = item

        run_sequentially = mode == "sequential" or any(
            item.tool.execution_mode == "sequential" for item in prepared.values()
        )

        if run_sequentially:
            for index, item in prepared.items():
                executed = self.execute_prepared_tool_call(item, ctx)
                results[index] = (
                    item.call_id,
                    self.finalize_executed_tool_call(
                        executed,
                        after_tool_call=after_tool_call,
                    ),
                )
        elif prepared:
            executed_by_index: dict[int, ExecutedToolCall] = {}
            with ThreadPoolExecutor(max_workers=min(len(prepared), 8)) as pool:
                futures = {
                    pool.submit(self.execute_prepared_tool_call, item, ctx): index
                    for index, item in prepared.items()
                }
                for future in as_completed(futures):
                    index = futures[future]
                    item = prepared[index]
                    try:
                        executed_by_index[index] = future.result()
                    except Exception as exc:
                        # This is the last-resort boundary. Stage methods are
                        # expected to message-ize their own exceptions.
                        executed_by_index[index] = ExecutedToolCall(
                            prepared=item,
                            result=self._error_result(
                                item.call_id,
                                item.name,
                                "execute",
                                exc,
                            ),
                        )
            for index, item in prepared.items():
                results[index] = (
                    item.call_id,
                    self.finalize_executed_tool_call(
                        executed_by_index[index],
                        after_tool_call=after_tool_call,
                    ),
                )

        return [(index, *results[index]) for index in range(len(calls))]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        ctx: ExecutionContext | None = None,
        *,
        tool_call: dict[str, Any] | None = None,
        before_tool_call: LoopBeforeHook | None = None,
        after_tool_call: LoopAfterHook | None = None,
    ) -> ToolResultMessage:
        call_id = str((tool_call or {}).get("id", ""))
        prepared, error = self.prepare_tool_call(
            name,
            arguments,
            call_id,
            ctx,
            tool_call=tool_call,
            before_tool_call=before_tool_call,
        )
        if error is not None:
            return error
        if prepared is None:
            return self._error_result(call_id, name, "prepare", "unknown tool")
        return self.finalize_executed_tool_call(
            self.execute_prepared_tool_call(prepared, ctx),
            after_tool_call=after_tool_call,
        )

    def prepare_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        call_id: str,
        ctx: ExecutionContext | None,
        *,
        tool_call: dict[str, Any] | None,
        before_tool_call: LoopBeforeHook | None,
    ) -> tuple[PreparedToolCall | None, ToolResultMessage | None]:
        tool = self._tools.get(name)
        if tool is None:
            return None, ToolResultMessage(
                output=f"Error: unknown tool '{name}'",
                is_error=True,
                tool_call_id=call_id,
                tool_name=name,
            )

        try:
            prepared_arguments = dict(arguments)
            if tool.prepare_arguments is not None:
                prepared_arguments = tool.prepare_arguments(prepared_arguments)
            if not isinstance(prepared_arguments, dict):
                raise TypeError("prepare_arguments must return a dict")
        except Exception as exc:
            return None, ToolResultMessage(
                output=f"Error preparing arguments for '{name}': {exc}",
                is_error=True,
                tool_call_id=call_id,
                tool_name=name,
            )

        try:
            validation_error = self._validate(tool, prepared_arguments)
        except Exception as exc:
            return None, self._error_result(call_id, name, "validation", exc)
        if validation_error is not None:
            return None, ToolResultMessage(
                output=validation_error,
                is_error=True,
                tool_call_id=call_id,
                tool_name=name,
            )

        if tool.effect not in self.allowed_effects:
            return None, ToolResultMessage(
                output=(
                    f"Error: tool '{name}' is blocked by the local side-effect policy"
                ),
                is_error=True,
                tool_call_id=call_id,
                tool_name=name,
            )

        if ctx is not None and ctx.approval_broker is not None and tool.effect != "read":
            try:
                approved, reason = ctx.approval_broker.request(
                    turn_id=ctx.turn_id,
                    session_id=ctx.session_id,
                    tool_name=tool.name,
                    effect=tool.effect,
                    arguments=prepared_arguments,
                    emit=ctx.emit,
                    interrupt=ctx.interrupt,
                )
            except Exception as exc:
                return None, self._error_result(call_id, name, "approval", exc)
            if not approved:
                return None, ToolResultMessage(
                    output=f"Error: tool '{name}' was not approved ({reason}).",
                    is_error=True,
                    details={"approval": "rejected", "reason": reason},
                    tool_call_id=call_id,
                    tool_name=name,
                )

        raw_tool_call = tool_call or {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": prepared_arguments},
        }

        if before_tool_call is not None:
            try:
                block_reason = before_tool_call(name, prepared_arguments, raw_tool_call)
            except Exception as exc:
                return None, self._error_result(call_id, name, "beforeToolCall", exc)
            if block_reason is not None:
                return None, ToolResultMessage(
                    output=f"Error: {block_reason}",
                    is_error=True,
                    tool_call_id=call_id,
                    tool_name=name,
                )

        before = tool.before_hook or self.before_hook
        if before is not None:
            try:
                block_reason = before(name, prepared_arguments)
            except Exception as exc:
                return None, self._error_result(call_id, name, "tool before hook", exc)
            if block_reason is not None:
                return None, ToolResultMessage(
                    output=f"Error: {block_reason}",
                    is_error=True,
                    tool_call_id=call_id,
                    tool_name=name,
                )

        return PreparedToolCall(
            call_id=call_id,
            name=name,
            tool=tool,
            arguments=prepared_arguments,
            raw_tool_call=raw_tool_call,
        ), None

    def execute_prepared_tool_call(
        self,
        prepared: PreparedToolCall,
        ctx: ExecutionContext | None,
    ) -> ExecutedToolCall:
        tool = prepared.tool
        gate = _ProgressGate()
        call_ctx = self._context_for_call(ctx, prepared, gate)
        if call_ctx.event_sink is not None:
            call_ctx.event_sink.process_event(ToolExecutionStartEvent(
                tool_call_id=prepared.call_id,
                tool_name=tool.name,
                label=tool.display_label,
                effect=tool.effect,
                args=prepared.arguments,
            ))
        elif call_ctx.emit is not None:
            call_ctx.emit("tool.started", {
                "tool": tool.name,
                "label": tool.display_label,
                "tool_call_id": prepared.call_id,
                "effect": tool.effect,
            })
        try:
            result = self._execute_with_timeout(tool, prepared.arguments, call_ctx)
        except Exception as exc:
            result = self._error_result(
                prepared.call_id,
                prepared.name,
                "execute",
                exc,
            )
        finally:
            # Closing immediately after settle prevents detached/background
            # workers from emitting stale progress into a later Turn.
            gate.close()
        result = self._with_identity(result, prepared.call_id, prepared.name)
        if call_ctx.event_sink is not None:
            call_ctx.event_sink.process_event(ToolExecutionEndEvent(
                tool_call_id=prepared.call_id,
                tool_name=tool.name,
                label=tool.display_label,
                is_error=result.is_error,
            ))
        elif call_ctx.emit is not None:
            call_ctx.emit("tool.execution_end", {
                "tool": tool.name,
                "label": tool.display_label,
                "tool_call_id": prepared.call_id,
                "is_error": result.is_error,
            })
        return ExecutedToolCall(prepared=prepared, result=result)

    def finalize_executed_tool_call(
        self,
        executed: ExecutedToolCall,
        *,
        after_tool_call: LoopAfterHook | None,
    ) -> ToolResultMessage:
        prepared = executed.prepared
        tool = prepared.tool
        result = executed.result

        after = tool.after_hook or self.after_hook
        if after is not None:
            try:
                result = self._normalize(after(prepared.name, prepared.arguments, result))
                result = self._with_identity(result, prepared.call_id, prepared.name)
            except Exception as exc:
                return self._error_result(
                    prepared.call_id,
                    prepared.name,
                    "tool after hook",
                    exc,
                )

        if tool.terminate_on_success and not result.is_error:
            result = replace(result, terminate=True)

        if after_tool_call is not None:
            try:
                result = self._normalize(after_tool_call(
                    prepared.name,
                    prepared.arguments,
                    prepared.raw_tool_call,
                    result,
                ))
                result = self._with_identity(result, prepared.call_id, prepared.name)
            except Exception as exc:
                return self._error_result(
                    prepared.call_id,
                    prepared.name,
                    "afterToolCall",
                    exc,
                )
        return result

    def _validate(self, tool: AgentTool, arguments: dict[str, Any]) -> str | None:
        if "__parse_error__" in arguments:
            return (
                f"Error: invalid arguments for '{tool.name}': $: invalid JSON "
                f"({arguments['__parse_error__']})"
            )
        errors = sorted(
            Draft202012Validator(tool.parameters).iter_errors(arguments),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if not errors:
            return None
        error = errors[0]
        path = "$"
        for part in error.absolute_path:
            path += f"[{part}]" if isinstance(part, int) else f".{part}"
        return f"Error: invalid arguments for '{tool.name}': {path}: {error.message}"

    def _context_for_call(
        self,
        ctx: ExecutionContext | None,
        prepared: PreparedToolCall,
        gate: _ProgressGate,
    ) -> ExecutionContext:
        base = ctx or ExecutionContext()

        def on_update(message: str) -> None:
            if not gate.accept():
                return
            base.on_update(message)
            if base.event_sink is not None:
                base.event_sink.process_event(ToolExecutionUpdateEvent(
                    tool_call_id=prepared.call_id,
                    tool_name=prepared.name,
                    label=prepared.tool.display_label,
                    partial=message,
                ))
            elif base.emit is not None:
                base.emit("tool.progress", {
                    "tool": prepared.name,
                    "label": prepared.tool.display_label,
                    "tool_call_id": prepared.call_id,
                    "delta": message,
                })

        # §9.2: each call gets its own CombinedAbortHandle layered on the
        # batch-wide handle, so a per-tool timeout abort (which calls
        # ``ctx.abort.abort()``) only cancels THIS tool.
        return replace(
            base,
            abort=CombinedAbortHandle(base.abort),
            on_update=on_update,
        )

    def _execute_with_timeout(
        self,
        tool: AgentTool,
        arguments: dict[str, Any],
        ctx: ExecutionContext,
    ) -> ToolResultMessage:
        timeout = ctx.timeout if ctx.timeout > 0 else tool.timeout
        kwargs = self._build_kwargs(tool, arguments, ctx)
        if ctx.abort.aborted:
            return ToolResultMessage(output="工具执行被中断。", is_error=True)
        if timeout <= 0:
            try:
                return self._normalize(tool.execute(**kwargs))
            except AbortedError:
                return ToolResultMessage(output="工具执行被中断。", is_error=True)
            except Exception as exc:
                return self._error_result("", tool.name, "execute", exc)

        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self._call_with_abort, tool, kwargs, ctx)
        try:
            return self._normalize(future.result(timeout=timeout))
        except TimeoutError:
            future.cancel()
            ctx.abort.abort()
            return ToolResultMessage(
                output=f"Error: tool '{tool.name}' timed out after {timeout:g}s",
                is_error=True,
                details={"timeout": timeout},
            )
        except AbortedError:
            return ToolResultMessage(output="工具执行被中断。", is_error=True)
        except Exception as exc:
            return self._error_result("", tool.name, "execute", exc)
        finally:
            pool.shutdown(wait=False)

    @staticmethod
    def _call_with_abort(
        tool: AgentTool,
        kwargs: dict[str, Any],
        ctx: ExecutionContext,
    ) -> Any:
        ctx.abort.check()
        return tool.execute(**kwargs)

    @staticmethod
    def _build_kwargs(
        tool: AgentTool,
        arguments: dict[str, Any],
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        kwargs = dict(arguments)
        params = inspect.signature(tool.execute).parameters
        if "_ctx" in params:
            kwargs["_ctx"] = ctx
        else:
            if "_abort" in params:
                kwargs["_abort"] = ctx.abort
            if "_on_update" in params:
                kwargs["_on_update"] = ctx.on_update
        return kwargs

    @staticmethod
    def _normalize(raw: Any) -> ToolResultMessage:
        if isinstance(raw, ToolResultMessage):
            return raw
        if isinstance(raw, str):
            return ToolResultMessage(
                output=raw,
                is_error=raw.lower().startswith("error"),
            )
        return ToolResultMessage(output=str(raw))

    @staticmethod
    def _with_identity(
        result: ToolResultMessage,
        call_id: str,
        name: str,
    ) -> ToolResultMessage:
        return replace(
            result,
            tool_call_id=result.tool_call_id or call_id,
            tool_name=result.tool_name or name,
        )

    @staticmethod
    def _error_result(
        call_id: str,
        name: str,
        stage: str,
        error: object,
    ) -> ToolResultMessage:
        return ToolResultMessage(
            output=f"Error during {stage} for '{name}': {error}",
            is_error=True,
            tool_call_id=call_id,
            tool_name=name,
        )


def prepare_tool_call(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, Any],
    call_id: str = "",
    ctx: ExecutionContext | None = None,
    *,
    tool_call: dict[str, Any] | None = None,
    before_tool_call: LoopBeforeHook | None = None,
) -> tuple[PreparedToolCall | None, ToolResultMessage | None]:
    """Functional entry point for the first stable pipeline stage."""
    return registry.prepare_tool_call(
        name,
        arguments,
        call_id,
        ctx,
        tool_call=tool_call,
        before_tool_call=before_tool_call,
    )


def execute_prepared_tool_call(
    registry: ToolRegistry,
    prepared: PreparedToolCall,
    ctx: ExecutionContext | None = None,
) -> ExecutedToolCall:
    """Functional entry point for the parallelizable execution stage."""
    return registry.execute_prepared_tool_call(prepared, ctx)


def finalize_executed_tool_call(
    registry: ToolRegistry,
    executed: ExecutedToolCall,
    *,
    after_tool_call: LoopAfterHook | None = None,
) -> ToolResultMessage:
    """Functional entry point for deterministic post-processing."""
    return registry.finalize_executed_tool_call(
        executed,
        after_tool_call=after_tool_call,
    )


__all__ = [
    "AbortedError",
    "AbortHandle",
    "AgentTool",
    "CombinedAbortHandle",
    "Effect",
    "ExecutedToolCall",
    "ExecutionContext",
    "PreparedToolCall",
    "Tool",
    "ToolExecutionMode",
    "ToolRegistry",
    "ToolResult",
    "ToolResultMessage",
    "execute_prepared_tool_call",
    "finalize_executed_tool_call",
    "prepare_tool_call",
]
