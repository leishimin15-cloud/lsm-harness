"""Model-visible tools plus a local side-effect policy.

Execution pipeline (each step can short-circuit):
  1. prepare_args(name, raw_args)           → validated & transformed dict
  2. validate(name, args)                   → check schema, policy, hooks
  3. execute(name, args, ctx)               → ToolResult (with timeout + abort)
  4. post_process(result)                   → format for the model

Tools opt into richer execution by declaring these keyword arguments:
  - _ctx: ExecutionContext    (abort signal, timeout, on_update callback)
  - _on_update: Callable      (older, simpler alternative)
  - _abort: AbortHandle       (standalone abort check)

Any tool that doesn't declare them just gets the plain **args — full
backward compatibility.
"""

from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


Effect = Literal["read", "local_write", "external_write"]

# ── hook signatures ──────────────────────────────────────────────

BeforeHook = Callable[[str, dict[str, Any]], str | None]
AfterHook = Callable[[str, dict[str, Any], "ToolResult"], "ToolResult"]

# ── structured result ────────────────────────────────────────────


@dataclass
class ToolResult:
    """The result of executing one tool.

    ``output`` is what the model sees.  ``details`` is structured
    metadata kept out of the model's context — it goes to the trace
    and the observer.
    """

    output: str
    is_error: bool = False
    terminate: bool = False
    details: dict[str, Any] | None = None

    # Backward-compat: ToolResult can be compared / formatted like a str
    def __str__(self) -> str:
        return self.output

    def __bool__(self) -> bool:
        return not self.is_error


# ── abort / execution context ─────────────────────────────────────


class AbortedError(Exception):
    """Raised (or used as a sentinel) when tool execution is cancelled."""


class AbortHandle:
    """Thread-safe handle that tools can poll during long operations.

    Usage inside a tool::

        def my_tool(large_file: str, _abort: AbortHandle | None = None) -> str:
            for chunk in read_chunks(large_file):
                if _abort and _abort.aborted:
                    return "Cancelled."
                process(chunk)
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def abort(self) -> None:
        self._event.set()

    def check(self) -> None:
        """Raise AbortedError if cancelled — for tools that prefer exceptions."""
        if self.aborted:
            raise AbortedError("tool execution cancelled")


@dataclass
class ExecutionContext:
    """Everything a tool might need beyond its declared arguments."""

    abort: AbortHandle = field(default_factory=AbortHandle)
    on_update: Callable[[str], None] = field(default_factory=lambda _msg: None)
    timeout: float = 0.0  # 0 = no timeout; set > 0 to run in a worker thread


# ── tool definition ──────────────────────────────────────────────


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]  # returns str | ToolResult
    effect: Effect = "read"
    # Lifecycle hooks (override registry-level defaults)
    before_hook: BeforeHook | None = None
    after_hook: AfterHook | None = None
    # If True, a successful result ends the agent loop
    terminate_on_success: bool = False
    # Parallel execution
    parallel_safe: bool | None = None  # None → auto from effect
    # Execution timeout in seconds (0 = no limit).  When > 0 the tool
    # runs in a worker thread so a hung tool doesn't block the loop.
    # Also settable via ExecutionContext.timeout at the call site.
    timeout: float = 0.0
    # Transform raw model-supplied arguments before validation.
    # Useful for normalising date strings, adding defaults, etc.
    prepare_args: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    @property
    def is_parallel_safe(self) -> bool:
        if self.parallel_safe is not None:
            return self.parallel_safe
        return self.effect == "read"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ── registry ──────────────────────────────────────────────────────


class ToolRegistry:
    def __init__(
        self,
        allowed_effects: set[Effect] | None = None,
        before_hook: BeforeHook | None = None,
        after_hook: AfterHook | None = None,
    ):
        self._tools: dict[str, Tool] = {}
        self.allowed_effects = allowed_effects or {"read", "local_write"}
        self.before_hook = before_hook
        self.after_hook = after_hook

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    # ── public API ────────────────────────────────────────────

    def execute_batch(
        self, calls: list[tuple[str, str, dict[str, Any]]], ctx: ExecutionContext | None = None
    ) -> list[tuple[int, str, ToolResult]]:
        """Execute multiple tool calls, parallel-safe ones concurrently.

        Returns results in the original call order.
        """
        # Split
        sequential: list[tuple[int, str, str, dict[str, Any]]] = []
        parallel: list[tuple[int, str, str, dict[str, Any]]] = []
        for idx, (call_id, name, args) in enumerate(calls):
            tool = self._tools.get(name)
            if tool and tool.is_parallel_safe:
                parallel.append((idx, call_id, name, args))
            else:
                sequential.append((idx, call_id, name, args))

        results: dict[int, tuple[str, ToolResult]] = {}

        # 1. Sequential
        for idx, call_id, name, args in sequential:
            results[idx] = self._execute_one(name, args, call_id, ctx)

        # 2. Parallel
        if parallel:
            with ThreadPoolExecutor(max_workers=min(len(parallel), 8)) as pool:
                futures = {
                    pool.submit(self._execute_one, name, args, call_id, ctx): (idx, call_id, name)
                    for idx, call_id, name, args in parallel
                }
                for future in as_completed(futures):
                    idx, call_id, name = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        results[idx] = (
                            call_id,
                            ToolResult(
                                output=f"Error running {name}: {exc}",
                                is_error=True,
                            ),
                        )

        return [(i, *results[i]) for i in range(len(calls))]

    def execute(
        self, name: str, arguments: dict[str, Any], ctx: ExecutionContext | None = None
    ) -> ToolResult:
        """Execute one tool through the full pipeline.

        Returns a ToolResult — never raises (errors become is_error=True).
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(output=f"Error: unknown tool '{name}'", is_error=True)

        # ── prepare_args ──
        try:
            if tool.prepare_args:
                arguments = tool.prepare_args(dict(arguments))
        except Exception as exc:
            return ToolResult(output=f"Error preparing arguments: {exc}", is_error=True)

        # ── validate ──
        err = self._validate(tool, arguments)
        if err:
            return ToolResult(output=err, is_error=True)

        # ── before hook ──
        before = tool.before_hook or self.before_hook
        if before:
            block = before(name, arguments)
            if block is not None:
                return ToolResult(output=f"Error: {block}", is_error=True)

        # ── execute (with timeout + abort) ──
        result = self._execute_with_timeout(tool, arguments, ctx)

        # ── after hook ──
        after = tool.after_hook or self.after_hook
        if after:
            result = after(name, arguments, result)

        # ── terminate check ──
        if tool.terminate_on_success and not result.is_error:
            result.terminate = True

        return result

    # ── internal ──────────────────────────────────────────────

    def _validate(self, tool: Tool, arguments: dict[str, Any]) -> str | None:
        if tool.effect not in self.allowed_effects:
            return f"Error: tool '{tool.name}' is blocked by the local side-effect policy"
        if "__parse_error__" in arguments:
            return f"Error: invalid JSON arguments ({arguments['__parse_error__']})"
        required = tool.input_schema.get("required", [])
        missing = [key for key in required if arguments.get(key) in (None, "")]
        if missing:
            return f"Error: missing required arguments: {', '.join(missing)}"
        return None

    def _execute_with_timeout(
        self, tool: Tool, arguments: dict[str, Any], ctx: ExecutionContext | None
    ) -> ToolResult:
        """Call tool.fn, respecting timeout and abort signal via ExecutionContext."""
        # ctx.timeout > 0 overrides; otherwise fall back to tool.timeout
        timeout = tool.timeout
        if ctx and ctx.timeout > 0:
            timeout = ctx.timeout

        # Build kwargs, passing ctx components the tool actually accepts
        kwargs = self._build_kwargs(tool, dict(arguments), ctx)

        # Fast path: no timeout, no abort → direct call
        if timeout <= 0 and (ctx is None or not ctx.abort.aborted):
            try:
                raw = tool.fn(**kwargs)
                return self._normalize(raw)
            except AbortedError:
                return ToolResult(output="工具执行被中断。", is_error=True, terminate=False)
            except Exception as exc:
                return ToolResult(output=f"Error running {tool.name}: {exc}", is_error=True)

        # Slow path: run in thread with timeout
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(self._call_with_abort, tool, kwargs, ctx)
            try:
                raw = future.result(timeout=timeout)
                return self._normalize(raw)
            except TimeoutError:
                future.cancel()
                if ctx:
                    ctx.abort.abort()
                return ToolResult(
                    output=f"Error: tool '{tool.name}' timed out after {timeout:.0f}s",
                    is_error=True,
                    details={"timeout": timeout},
                )
            except AbortedError:
                return ToolResult(output="工具执行被中断。", is_error=True)
            except Exception as exc:
                return ToolResult(output=f"Error running {tool.name}: {exc}", is_error=True)
        finally:
            pool.shutdown(wait=False)

    def _call_with_abort(
        self, tool: Tool, kwargs: dict[str, Any], ctx: ExecutionContext | None
    ) -> Any:
        """Called in a worker thread. Checks abort before invoking the tool."""
        if ctx and ctx.abort.aborted:
            raise AbortedError()
        try:
            return tool.fn(**kwargs)
        except AbortedError:
            raise
        except Exception:
            raise  # let _execute_with_timeout wrap it

    def _build_kwargs(
        self, tool: Tool, arguments: dict[str, Any], ctx: ExecutionContext | None
    ) -> dict[str, Any]:
        """Build fn(**kwargs), passing rich context only when the tool accepts it."""
        kwargs = dict(arguments)
        if ctx is None:
            return kwargs
        sig = inspect.signature(tool.fn)
        params = sig.parameters
        if "_ctx" in params:
            kwargs["_ctx"] = ctx
        else:
            if "_abort" in params:
                kwargs["_abort"] = ctx.abort
            if "_on_update" in params:
                kwargs["_on_update"] = ctx.on_update
        return kwargs

    def _execute_one(
        self, name: str, arguments: dict[str, Any], call_id: str, ctx: ExecutionContext | None = None
    ) -> tuple[str, ToolResult]:
        """Called by execute_batch. Returns (call_id, ToolResult)."""
        return call_id, self.execute(name, arguments, ctx)

    def _normalize(self, raw: Any) -> ToolResult:
        """Accept both old-style str returns and new ToolResult returns."""
        if isinstance(raw, ToolResult):
            return raw
        if isinstance(raw, str):
            is_error = raw.lower().startswith("error")
            return ToolResult(output=raw, is_error=is_error)
        return ToolResult(output=str(raw))
