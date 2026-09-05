from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Iterator

from lsm_harness.ai.types import ModelResponse, StreamDelta, Usage


def Tool(name, description="", input_schema=None, fn=None, effect="read",
         before_hook=None, after_hook=None, terminate_on_success=False,
         parallel_safe=None, timeout=0.0, prepare_args=None):
    """Test factory matching the deleted ``Tool()`` compat constructor.

    Returns a canonical ``AgentTool``; the old constructor's safety
    default is preserved (read tools parallel, effect tools sequential).
    """
    from lsm_harness.agent.tools import AgentTool

    if parallel_safe is not None:
        execution_mode = "parallel" if parallel_safe else "sequential"
    else:
        execution_mode = "parallel" if effect == "read" else "sequential"
    return AgentTool(
        name=name,
        description=description,
        parameters=input_schema or {"type": "object"},
        execute=fn,
        effect=effect,
        execution_mode=execution_mode,
        timeout=timeout,
        before_hook=before_hook,
        after_hook=after_hook,
        terminate_on_success=terminate_on_success,
        prepare_arguments=prepare_args,
    )


def run_test_loop(*, client=None, stream_fn=None, model="test-model",
                  system="system", messages=None, tools=None, emit=None,
                  max_iterations=10, max_tokens=100, interrupt=None,
                  **config_kw):
    """Drive ``run_agent_loop`` with the deleted flat ``run_loop()`` shape.

    Legacy dict messages are converted in place so callers keep observing
    the run's appends on their own list, exactly like the old API.
    """
    from lsm_harness.agent.agent_loop import run_agent_loop
    from lsm_harness.agent.types import AgentContext, AgentLoopConfig
    from lsm_harness.ai.stream import client_stream_function
    from lsm_harness.ai.types import Model
    from lsm_harness.ops.session_store import _message_from_dict

    if stream_fn is None:
        stream_fn = client_stream_function(client)
    # Dicts reuse the session-store deserialiser — the one remaining
    # dict → typed converter after the legacy adapters were deleted.
    typed = [
        _message_from_dict(m) if isinstance(m, dict) else m
        for m in (messages or [])
    ]
    if messages is not None:
        messages[:] = typed
        typed = messages
    context = AgentContext(
        system_prompt=system,
        messages=typed,
        tools=tools,
    )
    config = AgentLoopConfig(
        model=(
            model
            if isinstance(model, Model)
            else Model(id=model, api="legacy-client", provider="legacy")
        ),
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        **config_kw,
    )
    return run_agent_loop(
        context=context,
        config=config,
        stream_fn=stream_fn,
        emit=emit or (lambda kind, data: None),
        interrupt=interrupt,
    )


class QueueClient:
    """Scripted model client for deterministic tests.

    Supports both synchronous ``complete`` and ``stream_complete``.
    ``stream_complete`` converts each queued ``ModelResponse`` into
    the equivalent stream of ``StreamDelta`` events.
    """

    def __init__(self, *responses: ModelResponse):
        self.responses = deque(responses)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("scripted client ran out of responses")
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response

    def stream_complete(self, **kwargs) -> Iterator[StreamDelta]:
        """Simulate streaming by converting a ModelResponse into deltas."""
        self.calls.append(deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("scripted client ran out of responses")
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response

        # Emit text as a single delta (could be chunked but tests don't need it)
        if response.text:
            yield StreamDelta(kind="text_delta", text=response.text)

        # Emit tool calls
        for i, call in enumerate(response.tool_calls):
            import json
            args_str = json.dumps(call.arguments, ensure_ascii=False)
            yield StreamDelta(
                kind="tool_call_start",
                tool_index=i,
                tool_id=call.id,
                tool_name=call.name,
            )
            yield StreamDelta(
                kind="tool_call_delta",
                tool_index=i,
                tool_id=call.id,
                tool_name=call.name,
                arguments_delta=args_str,
            )

        # Done
        yield StreamDelta(
            kind="done",
            stop_reason=response.stop_reason,
            usage=response.usage,
        )
