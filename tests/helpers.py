from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Iterator

from lsm_harness.types import ModelResponse, StreamDelta, Usage


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
