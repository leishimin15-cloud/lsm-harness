from __future__ import annotations

from collections import deque
from copy import deepcopy


class QueueClient:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("scripted client ran out of responses")
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response
