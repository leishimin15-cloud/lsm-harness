"""Thread-safe pending-message queues owned by the stateful Agent."""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Literal

from lsm_harness.agent.messages import AgentMessage


QueueMode = Literal["one-at-a-time", "all"]
PendingMessage = str | AgentMessage


class PendingMessageQueue:
    """Pi-style queue that drains one message or the whole batch."""

    def __init__(
        self,
        mode: QueueMode = "one-at-a-time",
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.mode = mode
        self._on_change = on_change
        self._messages: deque[PendingMessage] = deque()
        self._lock = threading.Lock()

    def enqueue(self, message: PendingMessage) -> None:
        with self._lock:
            self._messages.append(message)
        self._notify_change()

    def has_items(self) -> bool:
        with self._lock:
            return bool(self._messages)

    def drain(self) -> list[PendingMessage]:
        with self._lock:
            if not self._messages:
                return []
            if self.mode == "all":
                drained = list(self._messages)
                self._messages.clear()
            else:
                drained = [self._messages.popleft()]
        self._notify_change()
        return drained

    def snapshot(self) -> list[PendingMessage]:
        with self._lock:
            return list(self._messages)

    def clear(self) -> list[PendingMessage]:
        with self._lock:
            drained = list(self._messages)
            self._messages.clear()
        if drained:
            self._notify_change()
        return drained

    def _notify_change(self) -> None:
        if self._on_change is not None:
            self._on_change()
