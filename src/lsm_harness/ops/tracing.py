"""Always-on, secret-free JSONL traces."""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path

from lsm_harness.events import HarnessEvent


class Tracer:
    def __init__(self, home: Path):
        self.path = home / "traces" / f"{date.today().isoformat()}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: HarnessEvent) -> None:
        line = json.dumps(event.as_dict(), ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
