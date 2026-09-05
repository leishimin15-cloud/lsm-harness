"""Always-on, secret-free JSONL traces + usage accounting."""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime
from pathlib import Path

from lsm_harness.events import HarnessEvent


class Tracer:
    def __init__(self, home: Path):
        self.home = home
        self._recompute_path()
        self._lock = threading.Lock()

    def _recompute_path(self) -> None:
        self.path = self.home / "traces" / f"{date.today().isoformat()}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def usage_path(self) -> Path:
        return self.home / "usage.jsonl"

    def write(self, event: HarnessEvent) -> None:
        line = json.dumps(event.as_dict(), ensure_ascii=False, default=str)
        with self._lock:
            # Recompute path in case date changed
            expected = self.home / "traces" / f"{date.today().isoformat()}.jsonl"
            if expected != self.path:
                self._recompute_path()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def log_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        turn_id: str = "",
    ) -> None:
        """Record token usage for cost tracking."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id[:8] if session_id else "",
            "turn_id": turn_id[:8] if turn_id else "",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock, self.usage_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def usage_summary(self) -> dict:
        """Aggregate token usage across all time."""
        total_in = 0
        total_out = 0
        by_model: dict[str, dict] = {}
        if not self.usage_path.exists():
            return {"total_input": 0, "total_output": 0, "by_model": {}}
        for line in self.usage_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                inp = e.get("input_tokens", 0)
                out = e.get("output_tokens", 0)
                model = e.get("model", "unknown")
                total_in += inp
                total_out += out
                if model not in by_model:
                    by_model[model] = {"input": 0, "output": 0, "calls": 0}
                by_model[model]["input"] += inp
                by_model[model]["output"] += out
                by_model[model]["calls"] += 1
            except json.JSONDecodeError:
                continue
        return {"total_input": total_in, "total_output": total_out, "by_model": by_model}


def list_recent_traces(home: str | None = None) -> None:
    """Print a summary of recent trace files (for `lsm traces`)."""
    base = Path(home or os.getenv("LSM_HOME", ".lsm")) / "traces"
    if not base.exists():
        print("No traces yet.")
        return

    files = sorted(base.glob("*.jsonl"), reverse=True)
    print(f"Traces ({len(files)} files in {base}):\n")

    for path in files[:10]:
        lines = 0
        try:
            for _ in path.open(encoding="utf-8"):
                lines += 1
        except Exception:
            pass
        size = path.stat().st_size
        print(f"  {path.name:40s}  {lines:>5} events  {_fmt_size(size)}")

    # Show usage summary
    usage_path = base.parent / "usage.jsonl"
    if usage_path.exists():
        tracer = Tracer(base.parent)
        summary = tracer.usage_summary()
        if summary["total_input"] > 0:
            print(f"\nUsage totals:")
            print(f"  input:  {summary['total_input']:>10,} tokens")
            print(f"  output: {summary['total_output']:>10,} tokens")
            for model, stats in summary.get("by_model", {}).items():
                print(f"  {model:30s}  {stats['calls']:>4} calls  "
                      f"in={stats['input']:>8,}  out={stats['output']:>8,}")


def _fmt_size(size: int) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
