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
        self._otel = _OpenTelemetrySink.from_environment()

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
        if self._otel is not None:
            self._otel.write(event)

    def log_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost_total: float = 0.0,
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
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "total_tokens": (
                input_tokens
                + output_tokens
                + cache_read_tokens
                + cache_write_tokens
            ),
            "cost_total": cost_total,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock, self.usage_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def usage_summary(self) -> dict:
        """Aggregate token usage across all time."""
        total_in = 0
        total_out = 0
        total_cache_read = 0
        total_cache_write = 0
        total_cost = 0.0
        by_model: dict[str, dict] = {}
        if not self.usage_path.exists():
            return {
                "total_input": 0,
                "total_output": 0,
                "total_cache_read": 0,
                "total_cache_write": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "by_model": {},
            }
        for line in self.usage_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                inp = e.get("input_tokens", 0)
                out = e.get("output_tokens", 0)
                cache_read = e.get("cache_read_tokens", 0)
                cache_write = e.get("cache_write_tokens", 0)
                cost = float(e.get("cost_total", 0.0))
                model = e.get("model", "unknown")
                total_in += inp
                total_out += out
                total_cache_read += cache_read
                total_cache_write += cache_write
                total_cost += cost
                if model not in by_model:
                    by_model[model] = {
                        "input": 0,
                        "output": 0,
                        "cache_read": 0,
                        "cache_write": 0,
                        "cost": 0.0,
                        "calls": 0,
                    }
                by_model[model]["input"] += inp
                by_model[model]["output"] += out
                by_model[model]["cache_read"] += cache_read
                by_model[model]["cache_write"] += cache_write
                by_model[model]["cost"] += cost
                by_model[model]["calls"] += 1
            except json.JSONDecodeError:
                continue
        return {
            "total_input": total_in,
            "total_output": total_out,
            "total_cache_read": total_cache_read,
            "total_cache_write": total_cache_write,
            "total_tokens": (
                total_in + total_out + total_cache_read + total_cache_write
            ),
            "total_cost": total_cost,
            "by_model": by_model,
        }


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
            print("\nUsage totals:")
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


class _OpenTelemetrySink:
    """Optional projection of native Harness events to OpenTelemetry spans.

    The dependency is deliberately optional.  JSONL tracing is always active;
    setting ``OTEL_EXPORTER_OTLP_ENDPOINT`` enables this second sink when the
    OpenTelemetry packages are installed.
    """

    def __init__(self, tracer, provider, trace_api):
        self._tracer = tracer
        self._provider = provider
        self._trace_api = trace_api
        self._roots: dict[str, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls):
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        if not endpoint:
            return None
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            return None
        provider = TracerProvider(
            resource=Resource.create({"service.name": "lsm-harness"})
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, insecure=True)
            )
        )
        return cls(provider.get_tracer("lsm-harness"), provider, trace)

    def write(self, event: HarnessEvent) -> None:
        attributes = {
            "openinference.span.kind": event.span_kind,
            "lsm.trace_id": event.trace_id,
            "lsm.span_id": event.span_id,
            "lsm.parent_span_id": event.parent_span_id,
            "lsm.event_type": event.type,
            "lsm.sequence": event.sequence,
            "lsm.session_id": event.session_id,
            "lsm.data": json.dumps(event.data, ensure_ascii=False, default=str),
        }
        terminal = event.type in {
            "trace.completed", "trace.failed", "trace.aborted", "trace.error"
        }
        with self._lock:
            if event.type == "trace.started":
                self._roots[event.trace_id] = self._tracer.start_span(
                    "agent.run", attributes=attributes
                )
                return
            root = self._roots.get(event.trace_id)
            context = (
                self._trace_api.set_span_in_context(root) if root is not None else None
            )
            child = self._tracer.start_span(
                event.type, context=context, attributes=attributes
            )
            child.end()
            if terminal and root is not None:
                root.end()
                self._roots.pop(event.trace_id, None)
                self._provider.force_flush(timeout_millis=2000)
