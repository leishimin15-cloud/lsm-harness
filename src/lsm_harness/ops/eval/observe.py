"""Run-health aggregation for the Observe stage of the LLM-Ops loop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from lsm_harness.ops.eval.artifacts import EvalRunArtifacts


@dataclass
class RunObservation:
    run_id: str = ""
    total_ms: float = 0.0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    first_token_ms: float | None = None
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    retries: int = 0
    compactions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error_events: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def observe_artifacts(artifacts: "EvalRunArtifacts") -> RunObservation:
    events = _parse_jsonl(artifacts.trace_jsonl)
    run_id = artifacts.run_id
    if not run_id and events:
        run_id = str(events[0].get("trace_id") or events[0].get("turn_id") or "")

    llm_started: dict[Any, float] = {}
    tool_started: dict[str, float] = {}
    first_token_samples: list[float] = []
    first_token_iterations: set[Any] = set()
    llm_ms = tool_ms = 0.0
    retries = compactions = 0
    errors: list[dict[str, Any]] = []

    for event in events:
        event_type = str(event.get("type", ""))
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        elapsed = float(event.get("duration_ms") or 0.0)
        iteration = data.get("iteration", data.get("turn_index"))
        if event_type == "llm.started":
            llm_started[iteration] = elapsed
        elif event_type in {"llm.text.start", "llm.thinking.start"}:
            start = llm_started.get(iteration)
            if start is not None and iteration not in first_token_iterations:
                first_token_samples.append(max(0.0, elapsed - start))
                first_token_iterations.add(iteration)
        elif event_type == "llm.completed":
            start = llm_started.get(iteration)
            if start is not None:
                llm_ms += max(0.0, elapsed - start)
        elif event_type == "tool.started":
            tool_started[str(data.get("tool_call_id") or data.get("tool") or "")] = elapsed
        elif event_type == "tool.completed":
            key = str(data.get("tool_call_id") or data.get("tool") or "")
            if key in tool_started:
                tool_ms += max(0.0, elapsed - tool_started[key])

        if "retry" in event_type:
            retries += 1
        if "compaction" in event_type and event_type.endswith(("completed", "end")):
            compactions += 1
        if (
            event_type.endswith((".error", ".failed"))
            or bool(data.get("is_error"))
            or data.get("status") in {"error", "failed", "length_exhausted"}
        ):
            errors.append({"type": event_type, "data": data})

    usage = artifacts.usage
    input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in usage)
    output_tokens = sum(int(row.get("output_tokens", 0) or 0) for row in usage)
    cache_read = sum(int(row.get("cache_read_tokens", 0) or 0) for row in usage)
    cache_write = sum(int(row.get("cache_write_tokens", 0) or 0) for row in usage)
    total_tokens = sum(int(row.get("total_tokens", 0) or 0) for row in usage)
    if not total_tokens:
        total_tokens = input_tokens + output_tokens + cache_read + cache_write

    return RunObservation(
        run_id=run_id,
        total_ms=artifacts.duration_ms,
        llm_ms=llm_ms,
        tool_ms=tool_ms,
        first_token_ms=(sum(first_token_samples) / len(first_token_samples)
                        if first_token_samples else None),
        llm_calls=sum(1 for event in events if event.get("type") == "llm.completed"),
        tool_calls=len(artifacts.tool_calls),
        tool_errors=sum(1 for call in artifacts.tool_calls if call.get("is_error")),
        retries=retries,
        compactions=compactions,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        total_tokens=total_tokens,
        estimated_cost_usd=sum(float(row.get("cost_total", 0.0) or 0.0) for row in usage),
        error_events=errors,
    )


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events
