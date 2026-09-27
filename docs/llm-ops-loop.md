# LSM Harness LLM-Ops Loop

The operations layer is outside the Agent Loop. It observes and evaluates a
run, but never changes model context or tool execution while the run is active.

```text
Harness run
  -> Trace (JSONL + optional OpenTelemetry)
  -> Eval outcome + Observe health metrics
  -> deterministic Diagnose report
  -> policy Gate
  -> Release Manifest
```

## Trace

Every `Harness.respond()` owns one `trace_id`. Events also expose portable
`span_id`, `parent_span_id`, and `span_kind` fields. JSONL is always enabled.
When `OTEL_EXPORTER_OTLP_ENDPOINT` is configured and the optional OpenTelemetry
packages are installed, the same events are exported under an `agent.run` root
span.

## Eval and Observe

Eval checks the task outcome: assertions, files, commands, session state, and
the final response. Observe measures run health: input/output/cache tokens,
estimated cost, total/model/tool latency, first-token latency, retries,
compactions, and tool errors. Both refer to the same run ID.

Each scenario artifact directory contains `result.json`, `trace.jsonl`,
`usage.json`, `observation.json`, and `diagnosis.json` alongside the existing
reply, tool-call, diff, and session artifacts.

## Diagnose

Diagnosis is deterministic first. It assigns failures to authentication,
provider calls, context overflow, schema validation, approval, tool execution,
verification commands, iteration limits, or final outcome assertions. A tool
error followed by a passing task is marked `recovered_tool_error` instead of an
unrecovered failure.

## Gate and release

Run an eval first, then gate its artifact directory:

```bash
lsm eval --offline
lsm eval gate --artifacts-dir .lsm/evals/<run-id>
```

The gate checks candidate correctness, completeness, unrecovered errors, and
paired token/latency/cost regressions. A failed gate writes `gate_report.json`
and exits non-zero. A passing gate also writes `release_manifest.json`, binding
the verdict to the evaluated Git revision and configuration hash. It does not
deploy or mutate production configuration.
