"""Scenario runner: execute an :class:`EvalScenario` in an isolated workspace.

The runner owns the whole lifecycle:
  1. materialize the fixture into a fresh temporary workspace,
  2. create a fresh temporary ``home`` (sessions / traces / usage live there),
  3. pass the workspace as ``workspace_root`` to ``CodingSession`` — no
     process-wide ``os.chdir``, so runs can be parallelized,
  4. build a scripted (or injected) model client + harness,
  5. run each step, collecting tool calls, replies, command outcomes,
  6. snapshot the workspace before/after and compute the diff,
  7. evaluate assertions, collect artifacts, and restore cwd.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsm_harness.ai.api.common import snapshot
from lsm_harness.ai.messages import ToolResultMessage
from lsm_harness.ai.types import (
    AssistantMessageEvent,
    ModelClient,
    ModelResponse,
    StreamFunction,
    Usage,
)
from lsm_harness.agent.events import (
    MessageEndEvent,
    ToolExecutionStartEvent,
)
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.ops.session_store import path_to_leaf, read_session_entries

from lsm_harness.ops.eval._compat import _client_stream_fn
from lsm_harness.ops.eval.artifacts import (
    EvalRunArtifacts,
    diff_snapshots,
    snapshot_workspace,
    unified_patch,
)
from lsm_harness.ops.eval.assertions import (
    Assertion,
    AssertionContext,
    CommandOutcome,
    evaluate,
)
from lsm_harness.ops.eval.fixture import new_workspace
from lsm_harness.ops.eval.judge import DeterministicJudge, JudgeVerdict, ModelJudge
from lsm_harness.ops.eval.scenario import EvalScenario, EvalStep
from lsm_harness.ops.eval.variant import (
    EvalVariant,
    apply_tool_policy,
    apply_variant_settings,
)


@dataclass
class StepResult:
    kind: str
    text: str = ""
    status: str = "ok"  # ok | aborted | failed | parked | skipped | error
    reply: str = ""
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class ScenarioResult:
    name: str
    passed: bool = False
    duration_ms: float = 0.0
    step_results: list[StepResult] = field(default_factory=list)
    assertion_results: list[tuple[Assertion, bool, str]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    final_reply: str = ""
    judge_verdict: JudgeVerdict | None = None
    artifacts: EvalRunArtifacts | None = None
    # 本会话累计 token/成本(tracer.usage_summary()),供成对指标报告。
    usage: dict[str, Any] = field(default_factory=dict)
    # LLM 调用次数(≈ agent loop 迭代数),从 usage by_model.calls 聚合。
    iterations: int = 0

    @property
    def tool_error_count(self) -> int:
        return sum(1 for call in self.tool_calls if call.get("is_error"))


class ScenarioClient:
    """Scripted model client for one scenario.

    Normal model calls pop from ``responses`` in order.  A compaction
    summarizer call (``tools == []``) returns ``summary_text`` — there is no
    fixed number of summarizer calls, so detecting them by their empty tool
    list is what makes compaction scenarios deterministic.
    """

    def __init__(self, responses: list[ModelResponse], summary_text: str = ""):
        self.responses: deque[ModelResponse] = deque(responses)
        self.summary_text = summary_text or "（已压缩早期对话，目标保持不变。）"
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(deepcopy(kwargs))
        # compaction summarizer 的判别:无 tools **且无 system**。
        # 普通模型调用总有非空 system(build_system 产出 persona + 时间 +
        # skills listing),即使 variant 禁用了全部工具;只看 tools 会把
        # 「零工具变体」的普通调用误判成 summarizer。
        if not kwargs.get("tools") and not kwargs.get("system"):
            return ModelResponse(
                text=self.summary_text, stop_reason="stop", usage=Usage(100, 20)
            )
        if not self.responses:
            raise AssertionError("scripted client ran out of responses")
        return self.responses.popleft()

    def as_stream_fn(self) -> StreamFunction:
        return _client_stream_fn(self)


def _parked_stream(entered: threading.Event) -> StreamFunction:
    """Block inside the stream until aborted, then end with an aborted error
    event — the deterministic "park" used by the abort scenario (mirrors
    ``tests/test_continue.py``)."""

    def stream(_model, _context, options):
        entered.set()
        while not (options.interrupt and options.interrupt.is_set()):
            time.sleep(0.005)
        yield AssistantMessageEvent(
            "error",
            snapshot(text="", thinking="", pending={}, stop_reason="aborted"),
            error_category="aborted",
        )

    return stream


def _make_typed_tool_collector(tool_calls: list[dict[str, Any]]):
    """Collect tool calls from the typed CodingSession event stream.

    ToolExecutionStartEvent opens a record (name + args); the following
    MessageEndEvent(source="tool") fills output / is_error by tool_call_id.
    This replaces the legacy ``HarnessEvent`` observer.
    """

    def listener(event) -> None:
        if isinstance(event, ToolExecutionStartEvent):
            tool_calls.append({
                "tool": event.tool_name,
                "args": dict(event.args or {}),
                "output": "",
                "is_error": False,
                "_call_id": event.tool_call_id,
            })
        elif (
            isinstance(event, MessageEndEvent)
            and event.source == "tool"
            and isinstance(event.message, ToolResultMessage)
        ):
            message = event.message
            for call in reversed(tool_calls):
                if call.get("_call_id") == message.tool_call_id:
                    call["output"] = message.content or ""
                    call["is_error"] = bool(message.is_error)
                    call.pop("_call_id", None)
                    break

    return listener


def _build_harness(
    settings: Settings,
    client: ModelClient,
    stream_fn: StreamFunction,
    workspace_root: Path | None = None,
) -> Harness:
    # check_same_thread=False: the abort scenario runs respond() on a worker
    # thread, so the SQLite connection must be usable across threads.
    conn = connect(settings.home, check_same_thread=False)
    return Harness(
        settings=settings,
        client=client,
        stream_fn=stream_fn,
        conn=conn,
        workspace_root=workspace_root,
        # 密封:eval 只扫临时 home 的项目级 skills,不加载真实
        # ~/.lsm/skills——否则 eval 结果随用户本机 skill 变化。
        skill_directories=[(settings.home / "skills", "project")],
    )


def _msg_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def _session_snapshot(harness: Harness):
    path = harness.session.jsonl_path
    entries = read_session_entries(path) if path.exists() else []
    leaf = harness.session._last_entry_id
    tree = path_to_leaf(entries, leaf)
    return entries, tree, path


def run_scenario(
    scenario: EvalScenario,
    *,
    artifacts_dir: str | Path | None = None,
    judge=None,
    judge_client: ModelClient | None = None,
    client: ModelClient | None = None,
    stream_fn: StreamFunction | None = None,
    variant: EvalVariant | None = None,
) -> ScenarioResult:
    """Run one scenario in isolation; return a :class:`ScenarioResult`.

    ``artifacts_dir`` is the *exact* directory the artifact bundle is written
    into (the caller controls nesting, e.g. ``<scenario>/<variant>/<rep>``).

    ``variant`` (optional) overrides provider/model/system-prompt/settings/
    tools for the run; it is applied after the scenario's own overrides, so
    it always wins.
    """
    if judge is None and judge_client is not None:
        judge = ModelJudge(judge_client)

    workspace = new_workspace(scenario.fixture)
    home = Path(tempfile.mkdtemp(prefix="lsm-eval-home-"))
    # revision 记录的是**被评测代码**(本模块所在仓库),不是调用者 cwd——
    # 在其他目录跑已安装的 lsm 时不能记错仓库;包目录无 .git 时回退 cwd。
    git_root = _find_git_root(Path(__file__).resolve()) or _find_git_root(
        Path.cwd()
    ) or Path.cwd()
    git_revision = _git_revision(git_root)

    try:
        result = _run_in_workspace(
            scenario, workspace, home,
            artifacts_dir=artifacts_dir,
            judge=judge,
            client=client,
            stream_fn=stream_fn,
            variant=variant,
            git_revision=git_revision,
        )
    except (Exception, SystemExit) as exc:  # setup/构建失败也是一次失败的 run
        # (providers 对未知 provider 抛 SystemExit,不是 Exception)
        result = ScenarioResult(
            name=scenario.name,
            passed=False,
            failures=[f"setup: {type(exc).__name__}: {exc}"],
        )
    finally:
        shutil.rmtree(home, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)

    return result


def _run_in_workspace(
    scenario: EvalScenario,
    workspace: Path,
    home: Path,
    *,
    artifacts_dir: str | Path | None,
    judge,
    client: ModelClient | None,
    stream_fn: StreamFunction | None,
    variant: EvalVariant | None,
    git_revision: str = "",
) -> ScenarioResult:
    settings = Settings(
        home=home,
        max_iterations=scenario.max_iterations,
    )
    if not scenario.use_real_api:
        # 脚本场景:占位 key;真实场景交给环境/variant 的 key 解析。
        settings.api_key = "scripted"
    for key, value in scenario.settings_overrides.items():
        setattr(settings, key, value)
    if variant is not None:
        apply_variant_settings(settings, variant)
    settings.ensure_home()

    before = snapshot_workspace(workspace)

    use_real = scenario.use_real_api
    if use_real:
        # 凭证按 variant 的 provider 重新解析:eval 用临时 home(密封
        # sessions/traces),但 /login 存的 key 在真实 home 的 auth.json。
        # 关键陷阱:Settings 默认从 .env 解析出 api_key(如 DEEPSEEK_API_KEY),
        # variant 把 provider 切到 kimi-coding 后旧 key 不会自动换——直接把
        # A provider 的 key 发给 B provider 就是 401。scenario 显式覆盖
        # api_key 或注入 client 时尊重调用方。
        if client is None and "api_key" not in scenario.settings_overrides:
            from lsm_harness.coding_agent.startup import (
                resolve_product_api_key,
            )
            resolved_key = resolve_product_api_key(
                settings.provider, home=Settings().home
            )
            if resolved_key:
                settings.api_key = resolved_key
        # 真实场景:client/stream_fn 可注入(测试);缺省时 Harness 的
        # ModelRuntime 从 settings(variant 的 provider/model + 环境 key)
        # 解析真实三元组——这条路径让 CLI 能直接跑真实模型 A/B。
        scenario_client = None
        model_client = client
        base_stream_fn = stream_fn
    else:
        scenario_client = ScenarioClient(
            scenario.scripted_responses, scenario.summary_text
        )
        model_client = scenario_client
        base_stream_fn = scenario_client.as_stream_fn()

    harness = _build_harness(settings, model_client, base_stream_fn, workspace_root=workspace)
    if variant is not None and variant.tool_policy is not None:
        apply_tool_policy(harness, variant.tool_policy)

    step_results: list[StepResult] = []
    replies: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    command_outcomes: list[CommandOutcome] = []
    system_prompts: list[str] = []
    listener = _make_typed_tool_collector(tool_calls)
    unsubscribe = harness.subscribe(listener)

    parked: dict[str, Any] = {"thread": None, "entered": threading.Event(), "result": []}

    started = time.monotonic()

    try:
        for step in scenario.steps:
            if step.kind == "reload":
                unsubscribe()
                harness.close()
                harness = _build_harness(settings, model_client, base_stream_fn, workspace_root=workspace)
                if variant is not None and variant.tool_policy is not None:
                    apply_tool_policy(harness, variant.tool_policy)
                unsubscribe = harness.subscribe(listener)
                step_results.append(StepResult(kind="reload", status="ok"))
                continue

            sr = _execute_step(
                step, harness, scenario, parked, workspace, command_outcomes
            )
            step_results.append(sr)
            if step.kind in ("prompt", "continue"):
                replies.append(sr.reply)
                # 每 run 重建的 system prompt(含 project memory 注入),
                # 供 system_prompt_contains 断言检查。
                prompt_text = getattr(
                    getattr(harness, "agent", None), "state", None
                )
                prompt_text = getattr(prompt_text, "system_prompt", "")
                if prompt_text:
                    system_prompts.append(prompt_text)

        final_reply = replies[-1] if replies else ""

        entries, tree, session_path = _session_snapshot(harness)
        user_messages = [
            _msg_text(e.message)
            for e in tree
            if getattr(e, "type", None) == "message"
            and getattr(getattr(e, "message", None), "role", None) == "user"
        ]
        try:
            compaction_summary = harness.session.summary()
        except Exception:
            compaction_summary = ""

        # trace_not_contains 断言的检查对象:trace 事件流 + 会话 JSONL
        # (密钥/敏感串两处都不该出现)。
        trace_parts = [session_path.read_text(encoding="utf-8")] if session_path.exists() else []
        traces_dir = home / "traces"
        if traces_dir.exists():
            for trace_file in sorted(traces_dir.glob("*.jsonl")):
                trace_parts.append(trace_file.read_text(encoding="utf-8"))
        trace_text = "".join(trace_parts)

        elapsed_ms = (time.monotonic() - started) * 1000

        after = snapshot_workspace(workspace)
        changed_files = diff_snapshots(before, after)
        patch = unified_patch(before, after)

        ctx = AssertionContext(
            tool_calls=tool_calls,
            replies=replies,
            final_reply=final_reply,
            workspace=workspace,
            session_entries=entries,
            session_tree=tree,
            compaction_summary=compaction_summary,
            command_outcomes=command_outcomes,
            user_messages=user_messages,
            trace_text=trace_text,
            system_prompts=system_prompts,
        )

        the_judge = judge or DeterministicJudge()
        verdict = the_judge.judge(
            scenario.name, final_reply, scenario.assertions, ctx
        )

        assertion_results: list[tuple[Assertion, bool, str]] = []
        for assertion in scenario.assertions:
            passed, message = evaluate(assertion, ctx)
            assertion_results.append((assertion, passed, message))

        failures = [
            f"{a.describe()}: {msg}" for a, passed, msg in assertion_results if not passed
        ]
        if not verdict.passed:
            failures.append(f"judge: {verdict.reason}")

        # A failed or errored step fails the run even if no assertion fires.
        for step_result in step_results:
            if step_result.status in ("failed", "error"):
                failures.append(
                    f"step {step_result.kind}: "
                    f"{step_result.error or step_result.reply or 'failed'}"
                )

        passed = len(failures) == 0

        try:
            usage_summary = harness.tracer.usage_summary()
        except Exception:
            usage_summary = {}
        iterations = sum(
            int(model.get("calls", 0))
            for model in usage_summary.get("by_model", {}).values()
        )

        artifacts = _collect_artifacts(
            scenario, workspace, home, session_path,
            final_reply, tool_calls, changed_files, patch,
            scenario_client if not use_real else None,
            settings=settings,
            git_revision=git_revision,
            status="ok" if passed else "failed",
            duration_ms=elapsed_ms,
            failures=list(failures),
        )
        if artifacts_dir is not None:
            artifacts.write(Path(artifacts_dir))

        return ScenarioResult(
            name=scenario.name,
            passed=passed,
            duration_ms=elapsed_ms,
            step_results=step_results,
            assertion_results=assertion_results,
            failures=failures,
            tool_calls=tool_calls,
            replies=replies,
            final_reply=final_reply,
            judge_verdict=verdict,
            artifacts=artifacts,
            usage=usage_summary,
            iterations=iterations,
        )
    finally:
        unsubscribe()
        harness.close()


def _run_prompt(
    harness: Harness,
    text: str,
    timeout: float,
    started: float,
    scenario: EvalScenario,
) -> StepResult:
    """Run a non-parked prompt, optionally under a wall-clock timeout."""
    if timeout and timeout > 0:
        holder: list[Any] = []
        thread = threading.Thread(
            target=lambda: holder.append(harness.respond(text, source="eval", approval_broker=scenario.approval_broker)),
            daemon=True,
        )
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            harness.abort()
            thread.join(5)
            return StepResult(kind="prompt", text=text, status="error",
                              error=f"prompt timed out after {timeout}s",
                              duration_ms=(time.monotonic() - started) * 1000)
        result = holder[0] if holder else None
    else:
        result = harness.respond(text, source="eval", approval_broker=scenario.approval_broker)
    if result is None:
        return StepResult(kind="prompt", text=text, status="error",
                          error="prompt returned no result",
                          duration_ms=(time.monotonic() - started) * 1000)
    return StepResult(kind="prompt", text=text, status=result.status,
                      reply=result.reply,
                      duration_ms=(time.monotonic() - started) * 1000)


def _execute_step(
    step: EvalStep,
    harness: Harness,
    scenario: EvalScenario,
    parked: dict[str, Any],
    workspace: Path,
    command_outcomes: list[CommandOutcome],
) -> StepResult:
    started = time.monotonic()
    try:
        if step.kind == "prompt":
            if step.park:
                entered: threading.Event = parked["entered"]
                entered.clear()
                harness.stream_fn = _parked_stream(entered)
                holder: list[Any] = []
                thread = threading.Thread(
                    target=lambda: holder.append(
                        harness.respond(step.text, source="eval", approval_broker=scenario.approval_broker)
                    ),
                    daemon=True,
                )
                parked["thread"] = thread
                parked["result"] = holder
                thread.start()
                if not entered.wait(10):
                    raise TimeoutError("parked stream never entered the model call")
                return StepResult(kind="prompt", text=step.text, status="parked",
                                  duration_ms=(time.monotonic() - started) * 1000)

            # 非 park prompt:可选超时(整个 step 的墙钟上限)。
            result = _run_prompt(harness, step.text, step.timeout, started, scenario)
            return result

        if step.kind == "abort":
            thread = parked["thread"]
            if thread is None or not thread.is_alive():
                return StepResult(kind="abort", status="error",
                                  error="no parked run to abort")
            harness.abort()
            thread.join(10)
            result = parked["result"][0] if parked["result"] else None
            parked["thread"] = None
            status = result.status if result is not None else "aborted"
            reply = result.reply if result is not None else ""
            return StepResult(kind="abort", status=status, reply=reply,
                              duration_ms=(time.monotonic() - started) * 1000)

        if step.kind == "continue":
            result = harness.respond_continue(source="eval", approval_broker=scenario.approval_broker)
            return StepResult(kind="continue", status=result.status, reply=result.reply,
                              duration_ms=(time.monotonic() - started) * 1000)

        if step.kind == "compact":
            ok = harness.compact()
            return StepResult(kind="compact", status="ok" if ok else "skipped",
                              reply=f"compacted={ok}",
                              duration_ms=(time.monotonic() - started) * 1000)

        if step.kind == "command":
            cwd = (workspace / step.cwd) if step.cwd else workspace
            proc = subprocess.run(
                step.text, shell=True, cwd=cwd, capture_output=True, text=True,
                timeout=step.timeout or 60,
            )
            command_outcomes.append(CommandOutcome(
                command=step.text, exit_code=proc.returncode,
                stdout=proc.stdout, stderr=proc.stderr,
            ))
            return StepResult(kind="command", text=step.text,
                              status="ok" if proc.returncode == 0 else "failed",
                              reply=proc.stdout[:500],
                              duration_ms=(time.monotonic() - started) * 1000)

        return StepResult(kind=step.kind, status="error",
                          error=f"unhandled step kind {step.kind!r}")

    except Exception as exc:  # noqa: BLE001 — a step failure must not crash the run
        return StepResult(kind=step.kind, status="error",
                          error=f"{type(exc).__name__}: {exc}",
                          duration_ms=(time.monotonic() - started) * 1000)


def _find_git_root(start: Path) -> Path | None:
    """向上走找含 .git 的目录(源码树/安装目录都安全)。"""
    try:
        current = start.resolve()
    except OSError:
        current = start
    for path in (current, *current.parents):
        if (path / ".git").exists():
            return path
    return None


def _git_revision(root: Path) -> str:
    """Best-effort ``git rev-parse HEAD`` of the code under eval."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return ""


_SETTINGS_META_FIELDS = (
    "model", "provider", "small_model", "thinking", "system_prompt",
    "max_iterations", "max_tokens", "context_budget_tokens",
    "context_reserve_tokens", "context_keep_recent_tokens",
    "summary_max_tokens", "shell_timeout", "max_model_retries",
)


def _settings_meta(settings: Settings) -> dict[str, Any]:
    return {key: getattr(settings, key, None) for key in _SETTINGS_META_FIELDS}


def _collect_artifacts(
    scenario: EvalScenario,
    workspace: Path,
    home: Path,
    session_path: Path,
    final_reply: str,
    tool_calls: list[dict[str, Any]],
    changed_files: list[tuple[str, str]],
    patch: str,
    scenario_client: ScenarioClient | None,
    *,
    settings: Settings,
    git_revision: str = "",
    status: str = "ok",
    duration_ms: float = 0.0,
    failures: list[str] | None = None,
) -> EvalRunArtifacts:
    session_jsonl = ""
    if session_path.exists():
        session_jsonl = session_path.read_text(encoding="utf-8")

    trace_parts: list[str] = []
    traces_dir = home / "traces"
    if traces_dir.exists():
        for trace_file in sorted(traces_dir.glob("*.jsonl")):
            trace_parts.append(trace_file.read_text(encoding="utf-8"))

    usage: list[dict[str, Any]] = []
    usage_path = home / "usage.jsonl"
    if usage_path.exists():
        for line in usage_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                usage.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return EvalRunArtifacts(
        name=scenario.name,
        status=status,
        duration_ms=duration_ms,
        final_answer=final_reply,
        tool_calls=tool_calls,
        changed_files=changed_files,
        workspace_diff=patch,
        session_jsonl=session_jsonl,
        trace_jsonl="".join(trace_parts),
        usage=usage,
        failures=failures or [],
        model=settings.model or "",
        provider=settings.provider or "",
        config=_settings_meta(settings),
        git_revision=git_revision,
    )
