# lsm-harness ↔ pi ↔ tau 核心架构对比（不含 TUI / eval）

对比对象：
- **pi** — `/Users/lsm/Desktop/pi-main`（TypeScript，`packages/{agent,ai,coding-agent,session-backends}`）
- **tau** — `/Users/lsm/tau-main`（Python，`src/{tau_agent,tau_coding,tau_ai}`）
- **lsm** — `/Users/lsm/Desktop/lsm-harness`（Python，`src/lsm_harness/{agent,ai,coding_agent}`）

lsm 的定位是「用 Python 复刻 pi 的架构」，tau 是另一个独立的 Python 复刻。
三个项目共享同一套核心心智模型（reason→act→observe loop、typed 事件、
append-only 会话树、steer/follow-up 队列），但在**并发模型**、**会话持久化协议**
和**工程完备度**上分道扬镳。

---

## 1. 依赖方向与分层

三者一致的三层依赖方向：`ai ← agent ← coding-agent`。

| 层 | pi | tau | lsm |
|---|---|---|---|
| AI provider | `packages/ai` | `src/tau_ai` | `src/lsm_harness/ai` |
| Agent 核心 | `packages/agent` | `src/tau_agent` | `src/lsm_harness/agent` |
| 产品/composition root | `packages/coding-agent` | `src/tau_coding` | `src/lsm_harness/coding_agent` |

**lsm 特有**：`ToolRegistry` 三层依赖 `AITool → AgentTool → ToolDefinition`
（工具定义层在 coding_agent，执行层在 agent，wire 层在 ai），这是 CLAUDE.md
明确点名的有意差异——pi/tau 都用「工具数组 + 单层 AgentTool」。

---

## 2. 并发 / 取消模型（三者最大的分岔）

| | pi | tau | lsm |
|---|---|---|---|
| 并发原语 | async/await | asyncio | **同步生成器 + ThreadPoolExecutor** |
| 取消 | `AbortSignal` | `CancellationToken`（自定义） | **`threading.Event`** |
| loop 形态 | async 函数 + emit 回调 | **async generator**（yield 事件） | 同步函数 + emit/sink 双通道 |
| 工具并行 | loop 内 `Promise.all`（parallel/sequential） | **串行**（`for call in calls`） | `ToolRegistry.execute_batch` 线程池并行 |

关键点：
- **pi**：`runAgentLoop(prompts, context, config, emit, signal, streamFn)` 是 async，
  事件通过 `await emit(event)` 推给 sink。
- **tau**：`run_agent_loop(...) -> AsyncIterator[AgentEvent]` 是 async generator，
  事件 **yield** 出去（consumer 拉取），这是它和 pi 的最大形态差异。
- **lsm**：`run_agent_loop(*, context, config, stream_fn, emit, interrupt, sink)` 是
  **同步**函数，事件走两条通道——`emit`（legacy 字符串，给 tracer/CLI/RPC 兼容）
  和 `sink.process_event`（typed，给新代码）。这是 CLAUDE.md 声明的「有意为之」：
  不抄 pi 的 Promise、不抄 tau 的 async generator。

**tau 的一个缺口**：loop 里工具是串行 `for`，即使 `AgentTool.execution_mode="parallel"`
  字段存在，tau 的 loop 也从未并发执行工具。lsm 和 pi 都真正支持并行工具批。

---

## 3. Agent 层（有状态壳）

| 职责 | pi `Agent`（588 行） | tau `AgentHarness`（259 行） | lsm `Agent`（616 行） |
|---|---|---|---|
| 入口 | `prompt` / `continue` | `prompt` / `continue_` | `prompt` / `continue_` |
| 取消 | `abort` / `signal` | `cancel` | `abort` / `signal` |
| 队列 | `steer` / `followUp` / `clear*Queue` | `steer` / `follow_up` / `clear_queues` | `steer` / `follow_up` / `clear_*_queue` |
| 空闲 | `waitForIdle` | —（无） | `wait_for_idle` |
| 重置 | `reset` | —（无） | `reset` |
| 状态 | `state`（MutableAgentState） | `messages` 直接暴露 | `state`（AgentState，经 sink） |
| 订阅 | `subscribe(listener)` | `subscribe(listener)` | `subscribe(listener, wrap=)` |
| 生命周期 | 私有 `runWithLifecycle` | `_run`（async gen） | **公开 `begin()`/`run()`/`finish()`** |

**lsm 特有**：把 pi 私有的 `runWithLifecycle` 拆成公开的三段
`begin()`（接受即 busy）→ `run()`（跑 loop）→ `finish()`（落结果+清投影），
让 `CodingSession` 能在 loop 返回后、finish 前做**持久化/终端事件**——pi 是在
`agent_end` listener 里做持久化，lsm 把时机显式化，顺带消灭「已接受但未 running」
的启动窗口（TUI/RPC 都受益）。

**pi 特有**：`processEvents` 里对每个 listener 都 `await`，且把 listener 的 settle
纳入 run 的 idle 判定（`agent_end` 之后、所有 await 的 listener 结束才算 idle）。
lsm 的 sink 是同步 dispatch，不做这个语义。

---

## 4. Agent Loop（无状态循环）

三者的循环结构几乎一致（双循环：内循环「工具+steering」、外循环「follow-up」）：

```
emit agent_start / turn_start
→ 注入 pending steering
→ stream assistant
→ 若 error/aborted → turn_end + agent_end，返回
→ 逐个执行工具，工具结果回流 context
→ turn_end
→ 内循环继续（还有工具或 steering）
→ 外循环：drain follow-up，继续或退出
→ agent_end
```

**lsm 的 loop 明显更「防御化」**——顶部注释列了「十个 guardrails」：
1. streaming 细粒度事件
2. truncation 保护（stop_reason=length 拒绝所有工具调用）
3. interrupt
4. steering
5. overflow 恢复（length 后 compaction 重试）
6. 空响应重试（最多 2 次）
7. length 恢复上限（每 trace 最多 3 次 compaction）
8. 重复工具错误早退
9. 错误分类（arrearage / rate-limit / transient / permanent）
10. 生命周期 hooks

这些在 pi 里分散在 `agent-loop.ts` + `compaction/` + retry 逻辑里；tau 的 loop 是
**最小实现**（376 行，没有 compaction/retry/overflow 恢复，那些在
`tau_coding/session.py` 上层）。lsm 把它们全收进了 loop + `AgentLoopConfig`
（config 字段比 pi 多出 `max_empty_retries`、`max_length_recoveries`、
`on_truncation`、`governor`、`hooks`、`initial_pending_messages`、
`approval_broker`、`trace_id` 等）。

---

## 5. 消息模型

| role | pi | tau | lsm |
|---|---|---|---|
| user | ✓ | ✓ | ✓ |
| assistant | ✓ | ✓ | ✓ |
| tool result | `toolResult` | `toolResult` | **`tool`** |
| custom | （content 里区分） | `custom` | `custom` |
| bashExecution | — | `bashExecution` | — |
| branchSummary | — | `branchSummary` | — |
| compactionSummary | — | `compactionSummary` | — |

**lsm 的一个命名残留**：ToolResultMessage 的 role 是 `"tool"`，而 pi/tau 规范是
`"toolResult"`（tau 的 `jsonl.py` 里甚至有一段 `role == "tool" → "toolResult"`
的 v1 迁移代码，说明 tau 早期也用 `"tool"` 后来改规范了）。lsm 沿用旧名，靠
`convert_to_llm` 在 LLM 边界转成 provider 要的 toolResult。这是可以对齐的点。

lsm 的消息分两层：`ai/messages.py`（wire 层 UserMessage/AssistantMessage/
ToolResultMessage）+ `agent/messages.py`（CustomMessage + role 常量），比 pi/tau
更明确地区分了「AI wire 形状」和「agent 内部形状」。

---

## 6. 事件模型 + 归约器

三者的 typed 事件集合几乎同构：
`agent_start/end, turn_start/end, message_start/update/end, tool_execution_start/update/end`。

| | pi | tau | lsm |
|---|---|---|---|
| 事件通道 | emit 回调 | yield | **双通道**：legacy `emit` 字符串 + typed `sink` |
| 归约器 | `Agent.processEvents`（switch）+ **`harness/reducer.ts`（667 行持久化状态机）** | 无独立归约器（直接 append） | `AgentEventSink._update_state` |

**pi 有两层归约**，这是它最重的部分：
1. `Agent.processEvents`——内存 state（streamingMessage/messages/pendingToolCalls/errorMessage）。
2. `harness/reducer.ts` + `harness/session/types.ts`——**lane/record 持久化归约器**，
   一个完整的崩溃恢复状态机（`operation_started`/`step_attempt`/`tool_started`/
   `queue_enqueued`/`write_deferred`/`usage` 等记录，`RecordLogCorruption` 分类）。

**lsm 只有一层**：`AgentEventSink._update_state`（内存归约，state-first 再 dispatch
listener），持久化靠 `session recorder` 把 entry append 进 JSONL，**没有 pi 的
lane/record 恢复协议**。lsm 的三层 listener 编排（legacy adapter 首位槽 →
persistent(wrap 档) → run_listeners）是对 pi `processEvents` 订阅序的精确复刻。

**tau 最简**：`AgentHarness` 没有独立归约器，loop 里 `messages.append()` 直接改
列表；中断补 tool result 的逻辑（`_append_interrupted_tool_results`）放在 harness 里。

---

## 7. 工具模型

| | pi | tau | lsm |
|---|---|---|---|
| 容器 | **数组** | 数组（`list[AgentTool]`） | **`ToolRegistry`**（字典 + 线程安全） |
| 执行签名 | loop 内 | `execute_fn`（Awaitable，带 `on_update` 进度） | `execute`（同步 Callable） |
| 并行 | loop `Promise.all` | loop 串行 | `execute_batch`（ThreadPoolExecutor） |
| 进度 | — | `ToolExecutionUpdateEvent.partial_result` | `on_update` 字符串进度 |
| 校验 | `validateToolArguments` | pydantic | `prepare_tool_call` |
| 渲染 | （coding-agent 层） | `render_call`/`render_result` 协议 | `ToolDefinition` + `tool_renderers` 字典 |
| 安全/效果 | — | — | **`effect`（read/local_write/external_write）** |
| hooks | `beforeToolCall`/`afterToolCall`（loop config） | `before_tool_call`/`after_tool_call`（loop 参数） | `before_hook`/`after_hook`（工具级）+ loop 级 |

**lsm 特有**：`ToolRegistry` 的 `execute_batch` 有完整的
`preflight → execute → finalize` 三阶段（先顺序 preflight，并行 execute，
顺序 finalize），并且「任一工具 sequential 则整批降级串行」。`effect` 三级
安全分类是 lsm 自创的（pi/tau 没有）。

**tau 特有**：工具进度走 `on_update(partial_result)` 回调 + `ToolExecutionUpdateEvent`，
比 lsm 的字符串进度更结构化。

---

## 8. 会话树 / 持久化（第二个大分岔）

| | pi | tau | lsm |
|---|---|---|---|
| 事实源 | `SessionStorage` 抽象（**lane + entry + record log**） | JSONL append-only | JSONL append-only |
| 后端 | JSONL（**原子整文件重写**）+ SQLite（`session-backends`） | JSONL | JSONL（事实源）+ SQLite（**投影**：chat_log/sessions/summaries） |
| 分支指针 | lane 的 `leafId` | **`LeafEntry` 持久化条目** | recorder 内存 `_last_entry_id`（文件最后一行） |
| 崩溃恢复 | record log 状态机（open operation 恢复） | 无 | torn record 修复 + 中断工具结果持久化 |

**pi 的会话协议是三者里最重的**：`Entry`（树节点）+ `LaneRecord`（操作记录：
operation_started/finished、step_attempt、tool_started、queue_enqueued、usage…）
+ lane（泳道，支持多分支并发）。JSONL 后端是「mutation 原子重写」，SQLite 后端是
`session-backends/sqlite-node`。这套协议让 pi 能做**崩溃后精确恢复**（还原 open
operation、unfinished step、pending queue），lsm/tau 都做不到。

**lsm 的定位**（CLAUDE.md 明确）：「JSONL Session Tree 是会话事实源，SQLite 只是
检索/记忆投影」——这是 tau 路线，不是 pi 路线。lsm 的 SQLite（chat_log）只是给
`list_sessions`/`summary_info` 的检索投影。

**entry 类型对比**：

| entry | pi | tau | lsm |
|---|---|---|---|
| message | ✓ | ✓ | ✓ |
| custom_message | — | — | **✓（独立类型）** |
| model_change | ✓ | ✓ | ✓ |
| thinking_level_change | ✓ | ✓ | ✓ |
| **active_tools_change** | ✓ | — | — |
| compaction | ✓（retainedTail） | ✓（replaces_entry_ids） | ✓（first_kept_entry_id 位置覆盖 + through_chat_id） |
| branch_summary | ✓ | ✓ | ✓ |
| label | — | ✓ | ✓ |
| leaf | — | **✓** | —（用 recorder 指针） |
| session_info | — | ✓ | ✓ |
| custom | ✓ | ✓ | ✓ |
| **session header** | — | — | **✓（首行文件头，载初始状态）** |

关键差异：
- **pi 有 `active_tools_change`**（工具集变更也是持久化节点），lsm/tau 没有。
- **tau 有 `LeafEntry`**（叶子指针是 JSONL 里的持久化条目），lsm 用 recorder 的
  内存态 + 文件末行，pi 用 lane 的 `leafId`。
- **lsm 有 `session` header**（文件首行，存 cwd/provider/model/small_model/thinking
  的初始状态，解决「从没切过模型就没状态节点」的 resume 问题），这是 lsm 独有。
- **lsm 的 compaction 有 `through_chat_id`**（legacy chat_log 投影标记）——
  因为 lsm 同时维护 SQLite 投影，pi/tau 没有这个包袱。

---

## 9. CodingSession / Harness 层

| | pi `agent-session.ts`（3342 行） | tau `session.py`（4497 行） | lsm `app.py`（979 行）+ `session.py`（1824 行） |
|---|---|---|---|
| 职责 | 会话生命周期、模型/thinking 切换、compaction、bash、切换/分支 | 同上 + UI 策略 | CodingSession（组合根）+ Session（持久化树） |
| 公开查询 API | —（session tree 直连） | — | **`list_sessions`/`usage_summary`/`summary_info`/`current_path_messages`/`current_path_entries`/`branch_to`** |

**pi 和 tau 都是「巨型单文件」**（3000+/4000+ 行），lsm 把组合根（CodingSession）
和持久化树（Session）拆成了两个文件，并且最近刚给 CodingSession 加了公开查询 API
（供 TUI/RPC 用，不碰 session 内部）。这是 lsm 在工程组织上比 pi/tau 更克制的点。

pi 的 `AgentHarness`（`harness/agent-harness.ts`，508 行）是更底层的一层——它直接
暴露 lane/operation 语义（`LaneBusy`/`NoActiveOperation`/`NothingToResume` 等
TaggedError），lsm 没有这一层（lsm 的 lane 概念不存在，分支就是移动 leaf 指针）。

---

## 10. 总结：lsm 的相对位置

### lsm 与 pi 的差距（pi 有、lsm 没有）
1. **lane/record log 持久化协议**——pi 能崩溃后精确恢复（open operation、pending
   queue、unfinished step），lsm 只有 torn record 修复。
2. **存储抽象 + 多后端**——pi 的 `SessionStorage` 接口（JSONL + SQLite 双实现），
   lsm 硬编码 JSONL 事实源 + SQLite 投影。
3. **`active_tools_change` entry** + 工具集作为会话状态的一部分。
4. **listener 的 async settle 语义**（`agent_end` 后 await 完 listener 才算 idle）。
5. **`Result`/`TaggedError`** 错误建模（pi 用类型化错误，lsm 用字符串/异常）。

### lsm 与 tau 的差距（tau 有、lsm 没有）
1. **async generator loop**——tau 的 `AsyncIterator[AgentEvent]` 是干净的拉取模型，
   lsm 用同步 + 双通道（这是 lsm 的有意选择，不算「缺」，但形态上 tau 更接近 pi
   的事件流）。
2. **`LeafEntry` 持久化叶子指针**（tau 的 leaf 是 JSONL 条目，lsm 是内存态）。
3. **结构化的工具进度**（tau 的 `partial_result`，lsm 是字符串）。
4. **`ResponseTiming`**（time-to-first-token 持久化，lsm 没有）。
5. **bashExecution/branchSummary/compactionSummary 独立消息 role**（lsm 用 note/custom）。

### lsm 相对两者的优势
1. **工具执行真正并行**（ThreadPoolExecutor 的 preflight/execute/finalize 三阶段），
   比 tau 的串行强，与 pi 的 `Promise.all` 对齐。
2. **loop 的十个 guardrails 显式化**（overflow 恢复、空响应重试、错误分类、
   重复工具错误早退），比 tau 的最小 loop 完整，比 pi 更集中。
3. **`ToolRegistry` + `effect` 三级安全分类**（自创）。
4. **公开的三段生命周期** `begin/run/finish`（消灭 busy 窗口 + 显式持久化时机）。
5. **CodingSession 公开查询 API**（前端不碰 session 内部）。
6. **session header 载初始状态**（resume 正确性）。

### 一句话
lsm 是 pi 的「同步生成器 + JSONL 事实源」精简版，架构心智与 pi 对齐、持久化协议
与 tau 对齐（JSONL append-only），并在工具并行、guardrails 显式化、三段生命周期
上做了自己的加法；最大的欠账是 **pi 的 lane/record 崩溃恢复协议**和 **tau 的
async 事件流形态**。

---

## 11. TUI 对比

规模：pi ≈ 41k LOC（双层：`packages/tui` ~24k 终端 UI 工具包 +
`coding-agent/modes/interactive` ~17k agent 应用层）；tau ≈ 15k LOC；lsm ≈ **1.7k LOC**。

| | pi | tau | lsm |
|---|---|---|---|
| 结构 | tui 工具包 + interactive 应用层两层 | state/adapter/app/widgets 四层 | state/adapter/app/widgets/commands/screens/bindings/autocomplete |
| 事件通道 | AgentSession 订阅 | typed 事件（subscribe → adapter → state） | **同 tau**（typed 事件，响应式原位更新） |
| 聊天渲染 | TranscriptView + markdown 组件（editor 2363 行/markdown 1010 行） | TranscriptView（**窗口化**、follow-scroll、markdown 流式 StreamingTranscriptMessageWidget） | TranscriptView（**窗口化**上限+边界占位）+ MessageWidget/ToolCallWidget/**AssistantWidget**（**Markdown 渲染**，流式节流重解析） |
| 工具折叠 | tool-execution 组件可折叠 | 每条目 widget 原位折叠 | 同（单击单项 + Ctrl+O 全局） |
| 会话树/选择器 | tree-selector **1427 行**、session-selector 1031 行 | SessionSidebar（侧栏） | 通用 PickerScreen（54 行）复用三个选择器 |
| 补全 | autocomplete 786 行 | autocomplete 579 行 | **autocomplete**（@文件子序列模糊 + /命令，Tab 接受） |
| 主题 | theme.ts 1308 行 | themes/ 516 行 | 无 |
| 其他 | latex/mermaid/图片/oauth 对话框/settings-selector(880)… | file_drop/local_backends(1313)/terminal_title… | 都没有 |
| 测试 | ~15.5k LOC（editor 行为/渲染快照） | 9 个测试文件 | 34 个测试（state 纯测 + widgets 纯测 + Pilot 冒烟/流式/窗口化/补全） |

**判断**：lsm 的 TUI 是**架构正确性优先的骨架**——typed 事件管线、state 单事实源、
条目控件原位流式更新、单项折叠，与 tau 同构且更干净（tau 的 TranscriptView 是
703-1818 行的巨兽，lsm 同等能力 ~1.7k LOC 总量）。阶段三补齐了三件关键差距：
**markdown 渲染**（顺带修掉 LLM 输出里 `[...]` 被 Rich 当标记解析的注入问题）、
**窗口化**（防长 transcript 堆积控件）、**autocomplete**（@文件 + /命令）。
与两者的 markdown 实现深度仍有差距（tau 用 `MarkdownStream` 增量 append 不重解析，
lsm 用节流全量重解析；pi 有 latex/mermaid/图片），以及主题系统、侧栏、
autocomplete 评分排序这些周边打磨仍是欠账。

## 12. Eval 对比

规模：pi ≈ 1.8k LOC（薄封装，外部依赖 `vitest-evals` + vitest）；tau = **没有**；
lsm ≈ 2.5k LOC **全自研零外部依赖**（+ 623 LOC 测试，28 个测试）。

| | pi | tau | lsm |
|---|---|---|---|
| 框架 | 外包给 `vitest-evals`(Sentry) + vitest runner | 无 | 全自研（types/scenario/runner/assertions/judge/artifacts/compare/variant/task/suites） |
| 模式 | 仅真实模型行为检查 | — | **确定性脚本 + 非脚本任务** 双模式（离线可跑，不需 API key） |
| A/B 变体 | 多个 `createPiCodingAgentHarness({name,model,transformSystemPrompt})` 工厂函数 | — | **`EvalVariant` 一等数据类型**（provider/model/system_prompt/settings_overrides/tool_policy） |
| 任务模型 | `run(prompt)` + `run([prompt,reload])` | — | **多步 scenario**（prompt/abort/reload/continue/compact/command）+ 非脚本 `EvalTask` |
| 断言 | vitest `expect`（代码） | — | 12 种**数据驱动断言**（文件/命令/工具/会话树/摘要）+ `createJudge` 等价物 |
| 统计 | CorrectnessLiftSummary（lift/wins/ties）+ PairedMetricSummary（tokens/ms/**cost**） | — | **成对指标已补齐**（`PairSummary`：lift/wins/ties/losses + tokens/ms/cost 三项 delta） |
| artifacts | `.eval/runs.jsonl` 索引 + session JSONL + source 附件 | — | workspace diff + session JSONL + trace + usage + **provenance（model/provider/config/git_revision）** |
| 重复/稳定性 | harness-table repetitions | — | repetitions × variants，可 `parallel=True`（去掉了 os.chdir 串行锁，线程数有上限） |
| CLI 真实 A/B | `npm run eval -- --provider --model`（vitest-evals） | — | **`lsm eval --compare --baseline kimi/k3 --candidate deepseek/deepseek-chat`**（真 client 由 ModelRuntime 按 variant 解析，setup 失败算一次 failed run 不炸 CLI） |
| 测试用例 | smoke(17 行)+ extensions(140 行) 真实套件 | — | 5 个确定性 scenario + 2 个 task（内置） |

**判断**：lsm 的 eval 框架在**仓库内能力**上其实比 pi 的封装更完整——pi 把框架外包给了
`vitest-evals`，自身只有 pi-harness 适配器；lsm 自研了 variant 一等类型、多步生命周期
scenario、typed 事件收集、离线确定性模式。CLI 侧已接通真 A/B（两个 EvalVariant 分别
构建自己的 CodingSession、真实任务进 `--suite tasks`、成对指标报告）。pi 仍赢在外围
生态：vitest runner 集成（watch/snapshot）、更大规模的真实模型用例，以及成熟的
CI 历史趋势展示。tau 完全没有 eval——lsm 明显领先 tau。

lsm 已补齐产品级 reporter：每次调用自动创建 `.lsm/evals/<run-id>`，场景级
artifact 之外还会生成逐 observation 的 `runs.jsonl` 和机器可读的
`summary.json`；CLI 明确区分 `--offline`、单模型真实评测和成对 A/B。
当前欠账是扩大真实任务集、增加统计置信区间、把历史结果接入 CI 趋势展示。
