# Pi 第六、七章消息系统与事件驱动架构对照

第六章回答"消息到底是什么、Agent 内部的消息和发给模型的消息一样吗"；第七章回答"事件怎么从 Agent 内部传到外部、谁在监听、为什么发完事件要等"。两章在类型上耦合（事件载荷就是消息），所以作为一个批次落地。本文档把两章的每个设计决策对应到本仓库的具体文件与代码。

---

## 1. 第六、七章位于三层架构的哪里

```
coding_agent/（产品层）
  ├── messages.py        ← ch6 应用层：注册自定义消息（核心空槽的填充物）
  ├── app.py             ← Harness.__init__ 调 register_coding_agent_messages()
  │                        respond() 把 agent.listeners 注入 AgentLoopConfig
  └── session.py         ← 用户消息改走构造器（make_user_message）

agent/（Agent 层）
  ├── messages.py        ← ch6 核心：AgentMessage/LlmMessage 别名、4 个构造器、
  │                        自定义消息注册表、default_convert_to_llm（跨层翻译）
  ├── events.py          ← ch7 核心：10 个内核事件、AgentEventSink（同步屏障）、
  │                        make_legacy_adapter（旧字符串事件复现器）
  ├── types.py           ← AgentContext.messages: list[AgentMessage]
  │                        AgentLoopConfig.listeners 字段；re-export convert
  ├── agent_loop.py      ← run_loop 构造 sink；turn/message/agent 事件发出点；
  │                        transform_context → convert_to_llm 管道（早已存在）
  ├── tools.py           ← ExecutionContext.event_sink 双通道；_ProgressGate
  │                        （= Pi 的 acceptingUpdates 闸门，ch5 已存在）
  └── runtime.py         ← Agent.subscribe() / Agent.listeners（Pi 的 Agent 壳）

ai/（模型调用层，ch4 不动）
  └── api/ 翻译器         ← provider 协议翻译，与 convert_to_llm 是两层独立抽象
```

分层约束由 `tests/test_architecture.py` 的 AST 扫描强制：`agent/messages.py`、`agent/events.py` 不 import `coding_agent`；`coding_agent/messages.py` 可以 import `agent`（单向）。

---

## 2. 内富外严：为什么消息要分两层

教程第六章的主线：**消息数据结构有两个读者，需求互相冲突**——

- **LLM**（协议强制，不能改）：只认识 `user` / `assistant` / `tool` 三种角色
- **功能层**（UI、持久化、可见性控制）：需要结构化字段（`details`、`terminate`、`thinking`、`tool_name`……）

提前拍扁（直接存 LLM 格式）→ 功能层永远拿不回结构；只存结构化、不进 LLM 上下文 → 模型失忆。Pi 的方案是两层各管各的，本仓库完全落地：

| 层 | 本仓库的形态 | 服务谁 |
|----|------------|--------|
| `AgentMessage`（内层） | `agent/messages.py` 的类型别名 + 构造器产出的 dict，可带内部键和 `role="custom"` | UI、trace、持久化、可见性控制 |
| `LlmMessage`（外层） | `default_convert_to_llm()` 的输出：3 种标准 role、内部键已剥离 | 模型 API |

自定义消息带来的三个独立能力在本仓库的对应物：

1. **UI 专用渲染** → 注册表 `CustomMessageType.render` 钩子（`coding_agent/messages.py` 的摘要渲染器）
2. **持久化保真** → 内层消息保持结构化 dict，session JSONL 存的是完整字段
3. **角色可见性控制** → `exclude_from_context` 字段 + `is_excluded_from_context()`：标准消息做不到"UI 可见、LLM 不可见"，自定义消息可以

---

## 3. 类型化构造器：dict 运行时上的正式契约

Pi 用 TS 联合类型定义消息。本仓库**运行时消息保持 dict**（159→189 个测试、session 持久化、两个 provider 翻译器全都吃 dict 形状），正式契约放在构造器上——这是 Pi 的类型在 Python 同步生态里的对应物：

```python
# agent/messages.py —— 全部 context 消息的唯一构造点
user_message(content, **extra)
assistant_message(content, *, thinking=None, tool_calls=None, **extra)
tool_result_message(result: ToolResultMessage)   # 收编 ch5 的信封
custom_message(custom_type, content, *, exclude_from_context=None, **fields)
```

改造前 6 个散落的 dict 字面量构造点（`agent_loop.py` 5 处 + `session.py` 1 处）全部收编。`tool_result_message()` 同时是 ch5 `ToolResultMessage` 信封进入 context 的唯一通道——ch5 文档预告的"完整双层体系属于第六章"在此兑现。

一个设计细节：`custom_message(exclude_from_context=None)` 用 `None` 表示"不置标志"，让**类型级默认**（注册表的 `CustomMessageType.exclude_from_context`）生效；显式传 bool 则消息级覆盖类型级。这是"三个可见性级别"（全可见 / LLM 不可见 / 仅持久化）的实现基础。

---

## 4. 自定义消息注册表

Pi 的 `CustomAgentMessages` 是核心包里的**空接口**，应用层用 TS 声明合并注入自己的类型——核心包零依赖，应用层全类型安全。Python 没有类型层面的声明合并，对应物是**运行时注册表**：

```python
# agent/messages.py —— 核心包的空槽
@dataclass(frozen=True)
class CustomMessageType:
    name: str
    to_llm: Callable[[AgentMessage], LlmMessage] | None = None  # None → 默认 user 翻译
    exclude_from_context: bool = False                          # 类型级默认可见性
    render: Callable[[AgentMessage], str] | None = None         # UI 渲染钩子

register_custom_message_type(spec)   # 重复注册报错（声明合并的"编译期冲突"对应物）
clear_custom_message_registry()      # 测试夹具
```

```python
# coding_agent/messages.py —— 应用层填充空槽
COMPACTION_SUMMARY = "compaction_summary"

def register_coding_agent_messages() -> None:  # 幂等，Harness.__init__ 调用
    ...  # to_llm: [会话摘要 v{N}]\n{content}；render: 🗜 摘要（覆盖至 #{id}）
```

对照 Pi coding-agent 的 4 种自定义消息，本仓库的取舍：

| Pi 类型 | 本仓库 | 原因 |
|---------|--------|------|
| compactionSummary | ✅ 注册（`ops/session_store.py` 已有 `CompactionEntry` 持久化对应物） | 有真实功能 |
| bashExecution | ❌ | 无 bash 执行记录消息功能 |
| branchSummary | ❌ | 无分支功能 |
| custom 通用槽 | ✅ 注册表机制本身就是槽 | 任何应用可注册 |

**未注册类型的回退**：翻译成默认 user 消息（`_default_custom_to_llm`），不静默丢弃——丢弃会让持久化重放后模型上下文无声变化。这也就是 Pi 的核心洞察在本仓库的体现：**所有自定义消息最终都变成 user 角色**（LLM API 要求 user/assistant 交替，系统注入的信息放 user 最安全）。

本批次**只注册+测试，不把摘要消息注入 context**——注入点属于 Session 层的上下文组装，归 ch8/ch9（见 §11）。

---

## 5. transform_context 与 convert_to_llm 管道

管道骨架在 ch3 就已存在（`agent_loop.py` 的 `run_loop` 内层循环开头），本章把它正式化：

```
context.messages: list[AgentMessage]         ← 内层，功能全量
      │
      ▼  transform_context（可选，同层 AgentMessage[] → AgentMessage[]）
      │     裁剪/注入/压缩的策略点（ch8/ch9 的主战场）
      ▼  convert_to_llm（必选，跨层 AgentMessage[] → LlmMessage[]）
      │     default_convert_to_llm：① 过滤 exclude_from_context
      │     ② custom → 注册表 to_llm（未注册回退 user）
      │     ③ 标准 role 透传  ④ 剥离 INTERNAL_KEYS
      ▼
AIContext.messages → ai/api/ 翻译器          ← provider 协议翻译（ch4 层）
```

本章对管道做的两个实质升级：

1. **内部键剥离上移到 convert_to_llm**。`details`/`terminate`/`thinking`/`tool_name` 原本靠 provider 翻译器的白名单读取"顺带"丢弃，现在由 `default_convert_to_llm` 显式剥离——**convert_to_llm 成为唯一的跨层边缘**，翻译器退为纵深防御。测试安全性：`test_loop.py` 对 `client.calls` 的断言只查 `role`/`content`，不查内部键。
2. **excludeFromContext 只作用请求快照**。被过滤的消息**仍在 `context.messages` 里**——UI 可见、持久化保留，只是那一次 LLM 调用看不到。可见性控制不破坏内层历史。

为什么换 LLM 提供商不用动 convert_to_llm：它输出的 3 种标准 `LlmMessage` 与 provider 无关；再把标准消息翻成各家私有格式是 `ai/api/` 翻译器的事。**消息类型翻译与 provider 协议翻译是两层独立抽象**，互不干扰。

---

## 6. 十种内核事件与四层嵌套生命周期

`agent/events.py` 定义 10 个 frozen dataclass 事件（各带 `kind: Literal[...]` 判别字段），对应 Pi 的 4 层嵌套：

```
agent_start ──────────────────────────────────── agent_end      （整个运行）
  turn_start ─────────────────────────────── turn_end            （一次模型调用+工具批次）
    message_start → message_update ×N → message_end              （一条消息）
    tool_execution_start → update ×N → tool_execution_end        （一次工具执行）
```

10 事件 × 旧字符串名对照（旧名由 `make_legacy_adapter` 复现，见 §7）：

| 内核事件 | 旧字符串名 | 备注 |
|----------|-----------|------|
| AgentStartEvent / AgentEndEvent | （无旧名，typed-only） | `trace.*` 产品事件仍覆盖运行边界 |
| TurnStartEvent / TurnEndEvent | `turn.started` / `turn.completed` | 载荷逐键一致（含 iteration、usage、tool_count…） |
| MessageStartEvent / EndEvent（source=assistant） | （无旧名） | assistant 消息边界是新增 typed 事件 |
| 同上（source=steering/follow_up） | `message.started` / `message.completed` | 200 字预览，与旧行为一致 |
| 同上（source=tool） | （无旧名） | 工具结果消息也有完整生命周期 |
| MessageUpdateEvent | `llm.text.*` / `llm.thinking.*` / `llm.tool_call.*`（9 个） | 按 `assistant_message_event.kind` 折叠 |
| ToolExecutionStartEvent / UpdateEvent / EndEvent | `tool.started` / `tool.progress` / `tool.execution_end` | 载荷逐键一致 |

**message_update 的透传设计**（教程第六节"text_delta 的完整旅程"的落地）：

```python
@dataclass(frozen=True)
class MessageUpdateEvent:
    message: AgentMessage                          # 累积快照
    assistant_message_event: AssistantMessageEvent # ai 层事件原样透传（is 同一对象）
    turn_index: int = 0
```

Agent Loop 不关心 delta 的具体类型——它只说"消息更新了"；关心细节的消费者（如 adapter）从透传字段里取。`tests/test_chapter7_events.py` 断言了透传对象的身份同一性。

**append 所有权**：Pi 的 `processEvents` 在 `message_end` 时把消息 append 进 `state.messages`。本仓库完全对齐——`AgentEventSink._update_state` 在 `MessageEndEvent` 时 append，`agent_loop.py` 不再自己 append（assistant 消息、steering/follow-up 注入、工具结果消息、截断拒绝消息全部改走 sink）。唯一的例外是 empty-retry 提示词（`user_message("Please provide a response.")`）：它走 prompt 注入路径，Pi 里这类注入也不走事件流，保留手动 append（见 §11）。

---

## 7. 同步屏障在同步 Python 中的翻译

Pi 的 `await emit(...)` 是**同步屏障**：先更新内部状态，再按订阅顺序逐一 await 监听器，全部处理完才走下一步。翻译到本仓库的 sync 架构（有意不用 asyncio）：

| Pi（async） | 本仓库（sync） | 论证 |
|-------------|---------------|------|
| `await emit(event)` | `AgentEventSink.process_event(event)` | 同步代码里"顺序调用全部监听器"本身就是屏障——不存在微任务交错 |
| 先更新 state 再 await listeners | `_update_state()` 先行，listener 循环在后 | 完全一致；测试断言 listener 触发时 `context.messages[-1] is event.message` |
| processEvents 拥有 append | sink 拥有 append（§6） | 完全一致 |
| `tool_execution_update` collect-then-batch | **不实现**——那是 async 调度产物（攒 Promise 避免微任务风暴）；同步调用天然有序、天然阻塞，缓冲反而引入延迟与丢失窗口 | 设计意图的两个承载物另有两处对应 ↓ |
| 高频 update 不拖垮内核 | update 事件**状态豁免**（`_update_state` 忽略 `ToolExecutionUpdateEvent`） | "高频低值可合并"的语义 |
| `acceptingUpdates` 闸门 | `_ProgressGate`（`agent/tools.py`，ch5 已存在） | execute settle 即关闭，迟到 update 静默丢弃；测试用后台线程验证 |

**监听器异常策略**（Pi 的保险丝原则）：

- 内核监听器（`wrap=False`，默认）：异常**不捕获**，直接冒泡——一个 UI 渲染 bug 应该让整个运行失败、问题立刻可见，而不是静默吞掉让状态悄悄错乱
- 包裹档（`subscribe(listener, wrap=True)`）：try/except 隔离，第三方扩展崩溃不拖垮运行
- `LoopHooks`：吞异常的 `_safe_call` 本就是扩展档语义（见 §9）

**兼容层：make_legacy_adapter**。typed 事件要落地，又不能破坏 tracer/flow 图/CLI/TUI/Web/65 个 loop 测试——解法是 adapter 把内核事件**逐字节复现**成旧字符串事件，且**第一个订阅**（保证旧事件相对顺序不变）。`test_loop.py` 全绿即"映射保真"的回归证明；`test_chapter7_events.py` 另有逐键载荷断言。旧字符串通道（`HarnessEvent` 信封 + `make_event` + redaction/sequence/usage 侧信道）原样保留为**产品层信封**。

---

## 8. 内核层与产品层事件

Pi 的分层规则：**拿掉某个事件后内核还能正常运行，它就属于外层**。本仓库全部事件按此分类：

**内核层（10 种 typed，`agent/events.py`）**：agent_start/end、turn_start/end、message_start/update/end、tool_execution_start/update/end。

**产品层（字符串事件，不变）**：

| 命名空间 | 事件数（约） | 为什么属于产品层 |
|----------|-----|------------------|
| `trace.*` | 9 | 运行边界的产品化封装（含 redaction/sequence/usage 记账），内核不需要 |
| `llm.started/completed/error/failed` | 4 | 服务 tracer 和 usage 侧信道（`app.py` keyed on `llm.completed`） |
| `tool.requested/completed/approval.*` | 4 | 审批流与结果记录是产品策略 |
| `loop.*`（limit_reached、aborted、steered、overflow_recovery、empty_response_retry、truncation_* 等约 20 种） | 20 | 全是诊断/策略事件——拿掉任何一个内核照常运行 |
| `context.*` / `memory.*` / `persistence.*` / `rag.*` / `subagent.*` / `session.replay.*` / `sandbox.*` | 余量 | 全是产品功能 |

注意一个边界案例：`llm.*` 的 **delta** 事件（text/thinking/tool_call 共 9 个）折叠进了内核的 `message_update`，但 `llm.started/completed/error` 留在产品层——判断依据就是上面的规则（usage 记账与内核无关）。

**预告**：Pi 的 Session 层还有 7 种产品事件（queue_update、compaction_start/end、auto_retry_start/end、session_info_changed、thinking_level_changed）。本仓库**本批次不实现**，只确立"内核 vs 产品"的分层模式；它们随压缩与会话管理功能归 ch9/ch10。

---

## 9. 三层听众与 LoopHooks 的定位

本仓库现在有三层听众，对应 Pi 的错误处理哲学（对内 fail-fast、对外隔离）：

```
1. 内核监听器（fail-fast）
   Agent.subscribe(listener) / AgentLoopConfig.listeners
   → 异常冒泡，运行失败，问题立刻可见
   → 框架内部组件走这层（legacy adapter 就是第一个内核监听器）

2. 包裹监听器（wrap=True）
   Agent.subscribe(listener, wrap=True)
   → try/except 隔离，第三方扩展崩溃不拖垮运行

3. LoopHooks（吞异常扩展档，12 个具名回调）
   agent/hooks.py，_safe_call 吞掉所有异常
   → 语义恰好就是 Pi 的"框架包裹第三方扩展"，本批次保留不动
```

为什么不把 LoopHooks 合并进监听器注册表：① 12 个 hook 与 10 个内核事件不是一一映射（`on_model_call`/`on_loop_error` 等无对应事件）；② 合并会改变吞异常语义，已有依赖"hook 炸了我没事"的代码会受影响；③ 合并应是统一评估后的决定，列为 ch8 候选（§11）。

`Agent.subscribe()`（`agent/runtime.py`）是 Pi 的 `Agent.subscribe` 对应物：壳持有监听器清单，`Harness.respond()` 每次运行时把 `agent.listeners` 注入 `AgentLoopConfig` → `run_loop` 构造 sink 时按顺序订阅（adapter 永远第一）。

---

## 10. 建议学习顺序

1. `src/lsm_harness/agent/messages.py` —— 内富外严的全部契约（别名/构造器/注册表/convert）
2. `src/lsm_harness/coding_agent/messages.py` —— 应用层怎么填核心的空槽
3. `src/lsm_harness/agent/events.py` —— 10 事件、sink 屏障、legacy adapter
4. `src/lsm_harness/agent/agent_loop.py` —— `run_loop` 开头 20 行（sink 构造 + 订阅顺序）、`_consume_assistant_stream`（delta 折叠）、`_inject_pending_messages`（append 所有权）
5. `src/lsm_harness/agent/tools.py` —— `ExecutionContext.event_sink` 双通道、`_ProgressGate` 闸门
6. `src/lsm_harness/agent/runtime.py` + `coding_agent/app.py` —— `Agent.subscribe` 与 listeners 注入
7. `tests/test_chapter6_messages.py` / `tests/test_chapter7_events.py` —— 全部语义的可执行规格

---

## 11. 当前有意保留的范围边界

**与 Pi 的偏差（如实记录）**：

- `tool_execution_update` 不实现 collect-then-batch——async 调度产物，同步语境下忠实翻译是"顺序调用 + 状态豁免 + 闸门"（§7 表格）
- empty-retry 提示词消息保留手动 append——Pi 里 prompt 注入也不走事件流；无对应内核事件
- 消息运行时保持 dict 而非 dataclass 联合类型——189 个测试、持久化、翻译器全都吃 dict；正式契约在构造器与注册表上（§3）
- 截断拒绝（truncation_rejected）的工具消息走 sink 的 message 事件（source="tool"），但 Pi 无完全对应物，属于本仓库 ch3 就有的截断保护机制

**后续章节拥有的事项**：

- **ch8/ch9**：compaction_summary 消息注入 context（本批次只注册+测试）；transform_context 的压缩策略；LoopHooks → listener 统一评估
- **ch9/ch10**：7 种 Session 层产品事件（queue_update、compaction_start/end、auto_retry_*、session_info_changed、thinking_level_changed）
- **不做**：bashExecution / branchSummary 自定义消息（无对应功能）；旧字符串事件名不退役（adapter 是 tracer/flow/web/tests 的供血管道，无限期保留）
