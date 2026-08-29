# LSM Harness：Pi 第 6–10 章架构对齐改造文档

> 文档性质：架构改造设计与实施依据  
> 改造范围：消息系统、Session Tree、上下文压缩、Skills 懒加载、模型调用入口及相关边界  
> 不在本轮范围：新增 Web 功能、Graph Workflow、外部服务工具、完整 UI 重做

## 1. 背景与结论

LSM Harness 已经具有可运行的 Pi 风格三层骨架：

```text
coding_agent → agent → ai
```

第 3 章 Agent Loop、第 5 章工具系统和第 7 章事件系统已经形成较稳定的主链路。当前主要问题不是功能缺失，而是第 6、9、10 章的数据结构没有使用同一套完整消息语义：

- Agent 内部消息仍是 `dict[str, Any]`；
- Session 只保存用户输入与最终回答，没有逐条保存 assistant tool call 和 tool result；
- Compaction 从线性 SQLite `chat_log` 读取，而不是从当前 Session Tree 路径读取；
- Session Tree 使用 `through_chat_id` 关联压缩范围，没有使用结构化的 `first_kept_entry_id`；
- Skills 通过关键词匹配后直接注入全文，不是 Pi 的“清单进入提示词、正文按需读取”；
- 主 Agent 调用通过 `client_stream_function()` 进入 Provider，没有统一经过 `stream_simple → registry`。

这些问题相互依赖，必须按以下顺序改造：

```text
第6章严格消息边界
→ 第10章完整 Session Tree
→ 第9章基于树路径的压缩
→ 第8章 Skills 与截断补齐
→ 第4章模型入口统一
→ 类型归属、CLI renderer 和文档清理
```

## 2. 总体目标架构

改造后的核心链路应为：

```text
CLI / TUI / Web
→ Harness.respond
→ SessionManager.build_session_context
    → 当前 leaf 的 Session Tree 路径
    → AgentMessage[]
    → model / thinking 状态
→ AgentContext + AgentLoopConfig
→ run_agent_loop
    → transform_context: AgentMessage[] → AgentMessage[]
    → convert_to_llm: AgentMessage[] → Message[]
    → stream_simple
    → API registry
    → Provider translator
    → 12 种 AssistantMessageEvent
    → 10 种 AgentEvent
    → SessionRecorder 持久化 MessageEndEvent
→ agent_end
→ 基于当前树路径判断并执行 Compaction
→ SQLite 只接收用于记忆/RAG的派生投影
```

核心数据所有权调整为：

| 数据 | 权威来源 | 派生用途 |
|---|---|---|
| 完整对话与分支 | JSONL Session Tree | 恢复、回退、分支、压缩 |
| 当前运行消息 | `AgentEventSink.messages` | Agent Loop 上下文 |
| 长期记忆原料 | SQLite `chat_log` | Consolidation、RAG |
| 模型可见消息 | `Message[]` 请求快照 | Provider API |
| Trace | JSONL Trace | 调试、审计、UI |

## 3. 改造原则

1. 第 3、5、7 章的稳定行为优先保持不变。
2. 先改类型与数据所有权，再改存储和算法。
3. JSONL Session Tree 保存完整 AgentMessage，不保存“为了展示而拼接的文本摘要”。
4. Provider 只能接收 AI 层标准 Message，不能接触 Agent 或产品扩展字段。
5. 压缩、回退和分支必须使用当前树路径，不能使用全会话线性记录代替。
6. SQLite 暂不删除，但从会话权威源降级为记忆系统的派生投影。
7. 每阶段独立可测试、可回滚，不一次性重写所有模块。

---

## 4. 阶段一：建立严格的第六章消息系统

### 4.1 目标

建立明确的两层消息：

```text
AI Message：Provider 中立、模型可见、字段严格
AgentMessage：AI Message + 产品自定义消息
```

### 4.2 文件调整

新增：

```text
src/lsm_harness/ai/messages.py
```

修改：

```text
src/lsm_harness/ai/types.py
src/lsm_harness/agent/messages.py
src/lsm_harness/agent/types.py
src/lsm_harness/agent/agent_loop.py
src/lsm_harness/ai/api/openai_compat.py
src/lsm_harness/ai/api/anthropic_messages.py
tests/test_chapter6_messages.py
```

### 4.3 AI 层标准消息

建议定义内容块：

```python
@dataclass(frozen=True)
class TextContent:
    text: str


@dataclass(frozen=True)
class ImageContent:
    url: str
    media_type: str = ""


@dataclass(frozen=True)
class ThinkingContent:
    thinking: str
    signature: str = ""


@dataclass(frozen=True)
class ToolCallContent:
    id: str
    name: str
    arguments: dict[str, Any]
```

标准消息至少包含：

```python
Message = UserMessage | AssistantMessage | ToolResultMessage
```

要求：

- `AssistantMessage` 能保存 text、thinking、tool call 和 Provider signature；
- `ToolResultMessage` 保存 `tool_call_id`、`tool_name`、模型可见 output 和 `is_error`；
- `AIContext.messages` 改为 `list[Message]`；
- `details`、`terminate`、renderer 数据不得进入 AI 层 Message；
- Provider 翻译器对 Message 联合类型进行穷举转换。

### 4.4 Agent 层消息

Agent 层定义：

```python
AgentMessage = (
    Message
    | CompactionSummaryMessage
    | BranchSummaryMessage
    | RegisteredCustomMessage
)
```

需要保留的 Agent 内部信息通过明确类型表达，不再依赖任意字典字段。

### 4.5 分离两个转换协议

```python
TransformContext = Callable[
    [list[AgentMessage]],
    list[AgentMessage],
]

ConvertToLlm = Callable[
    [list[AgentMessage]],
    list[Message],
]
```

执行顺序固定为：

```text
复制当前 AgentMessage 列表
→ transform_context
→ convert_to_llm
→ AIContext
→ Provider translator
```

### 4.6 转换安全规则

删除黑名单式转换：

```python
{key: value for key, value in message.items() if key not in INTERNAL_KEYS}
```

改为按消息类型显式构造标准 Message。未知自定义消息如果没有注册转换器，应产生明确错误；不能自动把任意结构发送给 Provider。

### 4.7 阶段一测试

- 标准 user/assistant/tool 消息转换顺序正确；
- `trace_secret`、`details`、`terminate` 不会进入 Provider 请求；
- 自定义 translator 返回非法消息时失败在 Agent/AI 边界；
- excluded custom message 不进入模型上下文，但仍保留在 Agent 状态；
- thinking signature 能经过 Session/下一 Turn 后再次发给对应 Provider；
- OpenAI 和 Anthropic 都只接受标准 Message；
- `transform_context` 不修改 Agent 持有的原始消息。

### 4.8 完成标准

项目内不再出现以下公共类型别名：

```python
AgentMessage = dict[str, Any]
LlmMessage = dict[str, Any]
```

---

## 5. 阶段二：建立完整的第十章 Session Tree

### 5.1 目标

Session Tree 保存 Agent 实际发生的每一条消息和状态变化，而不是在 Trace 结束后重建一个简化版本。

### 5.2 文件调整

主要修改：

```text
src/lsm_harness/ops/session_store.py
src/lsm_harness/coding_agent/session.py
src/lsm_harness/coding_agent/app.py
src/lsm_harness/coding_agent/messages.py
src/lsm_harness/coding_agent/cli.py
tests/test_chapter10_session_tree.py
```

建议新增：

```text
src/lsm_harness/coding_agent/session_recorder.py
```

### 5.3 Entry 类型

Session Header 作为文件元信息，不作为普通树节点。树节点至少支持：

```text
message
custom_message
compaction
branch_summary
model_change
thinking_level_change
label
session_info
custom
```

### 5.4 MessageEntry

从：

```python
class MessageEntry:
    role: str
    content: str
    tool_calls: list[dict[str, Any]]
```

改为：

```python
class MessageEntry:
    message: AgentMessage
```

真实工具 Turn 应持久化为：

```text
e1 UserMessage
e2 AssistantMessage(tool call)
e3 ToolResultMessage
e4 AssistantMessage(final answer)
```

禁止将工具调用压缩为：

```text
AssistantMessage("[tools used: ...]")
```

### 5.5 SessionRecorder

产品层建立 `SessionRecorder`，订阅 typed kernel events：

```text
MessageEndEvent(source=user/steering/follow_up)
MessageEndEvent(source=assistant)
MessageEndEvent(source=tool)
```

每次 `MessageEndEvent` 对应一个 Session Entry。EventSink 仍然负责运行期消息追加，SessionRecorder 只负责持久化，不重新构造消息。

初始用户消息需要在进入 Agent Loop 前形成同一种消息事件，或由 SessionRecorder 明确追加一次，不能在 Trace 结束后补写。

### 5.6 延迟首次写入

为避免只有用户输入、没有 Agent 回复的半截文件，可采用：

```text
新会话未完成第一个有效响应
→ Entry 暂存在内存
→ 第一个 assistant message 完成
→ 原子写入 Header + 累积 Entry
→ 后续 Entry 逐行 append
```

如果保留失败 Trace，也必须明确标记其状态，不能与成功对话混淆。

### 5.7 JSONL 与 SQLite

目标关系：

```text
JSONL Session Tree：权威会话记录
SQLite chat_log：成功对话的检索/记忆投影
```

`Session.add_exchange()` 应拆分为：

```text
SessionRecorder：保存完整 AgentMessage Tree
ChatProjector：将成功 Trace 投影到 chat_log，供 Memory 使用
```

### 5.8 build_session_context

返回结构建议为：

```python
@dataclass
class SessionContext:
    messages: list[AgentMessage]
    model: Model | None
    thinking_level: ThinkingLevel | None
```

处理规则：

| Entry | 行为 |
|---|---|
| message | 加入 AgentMessage |
| custom_message | 加入 AgentMessage |
| compaction | 注入摘要，并按 `first_kept_entry_id` 选择性保留 |
| branch_summary | 注入 BranchSummaryMessage |
| model_change | 覆盖当前 model |
| thinking_level_change | 覆盖当前 thinking |
| label/session_info/custom | 不进入 LLM 消息 |

### 5.9 状态节点接线

- `Harness.switch_model()` 成功后追加 `ModelChangeEntry`；
- CLI/TUI 切换 thinking 后追加 `ThinkingLevelChangeEntry`；
- 回退到旧节点后，`build_session_context()` 自动恢复当时的 model 和 thinking；
- 不再只依赖当前全局 `settings` 表示历史状态。

### 5.10 阶段二测试

- 一个两 Turn 工具请求产生完整的 user/assistant/tool/assistant 节点；
- Session JSONL 可独立恢复完整 AgentMessage；
- 回退后旧分支不进入当前上下文；
- 回退后追加消息自然形成兄弟分支；
- 模型和 thinking 随树路径恢复；
- custom message 能持久化和恢复；
- SQLite 不可用时，JSONL 仍能恢复会话；
- JSONL 写入失败不会伪装为完整持久化成功。

### 5.11 完成标准

删除 `add_exchange()` 中的 `[tools used: ...]` 拼接逻辑；Session Tree 能表达一次 Agent Loop 的全部消息边界。

---

## 6. 阶段三：将第九章压缩迁移到 Session Tree

### 6.1 目标

压缩只处理当前 leaf 对应的可见路径，任何已放弃分支都不能进入当前摘要。

### 6.2 文件调整

```text
src/lsm_harness/coding_agent/compaction.py
src/lsm_harness/coding_agent/session.py
src/lsm_harness/coding_agent/app.py
src/lsm_harness/ops/session_store.py
tests/test_chapter9_compaction.py
tests/test_chapter10_session_tree.py
```

### 6.3 压缩输入

禁止以以下查询作为压缩主输入：

```python
SELECT ... FROM chat_log WHERE id > through_chat_id
```

改为：

```python
path = path_to_leaf(entries, leaf_id)
messages = extract_compactable_messages(path)
```

### 6.4 CompactionEntry

建议结构：

```python
@dataclass
class CompactionEntry(SessionEntry):
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    source_entry_count: int
    read_files: list[str]
    modified_files: list[str]
```

`through_chat_id` 可在迁移期保留为旧格式兼容字段，但新逻辑不得依赖它。

### 6.5 切割点算法

分两步实施。

第一步：

- 压缩输入改成当前树路径；
- 暂时只允许 user 边界；
- 保证不会跨分支污染。

第二步：

- 有效切割点允许 user 和 assistant；
- tool result 不能作为切割点；
- assistant 切割导致 split Turn 时，寻找 Turn 起点；
- 分别生成主摘要和 `turnPrefix` 摘要；
- 合并后写入同一个 CompactionEntry。

### 6.6 自动压缩时机

正常路径：

```text
AgentEndEvent
→ 获取最后一次上下文 token usage
→ should_compact
→ compact current path
→ append CompactionEntry
```

异常恢复路径继续保留：

```text
stop_reason=length / context overflow
→ emergency compaction
→ rebuild context
→ retry
```

正常压缩与紧急压缩调用同一套树路径算法。

### 6.7 文件操作追踪

文件读取/修改信息从当前压缩范围内的 ToolResult/ToolCall 消息提取，不再从挂在 user entry 上的 `tool_calls` metadata 提取。

### 6.8 阶段三测试

- 红线判定为 `tokens > context_window - reserve_tokens`；
- 最近消息按 token 预算保留；
- tool result 不是合法切割点；
- split Turn 生成 turnPrefix；
- CompactionEntry 保存 `first_kept_entry_id` 和 `tokens_before`；
- `build_session_context()` 正确跳过被摘要覆盖的节点；
- 回退到 CompactionEntry 之前时，旧消息重新可见；
- A 分支被放弃后，在 B 分支压缩，摘要不得包含 A；
- 连续压缩能增量合并文件列表；
- 压缩失败不移动 leaf、不写入半成品 Entry。

### 6.9 完成标准

Compaction 核心代码不再读取线性 `chat_log`；分支与压缩组合测试通过。

---

## 7. 阶段四：补齐第八章上下文工程

### 7.1 Skills 懒加载

当前关键词匹配后直接注入 Skill 正文的方式改为默认只注入清单：

```xml
<available_skills>
  <skill>
    <name>code-review</name>
    <description>Review code for correctness and safety.</description>
    <location>.lsm/skills/code-review/SKILL.md</location>
  </skill>
</available_skills>
```

清单前加入明确契约：

```text
When a skill matches the task, use read_file to load its SKILL.md before acting.
```

调整文件：

```text
src/lsm_harness/memory/skills.py
src/lsm_harness/coding_agent/session.py
tests/test_chapter8_context.py
```

原关键词匹配模式可作为可选增强保留，例如：

```text
LSM_SKILL_LOADING=lazy     # 默认，Pi 模式
LSM_SKILL_LOADING=matched  # 兼容旧行为
```

### 7.2 Shell 完整输出逃生通道

完整输出必须写到 `read_file` 可访问的位置：

```text
<workspace>/.lsm/tool-results/exec-<id>.log
```

不能继续使用系统 `/tmp/lsm-exec-*.log` 作为模型可读取路径。

返回结果必须提供工作区相对路径，例如：

```text
Full output: .lsm/tool-results/exec-a1b2.log
```

### 7.3 统一截断出口

梳理两套截断：

- 工具专用 `truncate_head/truncate_tail/truncate_line`；
- 通用 `ContextGovernor.manage_result()`。

目标是避免同一输出被重复截断或先落盘后再次替换。建议：

- read、shell、grep 使用工具专用 Pi 截断；
- 其他未知/第三方工具使用 ContextGovernor 兜底；
- ToolResult details 标记是否已经治理，避免重复处理。

### 7.4 阶段四测试

- System Prompt 只包含 Skill 元数据，不包含正文；
- 模型可通过 `read_file` 读取 Skill；
- Shell 截断后的完整文件能被 `read_file` 读取；
- 多字节字符不被切坏；
- 单超长行仍保留头部或尾部；
- 同一工具结果只经过一次主要截断。

---

## 8. 阶段五：统一第四章模型调用入口

### 8.1 目标

主 Agent 的唯一调用链统一为：

```text
Model
→ stream_simple
→ resolve_api_provider(Model.api)
→ translator
→ 12 种 AssistantMessageEvent
→ Agent Loop
```

### 8.2 调整内容

在 `Harness` 主链路中，不再使用：

```python
self.stream_fn = client_stream_function(self.client)
```

改为使用注册表标准入口。依赖注入测试通过临时注册 `ApiProvider` 或直接向 Harness 注入 `StreamFunction` 完成。

`ModelClient.complete()` 兼容门面继续服务于：

- Memory Gate；
- Consolidation；
- RAG rerank；
- Compaction summary；
- 旧集成。

主 Agent Loop 不再依赖 `ModelClient`。

### 8.3 阶段五测试

- 真实 Harness 调用经过 registry；
- 未注册 `Model.api` 返回结构化失败；
- thinking/cache capability fallback 只执行一次；
- retry 只发生在语义输出开始前；
- OpenAI/Anthropic translator 都产生完整 12 事件协议；
- 测试 Provider 不需要真实 API key。

---

## 9. 阶段六：边界清理

### 9.1 类型归属

将以下 Agent 运行概念从 AI 层移至 Agent 层：

```text
TraceResult
TraceStatus
TurnResult 兼容别名
```

AI 层只保留模型请求、模型响应、内容块、工具描述、usage、stop reason 和流事件。

### 9.2 工具独立中断

当前同批工具共享 `AbortHandle`。改为：

```text
global interrupt
+ per-tool timeout abort
→ CombinedAbortHandle
```

一个工具超时不能自动污染同批其他工具。全局用户中断仍应终止全部工具。

### 9.3 CLI renderer

将 `ToolDefinition.render_call` 和 `render_result` 接入 CLI/TUI typed event listener。显示优先级：

```text
custom renderer
→ label
→ tool name
```

### 9.4 Session CLI

建议加入：

```text
/tree
/branch <entry-id>
/branch-summary <entry-id>
/label <entry-id> <name>
```

### 9.5 terminate 语义

项目需要明确选择并统一文档：

```text
any：任意工具 terminate，批次结束后停止后续 Turn
every：只有整批工具全部 terminate 才停止
```

当前实现和已批准的第五章计划使用 `any`；第三章 mapping 仍写 `every`。在没有重新决定前，保留当前 `any` 行为并修正文档，避免行为变化。

---

## 10. 数据迁移策略

### 10.1 旧 JSONL

读取旧 `MessageEntry(role, content, tool_calls)` 时，转换为新的 `MessageEntry(message=...)`：

- user → UserMessage；
- assistant → AssistantMessage；
- 旧 `tool_calls` metadata 只能作为迁移细节保留，不能伪造过去不存在的精确 ToolResult 顺序；
- 在 entry metadata 中标记 `migrated_from_v1=True`。

### 10.2 旧 CompactionEntry

旧条目只有 `through_chat_id` 时：

- 尝试通过 message metadata 的 chat id 找到第一个保留节点；
- 生成内存中的 `first_kept_entry_id`；
- 不原地修改旧 JSONL；
- 后续新压缩写入新版 Entry。

### 10.3 Session 版本

Session Header 版本从 1 升到 2：

```json
{"type":"session","version":2}
```

读取器同时支持 v1/v2；写入器只写 v2。

### 10.4 SQLite

短期不迁移或删除 `chat_log`。新版 SessionRecorder 持久化 JSONL，成功 Trace 完成后由 ChatProjector 继续写 SQLite，保证 Memory 和 Consolidation 不回归。

---

## 11. 实施批次

建议拆为六个可独立验收的批次：

| 批次 | 内容 | 不应同时做的事 |
|---|---|---|
| A | 严格 AI Message + AgentMessage | 不改 Session Schema |
| B | 新 SessionEntry + SessionRecorder | 不改 Compaction 算法 |
| C | 树路径 Compaction | 不改 Provider |
| D | Skills 懒加载 + 输出路径 | 不改 Loop |
| E | `stream_simple` 生产接线 | 不改 Session |
| F | 类型归属、abort、renderer、CLI、文档 | 不引入新产品能力 |

每个批次都必须满足：

1. 对应章节测试通过；
2. `tests/test_loop.py` 通过；
3. `tests/test_architecture.py` 通过；
4. 完整离线测试通过；
5. `git diff --check` 通过；
6. 不调用真实付费 API。

## 12. 总体验收场景

最终使用一个脚本化模型验证以下完整流程：

```text
1. 用户提出读取并修改文件的任务
2. Turn 1 产生 read_file tool call
3. ToolResult 进入消息、事件和 Session Tree
4. Turn 2 产生 write_file tool call
5. ToolResult 进入消息、事件和 Session Tree
6. Turn 3 生成最终回答
7. 切换模型和 thinking，形成状态节点
8. 创建 A 分支并执行若干消息
9. 回退后创建 B 分支
10. B 分支触发 Compaction
11. 摘要不包含 A 分支内容
12. 重启进程
13. 从 JSONL 恢复 B 分支的消息、模型和 thinking
14. 下一次模型请求只包含标准 Message
15. Provider 请求中不存在 details、terminate 或产品字段
```

## 13. 完成定义

只有同时满足以下条件，才能称为第 6–10 章架构完成：

- AgentMessage 与 AI Message 在类型和运行时转换上都有明确边界；
- 一次工具 Turn 的所有消息独立进入 Session Tree；
- JSONL 能独立恢复当前分支；
- 模型和 thinking 能随树回退；
- Compaction 只读取当前树路径；
- `first_kept_entry_id` 决定压缩后的上下文视图；
- Skills 默认使用清单 + read 懒加载；
- 主 Agent 真实经过 `stream_simple → registry → translator`；
- 所有 Provider 请求不包含 Agent/产品内部字段；
- 章节测试、Loop 测试和完整离线回归全部通过。

## 14. 第一项实施任务

第一项任务严格限制在阶段一：

```text
1. 新建 ai/messages.py
2. 定义标准内容块与 Message 联合类型
3. 将 AIContext.messages 改成 list[Message]
4. 将 AgentMessage 改成 Message + CustomMessage 联合类型
5. 分离 TransformContext 与 ConvertToLlm
6. 将 default_convert_to_llm 改成穷举白名单转换
7. 更新两个 Provider translator
8. 增加字段泄漏、非法 custom translator、thinking signature 测试
```

该任务完成并验证后，才进入 Session Tree 改造。
