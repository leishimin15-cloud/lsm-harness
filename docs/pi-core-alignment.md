# Pi 核心对齐状态

当前运行依赖方向固定为：

```text
CodingSession → Agent → Agent Loop → AI Provider
       ↓           ↓
 Session Tree   AgentState
```

`Harness` 是 `CodingSession` 的兼容名称，旧 CLI/RPC 调用方无需迁移。

## 已对齐

- Agent 持有 `AgentState`、双队列、活动运行、取消信号和持久事件归约器。
- `prompt()`、`continue_()`、`reset()`、`signal`、queue mode 与 Pi Agent
  的职责一致。
- AssistantMessage 持久化 model/provider/api/usage/stop reason/error/timestamp。
- 失败运行走 `message_start → message_end → turn_end → agent_end`。
- JSONL Session Tree 是会话事实源；SQLite 是检索和记忆投影。
- Tool history 会在进入 Provider 前规范化：补缺失结果、移动乱序结果、删除
  orphan/duplicate 结果，并输出修复计数；重启时仍会把末尾中断结果持久化。
- Provider 请求快照还会执行 Pi 风格的 `transform_messages`：非视觉模型图片
  降级、跨模型 thinking/signature 转换、跨模型 tool call ID 规范化、跳过
  error/aborted AssistantMessage，并为孤立工具调用补临时错误结果。持久历史不被改写。
- JSONL 只自动修复最后一条 torn record；中间损坏明确抛出
  `SessionFileError`，不会再静默跳过。首次写入原子发布，普通追加会 fsync。
- Agent 自持 `transform_context`、`convert_to_llm`、payload/response hook、
  thinking budgets、transport 和最大重试延迟；每次调用进入 `StreamOptions`。
- CodingSession 提供 Agent + queue/retry/error/state/settled/compaction/
  entry-appended 的统一 typed event 流；model/client/stream 的组合被抽到
  `ModelRuntime`。
- 工具调用从 preflight 开始就形成成对的 `tool_execution_start/end`；进度事件
  携带 args，end 事件携带经过 after hook 的最终结果。并行批次按完成顺序发
  end，写入模型上下文时仍保持原调用顺序。
- 模型中断会在内核事件和当次 AgentState 中保留已经收到的 partial
  AssistantMessage；产品 JSONL 仍让分支停在未回答的 user/tool 节点，以支持
  `respond_continue()`。重试退避可被取消，且 OpenAI/Anthropic SDK 的内建重试
  关闭，避免与 Agent 外层重试叠加。
- Usage 包含 uncached input、output、cache read/write、total tokens；模型目录
  提供费率时会计算分项费用，未知费率保持 0，不猜价格。
- Eval 的脚本输入与断言分离，golden 会被读取并比较。

## 有意保留的 Python 差异

- 运行模型是同步生成器和线程池，而不是 TypeScript Promise/async iterator。
- 取消使用 `threading.Event`，不是 `AbortSignal`。
- 工具集合使用 `ToolRegistry`，不是数组；它仍遵守 AI Tool、AgentTool、
  ToolDefinition 三层依赖方向。
- CLI 和 TUI 订阅 `CodingSession.subscribe()` 的 typed 事件；RPC 同时输出新的
  `session_event` typed 通道和旧 `event` 兼容通道。Trace 的旧字符串事件仍作为
  兼容投影保留，不再是产品界面的主要状态源。

## 暂不处理

- TUI 的完整分支树可视化和更丰富的队列编辑体验。
- 真实付费模型 Eval；离线 CI 不需要 API key。
