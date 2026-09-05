# Pi 第四章模型调用对照

这份文档把教学版第四章“模型调用”逐项对应到 LSM Harness。本章改造的
核心目标是：**Agent Loop 只依赖统一事件流，不再理解 OpenAI 或 Anthropic
SDK 的返回结构。** 第三章已经完成的 Trace、Turn、内外循环和工具批次语义
保持不变。

## 1. 第四章位于三层架构的哪一层

```text
coding_agent（产品层）
        │ 选择 Model、组装 AgentContext / AgentLoopConfig
        ▼
agent（循环层）
        │ 调用 StreamFunction，只消费统一事件
        ▼
ai（模型层）
        ├─ Model / AIContext / StreamOptions
        ├─ stream / stream_simple
        ├─ API translator registry
        └─ OpenAI-compatible / Anthropic translators
```

第四章主要落在 `src/lsm_harness/ai/`。`agent/agent_loop.py` 只负责把统一
AI 事件转换成一个 Turn 的 assistant message，再根据 `stop_reason` 决定
继续工具循环、正常结束、恢复长度溢出、失败或中断。

## 2. 三个调用输入

| 教学版概念 | LSM Harness | 作用 |
|---|---|---|
| `Model` | `ai/types.py::Model` | 模型 ID、API 方言、Provider、能力声明 |
| `Context` | `ai/types.py::AIContext` | system prompt、消息和工具 schema |
| `StreamOptions` | `ai/types.py::StreamOptions` | token 上限、thinking、cache、重试和中断 |

产品层不再只把一个模型名称字符串交给循环，而是传入完整 `Model`。因此
“调用哪个模型”和“用哪一种 API 格式调用”已经分开。

## 3. StreamFunction 是 Agent 与 AI 的边界

统一签名是：

```python
StreamFunction = Callable[
    [Model, AIContext, StreamOptions],
    Iterator[AssistantMessageEvent],
]
```

`run_agent_loop()` 接收 `stream_fn`，不接收某一家 SDK client。旧的
`ModelClient.complete()` / `stream_complete()` 仍可通过
`client_stream_function()` 适配，供现有 Memory、RAG、测试和外部调用逐步迁移。

这条兼容门面只在边界处存在；新的 Agent Loop 主路径已经是第四章接口。

## 4. 十二种统一流事件

`AssistantMessageEvent` 定义了十二种事件：

| 生命周期 | 事件 |
|---|---|
| 响应 | `start`、`done`、`error` |
| 正文 | `text_start`、`text_delta`、`text_end` |
| 思考 | `thinking_start`、`thinking_delta`、`thinking_end` |
| 工具 | `toolcall_start`、`toolcall_delta`、`toolcall_end` |

每个事件都带有 `partial: ModelResponse`，它是截至当前事件的完整快照。消费方
可以显示增量，也可以随时读取当前全文、thinking、工具调用、usage 和
`stop_reason`，不需要自己理解 Provider 的 chunk 类型。

典型工具调用事件序列：

```text
start
→ thinking_start → thinking_delta* → thinking_end
→ text_start → text_delta* → text_end
→ toolcall_start → toolcall_delta* → toolcall_end
→ done
```

正文、thinking 和工具调用可以交错；只有 `done` 或 `error` 是终止事件。

## 5. API 翻译器注册表

`ai/registry.py` 按 API 方言注册 `ApiProvider`。模型通过 `Model.api` 选择翻译器，
不是在 Agent Loop 里写 Provider 条件分支。

当前实际实现两个方言：

| API 方言 | 翻译器 | 覆盖的服务 |
|---|---|---|
| `openai-completions` | `ai/api/openai_compat.py` | OpenAI、DeepSeek、Gemini OpenAI 兼容端点、OpenRouter、xAI |
| `anthropic-messages` | `ai/api/anthropic_messages.py` | Anthropic、Kimi、GLM、MiniMax 的 Anthropic 兼容端点 |

新增同方言 Provider 只需增加模型/服务元数据；新增真正不同的协议，才需要注册
新的翻译器。

## 6. 两个完整请求与响应翻译器

### OpenAI-compatible

请求翻译负责：

- 把 system、messages 和中立工具 schema 转为 Chat Completions 格式；
- 映射 `reasoning_effort` 或 DeepSeek thinking；
- 在模型声明支持时映射 prompt cache；
- 使用模型的 `base_url` 连接兼容服务。

响应翻译负责正文、reasoning、并行工具调用参数、usage 和 finish reason 的
增量组装。

### Anthropic Messages

请求翻译负责：

- 把 assistant 的 `tool_calls` 转成 `tool_use` block；
- 把 OpenAI 风格的 `role="tool"` 转成 user `tool_result` block；
- 合并 Anthropic 要求的相邻同角色消息；
- 映射 thinking budget 和 cache control；
- 转换文本和图片内容块。

响应翻译直接消费 Anthropic 原始流事件，因此工具调用不会再被 `text_stream`
丢失。

## 7. stream 与 stream_simple

- `stream()`：直接选择注册的翻译器，适合已经准备好精确能力参数的调用方。
- `stream_simple()`：先根据 `Model` 能力降级参数，再进行安全重试。

能力降级包括：

- `max_tokens` 不超过模型声明上限；
- Anthropic thinking 开启时，为 thinking budget 之外至少保留 1024 个输出 token；
- 不支持所选 thinking 档位时，选择最近的可用档位；
- 不支持长期 cache 时将 `long` 降为 `short`；
- 完全不支持 cache 时降为 `none`。

重试只允许发生在正文、thinking 或工具参数尚未开始输出之前。这样可以避免
已经向用户展示半段结果后再次请求，造成重复文本或重复工具调用。

## 8. Thinking、Cache 与 Overflow

### Thinking

统一档位是：

```text
off → minimal → low → medium → high → xhigh
```

产品层原有 `disabled`、`enabled`、`auto` 仍可使用：`disabled` 映射 `off`，
`enabled` 映射 `high`，`auto` 根据任务复杂度在二者间选择。最终由模型能力表
翻译成 reasoning effort、DeepSeek 开关或 Anthropic token budget。

### Cache

统一策略是 `none`、`short`、`long`。是否真正发送 cache 参数由 `Model` 的
能力声明决定，不支持的兼容服务不会收到错误的 Anthropic/OpenAI 参数。

### Overflow

AI 层识别显式 context-window 错误以及无输出的 length 响应，并归一为
`stop_reason="length"`。Agent 层继续使用第三章已有的 `on_truncation`：压缩
上下文后开启下一个 Turn，并受 `max_length_recoveries` 限制。

## 9. 错误为何是事件而不是异常

翻译器把 SDK 异常分类为：

```text
arrearage | rate_limit | transient | permanent | aborted
```

并输出 `error` 事件。限流和瞬时错误可以在无可见输出时重试；欠费、永久错误和
已开始输出后的错误直接交还 Agent Loop。这样正常完成与失败走同一条流协议，
Trace 也能记录完整生命周期。

## 10. 与第三章如何衔接

```text
一个 Turn 开始
→ Agent 调用 StreamFunction
→ AI translator 产生统一事件
→ Agent 组装 assistant message
→ tool_calls：执行工具批次，继续内循环
→ stop：结束内循环，外循环检查 follow-up
→ length：压缩上下文后恢复
→ error：Trace failed
→ aborted：Trace aborted
```

第四章替换的是“一个 Turn 内怎样调用模型”，没有改变第三章的 Turn 定义、
tool batch、steering、follow-up 或终止规则。

## 11. 建议学习顺序

1. `ai/types.py`：先看三个输入和十二种事件。
2. `ai/registry.py`、`ai/stream.py`：理解注册、能力降级与重试边界。
3. `ai/api/openai_compat.py`：看一种熟悉协议如何翻译成统一事件。
4. `ai/api/anthropic_messages.py`：对比不同消息和工具协议。
5. `agent/agent_loop.py::_consume_assistant_stream`：看 Agent 如何只消费中立事件。
6. `coding_agent/app.py`：看产品层如何选择模型并注入 StreamFunction。
7. `tests/test_ai_stream.py`、`tests/test_architecture.py`：用确定性测试核对每条规则。

## 12. 当前有意保留的范围边界

- 目前只有 OpenAI-compatible 和 Anthropic Messages 两个原生翻译器。
- Gemini 原生协议、Bedrock、OpenAI Responses API 尚未实现；现有 Gemini 走
  官方 OpenAI-compatible endpoint。
- Memory 与 RAG 的辅助模型调用仍可使用旧 `ModelClient.complete()`；主 Agent
  Loop 已经迁移到 StreamFunction。兼容门面后续可以按模块逐步收窄。
