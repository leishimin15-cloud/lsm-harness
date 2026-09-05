# Pi 第五章工具系统对照

这份文档把教学版第五章“工具系统”逐项对应到 LSM Harness。第五章改造的
目标不是增加更多工具，而是把“模型看见什么、Agent 怎样执行、产品怎样组装”
分开，并让每次工具调用都经过同一条可验证的管道。

## 1. 第五章位于三层架构的哪里

```text
coding_agent / tools.py
  ToolDefinition
  prompt_snippet / renderer / product_context
             │ wrap_tool_definition
             ▼
agent / tools.py
  AgentTool
  prepare / validate / hooks / execute / finalize / scheduling
             │ descriptor
             ▼
ai / types.py
  Tool(name, description, parameters)
  Provider translator 只读取这三个字段
```

依赖方向仍然是 `coding_agent → agent → ai`：

- AI 层不知道函数、权限、CLI 或执行环境；
- Agent 层不知道文件系统、Shell、记忆或产品提示词；
- Coding Agent 层把具体依赖绑定成 `ToolDefinition`，再擦除产品元数据，包装成
  `AgentTool`。

## 2. 三种 Tool 分别解决什么问题

| 层 | 类型 | 负责 | 不负责 |
|---|---|---|---|
| AI | `ai.types.Tool` | `name`、`description`、JSON Schema `parameters` | 执行、权限、UI |
| Agent | `agent.tools.AgentTool` | prepare、execute、effect、execution mode、timeout、hooks | 产品提示词、具体 UI |
| Coding Agent | `coding_agent.tools.ToolDefinition` | label、prompt snippet、CLI renderer、产品上下文和原生实现 | Provider 消息格式 |

`ToolRegistry.ai_tools()` 会把注册的 `AgentTool` 降为纯描述型 AI `Tool`。
OpenAI 和 Anthropic 翻译器只读取这个描述，不可能拿到执行函数或产品上下文。

旧代码仍可使用：

```python
from lsm_harness.tools.registry import Tool, ToolResult

Tool(name, description, input_schema, fn)
```

兼容构造器会映射到 `AgentTool`；`ToolResult` 是 `ToolResultMessage` 的别名，
不会输出运行时警告。

## 3. 五步管道怎样落到代码

对外稳定的三个阶段位于 `agent/tools.py`：

```text
prepare_tool_call
→ execute_prepared_tool_call
→ finalize_executed_tool_call
```

三个阶段内部严格执行第五章的五步顺序：

```text
prepareArguments
→ Draft 2020-12 JSON Schema 验证
→ effect / approval / beforeToolCall / tool before hook
→ execute
→ tool after hook / AgentLoopConfig.afterToolCall
→ ToolResultMessage
```

拆成三个阶段的原因是：preflight 必须顺序执行，只有 `execute` 可以并行，
finalize 又必须回到确定性顺序。

### prepareArguments

`AgentTool.prepare_arguments` 可以补默认值、规范日期或清理字符串。Schema 验证
发生在 prepare 之后，因此模型给出的可修复输入不会被过早拒绝。

### JSON Schema

验证器使用 `jsonschema.Draft202012Validator`，覆盖类型、`required`、`enum`、
数组、嵌套对象和显式 `additionalProperties`。如果 Schema 没有声明
`additionalProperties`，保持 JSON Schema 默认的允许行为。

错误包含工具名和 JSON 路径，例如：

```text
Error: invalid arguments for 'read_file': $.offset: 'x' is not of type 'integer'
Error: invalid arguments for 'tool': $.rows[0].count: 'x' is not of type 'integer'
```

### before 与 approval

验证通过后才检查副作用策略、审批和 before hooks，避免对无效参数发起审批。
产品级 `AgentLoopConfig.before_tool_call` 可以阻止调用；工具自己的 before hook
和注册表默认 hook 继续作为兼容扩展点。

### execute

工具仍是同步函数。需要并行或 timeout 时由线程池包裹，不把整个项目改成
`asyncio`。工具可选择接收 `_ctx`、`_abort` 或 `_on_update`。

### after 与字段覆盖

工具 after hook 先执行，`AgentLoopConfig.after_tool_call` 后执行。后者继续返回
`AfterToolCallResult`，只覆盖明确给出的 `output`、`details`、`is_error` 或
`terminate` 字段。

## 4. 为什么任何工具错误都不穿透 Loop

下面的失败都被转换为 `ToolResultMessage(is_error=True)`：

- 未知工具；
- prepare 异常；
- Schema 验证失败；
- approval 异常或拒绝；
- before hook 异常或阻止；
- execute 异常、超时或中断；
- tool after hook 异常；
- `AgentLoopConfig.after_tool_call` 异常。

框架增加阶段和工具名上下文，但不会把工具返回的具体错误改写成笼统的
“执行失败”。模型能看到真实错误并在下一个 Turn 修正参数；对应的离线端到端
测试会先提交错误类型，再读取 `is_error` 并自行纠正。

## 5. ToolResultMessage 信封

第五章只建立最小工具结果信封：

```python
ToolResultMessage(
    tool_call_id="call-1",
    tool_name="read_file",
    output="...",
    details={...},
    is_error=False,
    terminate=False,
)
```

- `output` 是模型主要看到的内容；
- `is_error` 告诉模型该结果是否失败；
- `details` 只进入内部消息、Trace 和 UI；
- `terminate` 控制工具批次后是否停止内循环；
- call id 和 tool name 让并行进度、Trace 与 UI 可以准确关联。

Agent Loop 写入 Trace 时记录 `tool_call_id` 和 `is_error`。Anthropic 翻译器把
错误映射到 `tool_result.is_error`；OpenAI 翻译器只保留该协议允许的
`role`、`tool_call_id` 和 `content`，移除 `details`、`terminate`、`timestamp`
等内部字段。

完整的“内部消息类型与 LLM 消息类型”双层体系属于第六章，本章没有提前改动
持久化会话格式。

## 6. 批次调度怎样对齐 Pi

并行批次固定为三段：

```text
顺序 preflight
→ 仅 execute 并行（最多 8 个线程）
→ tool.execution_end 按完成顺序产生
→ finalize 与 ToolResultMessage 按模型调用顺序返回
```

这样既允许两个只读工具真正重叠，又不会让 approval、before hook 或消息顺序
变得不确定。

两条强制串行规则：

1. 批次中任意一个 `AgentTool.execution_mode == "sequential"`，整批串行；
2. `AgentLoopConfig.tool_execution == "sequential"`，整批串行。

所有产品内置工具显式声明模式：文件读取、目录、grep、Web 搜索、RAG 查询等
为 `parallel`；文件写入、Shell、日历写入、记忆写入、MCP 和子 Agent 启动为
`sequential`。

批次中任意一个结果 `terminate=True` 就停止后续 Turn，不再要求整批结果全部
terminate。

## 7. 进度更新为什么带 call id

每个执行调用都有独立的进度闸门。有效事件包含：

```json
{
  "type": "tool.progress",
  "data": {
    "tool": "delegate_code",
    "label": "委托给 Pi",
    "tool_call_id": "call-7",
    "delta": "..."
  }
}
```

`execute` settle 后立刻关闭 `accepting_updates`。如果工具自行启动的后台线程稍后
再次调用 `_on_update`，更新会被静默丢弃，不会污染下一个 Turn。CLI 展示优先
使用 `label`，没有 label 时回退到 `name`。

## 8. Operations 如何隔离产品执行环境

`coding_agent/operations.py` 定义两组最小协议。

### 文件

- `FileOperations`：resolve、stat/read/write、list、walk、relative；
- `LocalFileOperations`：真实工作区实现并执行路径边界检查；
- `MockFileOperations`：完全内存实现，测试不会访问宿主机；
- `TrackingFileOperations`：writer decorator，在写前 snapshot、写后 record。

因此 `filesystem.py` 只处理 offset/limit、grep、大小限制和输出格式，不再把
`Path.write_text()` 与 `FileState` 写死在工具函数中。

### Shell

- `LocalShellOperations`：调用本地 `subprocess.run`；
- `SandboxShellOperations`：调用已注入的 Docker Sandbox；
- `MockShellOperations`：记录命令并返回预设结果；
- `UnavailableShellOperations`：要求沙箱但不可用时失败关闭。

`shell.py` 只负责命令解析、allow/deny 策略、timeout 范围和结果格式。选择本地
还是 Docker 是产品组装决定，不是工具执行时临时决定。

Calendar、Memory、RAG、Web 和 MCP 本章只迁移三层工具类型。它们已有 SQLite、
Memory、RAG Engine、HTTP 或 MCP Session 等服务对象，不再为了形式增加一层空
Operations。

## 9. 代码阅读顺序

1. `ai/types.py::Tool`：模型看到的最小描述。
2. `coding_agent/tools.py::ToolDefinition`：产品元数据与包装器。
3. `agent/tools.py::AgentTool`：Agent 执行协议。
4. `agent/tools.py::prepare_tool_call`：prepare、Schema 与 preflight。
5. `agent/tools.py::execute_prepared_tool_call`：timeout、abort 和进度闸门。
6. `agent/tools.py::finalize_executed_tool_call`：两级 after hooks。
7. `agent/tools.py::ToolRegistry.execute_batch`：Pi 三阶段调度。
8. `coding_agent/operations.py`：文件与 Shell 的可替换执行环境。
9. `agent/agent_loop.py::_execute_tool_calls`：信封进入消息、Trace 与下一 Turn。
10. `tests/test_chapter5_tools.py`：逐项运行本章合约。

## 10. 本章有意保留的边界

- 保持同步生成器、`ThreadPoolExecutor` 和 `threading.Event`，不改成 asyncio。
- 不修改现有会话数据格式。
- Web 只保证回归，不增加第五章专属展示；CLI 先显示 label。
- 外部旧 `Tool(...)` 调用和旧导入路径继续工作。
- 第六章再建立完整内部消息/LLM 消息双层类型；本章只做 Provider 安全转换。
