# LSM 的个人 Harness：功能与架构说明

> 文档版本：v1.0  
> 项目定位：中文优先、状态本地保存、核心链路可读可调试的个人 Agent Harness

## 1. 这个项目是什么

LSM Harness 不是一个简单的“DeepSeek 聊天窗口”，而是一套包在大语言模型外面的 Agent 运行系统。

大语言模型负责理解问题、生成回答和决定是否调用工具；Harness 负责给模型准备上下文、管理记忆、执行工具、限制副作用、保存数据并记录整个运行过程。

可以把两者的关系理解为：

- **DeepSeek 是大脑**：负责推理与决策。
- **Harness 是身体和运行环境**：负责记忆、行动、状态、边界和可观测性。
- **CLI Gateway 是入口**：负责接收用户输入、展示过程与输出。

当前 v1.0 的目标，是把下面这条最小但真实的闭环完整跑通：

```text
用户输入
  → 上下文与记忆组装
  → DeepSeek 决策
  → 本地工具执行
  → 工具结果返回 DeepSeek
  → 最终回答
  → 对话与长期记忆落盘
  → 全过程写入 Trace
```

这个项目的重点不是功能数量，而是证明你理解一个 Agent Harness 最核心的内部机制，并且能够自己实现、调试和解释这些机制。

## 2. 当前完成了哪些能力

| 能力 | 当前实现 | 作用 |
| --- | --- | --- |
| Gateway | 交互式终端 CLI | 接收输入并展示回答、记忆决策和工具状态 |
| 模型接入 | DeepSeek OpenAI-compatible API | 使用真实模型完成回答和工具决策 |
| Agent Loop | `reason → act → observe` 多轮循环 | 让模型能够连续调用工具，而不只生成一次文本 |
| Working Memory | 最近 12 轮会话 | 维持当前对话的短期连贯性 |
| Semantic Memory | SQLite `facts` | 保存用户、项目、偏好等长期事实 |
| Episodic Memory | SQLite `episodes` | 保存一段时间内发生过的事情 |
| Procedural Memory | `SOUL.md` 与 `SKILL.md` | 定义 Agent 的行为原则和做事方法 |
| 原始对话 | SQLite `chat_log` | 为上下文恢复和记忆整合保留原始材料 |
| RAG | Gate + FTS5 trigram + LIKE + 中文相关性排序 | 按需从长期记忆中找回相关内容 |
| Consolidation | 每 6 次完整对话自动触发 | 将原始对话提炼成 facts 和 episode |
| 本地工具 | 日历、笔记、记忆、Soul、Skill | 让 Agent 能够执行可验证的本地动作 |
| 副作用策略 | 只允许 `read` 和 `local_write` | 防止第一阶段误操作外部系统 |
| Trace | 统一事件结构写入 JSONL | 复盘每轮模型调用、工具调用和异常 |
| 自检 | `doctor` 与 `smoke` | 分别检查环境和验证核心链路 |

## 3. 整体架构

```mermaid
flowchart TD
    U["用户"] --> CLI["CLI Gateway"]
    CLI --> H["Harness.respond"]

    H --> S["Session / Working Memory"]
    S --> G{"Retrieval Gate"}
    G -->|skip| C["组装模型上下文"]
    G -->|retrieve| R["RAG 检索"]
    R --> F["Facts / Episodes / Skills"]
    F --> C

    C --> L["Agent Loop"]
    L --> M["DeepSeek 主模型"]
    M -->|自然语言回答| O["最终输出"]
    M -->|Tool Calls| T["Tool Registry"]
    T --> P{"副作用策略"}
    P -->|允许| X["执行本地工具"]
    P -->|阻止| E["错误结果"]
    X --> L
    E --> L

    O --> W["保存 chat_log"]
    W --> K{"达到整合阈值？"}
    K -->|是| D["Consolidation"]
    D --> F

    H -. "统一事件" .-> J["JSONL Trace"]
    L -. "统一事件" .-> J
    T -. "统一事件" .-> J
```

项目按职责拆分，而不是把所有逻辑写在一个聊天函数里：

```text
src/lsm_harness/
├── gateway/          # CLI 输入与展示
├── loop/             # Agent Loop
├── memory/           # 检索、存储、整合、Skills
├── tools/            # 工具及副作用策略
├── ops/              # Trace
├── app.py            # Harness 组合入口
├── models.py         # DeepSeek 模型适配器
├── runtime.py        # Session 与上下文组装
├── config.py         # 配置读取
└── db.py             # SQLite Schema 与连接
```

## 4. 一次请求具体发生了什么

假设用户输入：

```text
我现在正在开发什么项目？
```

系统会依次完成以下工作。

### 4.1 Gateway 接收输入

CLI 只负责入口和展示，它不会直接把字符串裸传给 DeepSeek。它调用公共入口：

```python
Harness.respond(user_message)
```

这样以后即使增加 Web、桌面端或 API Gateway，它们也能复用同一个 Harness Core。

### 4.2 创建 Turn 和 Trace

系统为本轮请求生成唯一的 `turn_id`，首先记录 `turn.started`。后续的模型调用、工具调用和最终结果都使用同一个 `turn_id`，因此可以完整还原一轮请求。

### 4.3 构造上下文

Session 会组合：

1. `SOUL.md` 中的行为与人格设定；
2. 匹配到的本地 Skills；
3. 最近 12 轮 Working Memory；
4. Retrieval Gate 找回的长期记忆；
5. 当前用户消息。

这一步是 Harness 的重要价值：模型的效果不仅取决于模型本身，也取决于给它什么上下文、按照什么顺序给、哪些内容应该省略。

### 4.4 Retrieval Gate 判断是否需要记忆

`deepseek-v4-flash` 先判断当前问题属于：

- `skip`：不需要长期记忆，例如普通数学题；
- `retrieve`：涉及用户、项目、偏好或过去事件，需要检索。

Gate 的目的，是减少无意义检索产生的噪声。如果 Gate 调用失败，系统会 **fail-open**，默认执行检索，以免因为辅助模型异常而漏掉重要记忆。

### 4.5 Agent Loop 驱动主模型

组装完成后，`deepseek-v4-pro` 进入 Agent Loop。

```text
调用模型
  ├─ 没有 Tool Call → 返回最终回答
  └─ 有 Tool Call
       → 校验工具与参数
       → 执行工具
       → 把工具结果作为 tool message 返回模型
       → 再次调用模型
```

循环最多执行 10 次。达到上限后会强制退出，防止模型陷入无限工具调用。

工具报错不会直接让进程崩溃。错误会以工具结果的形式返回模型，让模型有机会修正参数、选择其他工具，或者向用户解释失败原因。

### 4.6 保存对话并尝试整合

最终回答产生后，用户消息和助手回答会写入 `chat_log`。当累计达到默认的 6 次完整对话时，Consolidation 会尝试把尚未整合的原始对话提炼为长期记忆。

最后，系统导出可读的 `MEMORY.md`，并记录 `turn.completed`。

## 5. 中立模型接口为什么重要

Harness Core 没有直接依赖 DeepSeek SDK 的具体返回结构，而是依赖中立接口：

```python
ModelClient.complete(...) -> ModelResponse
```

`ModelResponse` 统一包含：

- 文本回答；
- `ToolCall[]`；
- 停止原因；
- Token 用量。

这意味着：

- 当前可以使用 DeepSeek；
- 测试时可以替换为脚本化模型，不产生 API 费用；
- 未来可以增加 OpenAI、Anthropic 或本地模型适配器；
- Agent Loop 不需要因模型供应商变化而重写。

当前模型分工如下：

| 模型 | 职责 | 原因 |
| --- | --- | --- |
| `deepseek-v4-pro` | 主回答、工具选择、多轮 Agent Loop | 负责核心理解与决策 |
| `deepseek-v4-flash` | Retrieval Gate、Consolidation | 任务较小，强调速度与成本 |

第一阶段显式关闭 Thinking Mode，避免工具多轮调用中额外处理 `reasoning_content`，先保证 Harness 主链路稳定。

## 6. 记忆系统

这个项目不只是“三个记忆文件”，而是不同生命周期、不同职责的几类状态。

### 6.1 Working Memory：当前会话记忆

Working Memory 默认保存最近 12 轮对话，直接进入主模型上下文。

它解决的是“刚才说过什么”，特点是：

- 容量有限；
- 与当前会话强相关；
- `/new` 会开启新的 Working Memory 会话；
- 不适合无限保存历史信息。

### 6.2 Semantic Memory：事实记忆

Semantic Memory 保存可以反复使用的稳定事实，例如：

```text
用户正在开发 LSM Harness。
用户偏好使用简体中文。
```

这些数据存储在 SQLite 的 `facts` 表。`save_note` 可以显式保存事实，Consolidation 也可以从对话中自动提炼事实。

### 6.3 Episodic Memory：事件记忆

Episodic Memory 保存“发生过什么”，例如：

```text
用户在本轮讨论中确认先跑通本地 Harness Core，再考虑全栈界面。
```

它和事实记忆的区别是：事实描述长期状态，episode 描述一段具体经历或决策过程。

### 6.4 Procedural Memory：做事方式

Procedural Memory 由两部分组成：

- `SOUL.md`：Agent 的身份、风格、行为原则；
- `skills/<name>/SKILL.md`：针对特定任务的步骤、知识和约束。

它保存的不是“用户是谁”或“过去发生了什么”，而是“Agent 应该如何做事”。`update_soul` 和 `create_skill` 可以在运行时更新这部分能力。

### 6.5 `chat_log`：原始材料

`chat_log` 不是已经提炼好的长期记忆，而是原始对话记录。它承担两个作用：

- 为当前或后续会话保留原始输入输出；
- 为 Consolidation 提供尚未加工的材料。

这类似于：`chat_log` 是原始笔记，facts 和 episodes 是整理后的知识。

## 7. 记忆怎样被写入

目前有两条写入路径。

### 7.1 显式记忆

当用户明确说“记住……”时，主模型可以调用 `save_note`，立即把信息写入 `facts`。

```text
用户：记住，我正在开发 LSM Harness。
模型：调用 save_note
工具：写入 facts
模型：确认已经记住
```

这条路径适合明确、重要、应该马上持久化的信息。

### 7.2 自动整合

每达到 6 次完整对话，Flash 模型会读取尚未整合的 `chat_log`，生成：

- 若干条 facts；
- 一条 episode。

只有成功解析并写入长期记忆后，原始对话才会标记为已整合。若模型调用或解析失败，原始对话仍保持未整合状态，等待后续重试，不会静默丢失。

## 8. RAG 在这里怎样工作

本项目确实使用了 RAG，但第一阶段不是“向量数据库 RAG”，而是基于本地 SQLite 的文本检索 RAG。

RAG 的完整过程是：

```text
用户问题
  → Retrieval Gate 判断是否需要历史信息
  → 从 facts / episodes 中检索候选内容
  → 对中文候选结果做相关性排序
  → 取 Top K 注入上下文
  → 主模型基于当前问题和历史记忆回答
```

中文检索使用三层策略：

1. **FTS5 trigram**：适合长度至少为 3 的连续文本匹配；
2. **参数化 `LIKE` 回退**：处理两字姓名、短词或 FTS 无结果；
3. **中文二元组相关性排序**：处理用户问法与原记忆不完全连续的情况。

例如事实原文是：

```text
我正在开发 LSM Harness。
```

用户后来问：

```text
我当前开发的项目是什么？
```

两句话语义相关，但没有足够长的连续相同片段。仅依赖 trigram 或整句 `LIKE` 容易漏检。当前实现会拆出中文二元组并按重合度排序，因此可以找回这条事实。

需要注意：检索只负责找候选信息，不等于模型必然正确使用信息。最终回答仍由主模型结合上下文生成，这也是未来需要 Evaluation / LLM Ops 的原因。

## 9. Agent Loop 与工具系统

### 9.1 当前工具

| 工具 | 作用 | 写入位置 |
| --- | --- | --- |
| `create_event` | 创建本地日历事件 | `calendar_events` + `calendar.ics` |
| `list_events` | 查询本地日历事件 | 只读 SQLite |
| `save_note` | 保存一条长期事实 | `facts` + `MEMORY.md` |
| `manage_memory` | 查询、更新或删除记忆 | `facts` / `episodes` |
| `update_soul` | 更新 Agent 行为原则 | `SOUL.md` |
| `create_skill` | 创建本地 Skill | `skills/<name>/SKILL.md` |

Tool Registry 统一负责：

- 向模型暴露工具 Schema；
- 根据工具名找到实现；
- 校验必填参数；
- 捕获未知工具和执行异常；
- 检查副作用是否被允许。

### 9.2 副作用边界

工具被分为：

- `read`：只读取状态；
- `local_write`：只修改 `.lsm/` 下的本地状态；
- `external_write`：修改外部服务或真实应用。

v1.0 只允许前两类。因此当前创建日历事件并不会直接修改 macOS Calendar，也不会发送消息、运行 Shell 或调用外部写入 API。

这个限制是产品设计的一部分：先让 Agent 的内部闭环可验证，再逐步开放需要授权、幂等、回滚与安全确认的外部动作。

## 10. 本地数据保存在哪里

所有运行状态默认保存在项目根目录的 `.lsm/`：

```text
.lsm/
├── state.db          # SQLite 主状态库
├── SOUL.md           # Agent 行为原则
├── MEMORY.md         # 方便人阅读的长期记忆导出
├── calendar.ics      # 本地 iCalendar 文件
├── skills/           # 本地 Skills
└── traces/           # 每日 JSONL Trace
```

SQLite 中当前主要有以下表：

| 表 | 内容 |
| --- | --- |
| `facts` | Semantic Memory |
| `facts_fts` | facts 的 FTS5 trigram 索引 |
| `episodes` | Episodic Memory |
| `episodes_fts` | episodes 的 FTS5 trigram 索引 |
| `chat_log` | 用户与助手的原始对话 |
| `calendar_events` | Agent 创建的本地日历事件 |

`.lsm/calendar.ics` 是标准 iCalendar 文件，可以被日历软件打开或导入，但它不是 macOS Calendar 数据库。当前项目不会自动把事件写进 Mac 系统日历。

## 11. Trace 与可观测性

所有运行事件统一使用：

```json
{
  "type": "tool.completed",
  "turn_id": "同一轮请求的唯一 ID",
  "timestamp": "事件时间",
  "data": {}
}
```

同一个事件会同时提供给：

- CLI Observer：在终端显示 `memory · retrieve`、`tool · create_event → ok` 等状态；
- JSONL Tracer：写入 `.lsm/traces/<日期>.jsonl`，供排错和评估。

一轮包含工具调用的典型事件顺序是：

```text
turn.started
memory.gate
memory.retrieved
llm.completed
tool.requested
tool.completed
llm.completed
turn.completed
```

达到循环上限时还会记录 `loop.limit_reached`；未捕获异常会记录 `turn.failed`。

Trace 的价值在于把“Agent 为什么这样回答”从黑盒变成可以追踪的问题：它检索了什么、模型调用了几次、选择了哪个工具、参数是什么、工具是否成功、消耗了多少 Token。

## 12. 三个公共入口

### 12.1 `lsm`

启动真实 DeepSeek 交互式 CLI：

```bash
cd /Users/lsm/Desktop/lsm-harness
source .venv/bin/activate
.venv/bin/lsm
```

内置命令：

- `/memory`：查看当前 facts 与 episodes；
- `/new`：开启新的 Working Memory 会话；
- `/quit`：退出程序，长期记忆不会删除。

macOS 上可能已经存在另一个同名系统命令 `lsm`。如果终端命中了错误程序，使用 `.venv/bin/lsm` 最可靠；激活虚拟环境后也可以执行 `rehash` 再尝试 `lsm`。

### 12.2 `lsm doctor`

检查：

- Python 版本；
- SQLite FTS5；
- trigram tokenizer；
- 配置项；
- `DEEPSEEK_API_KEY` 是否存在。

它不会调用 DeepSeek，因此不会产生模型费用。

```bash
.venv/bin/lsm doctor
```

### 12.3 `lsm smoke`

使用脚本化模型验证 Agent Loop、工具、记忆和 Trace，不需要 API Key，也不产生模型费用。

```bash
.venv/bin/lsm smoke
```

Smoke Test 的意义是区分两类故障：如果 smoke 成功、真实聊天失败，问题通常在 API、网络或模型配置；如果 smoke 也失败，问题更可能在本地 Harness Core。

## 13. 三个实际使用示例

### 13.1 保存并跨重启找回事实

```text
你：记住，我正在开发 LSM Harness。
LSM：调用 save_note，并确认保存。

退出后重新运行：

你：我当前开发的项目是什么？
LSM：Retrieval Gate 选择 retrieve，RAG 找到事实并回答。
```

这里验证了模型工具调用、SQLite 写入、进程重启后的持久化和 RAG 找回四个环节。

### 13.2 创建本地日历事件

```text
你：帮我添加今天上午 10 点到 11 点的 Harness 优化计划。
LSM：调用 create_event。
```

事件会进入 `state.db` 和 `calendar.ics`。相同标题和开始时间具有幂等保护，避免模型重复调用时产生重复事件。

### 13.3 创建做事方法

```text
你：创建一个 code-review skill，要求先检查正确性，再检查安全性，最后给修改建议。
LSM：调用 create_skill。
```

Skill 写入本地后会即时刷新。后续匹配到相关任务时，它会作为 Procedural Memory 进入上下文。

## 14. 已解决的关键工程问题

### 14.1 `lsm` 命令名冲突

macOS 环境中存在另一个名为 `lsm` 的命令，导致最初执行 `lsm doctor` 时启动了错误程序。当前运行方式已明确使用虚拟环境中的 `.venv/bin/lsm`；后续可以考虑将正式命令改名为 `lsm-harness`，彻底避免冲突。

### 14.2 中文输入第一个字无法删除

最初 CLI 使用普通终端输入方案，中文输入法与行编辑状态组合时出现首字符无法删除的问题。当前已改用 `prompt_toolkit.PromptSession`，由专门的交互式输入组件处理光标、退格和中文输入。

### 14.3 中文记忆“明明存了却检索不到”

原始实现过度依赖连续 trigram 匹配。事实“我正在开发 LSM Harness”和问题“当前开发项目是什么”词序和连接词不同，因此可能返回空结果。当前已加入中文二元组相关性排序，并增加了对应回归测试。

## 15. 测试覆盖和当前验收状态

目前自动化测试共 **21 项**，覆盖：

- 纯文本回答；
- 单工具与多轮工具调用；
- 未知工具；
- 工具异常；
- 最大迭代限制；
- 中文记忆跨重启检索；
- 两字短词 `LIKE` 回退；
- 非连续中文项目问法检索；
- Retrieval Gate 的 `skip`、`retrieve` 和 fail-open；
- Consolidation 成功与解析失败；
- Skill 创建、匹配、重名拒绝和即时刷新；
- 本地日历写入与重复事件幂等；
- 记忆更新和删除；
- Trace 顺序和敏感信息保护。

真实 DeepSeek 普通问答、工具调用和本地日历写入也已经跑通。

## 16. 当前明确没有做什么

v1.0 暂时不包含：

- Web 或桌面 Dashboard；
- Graph Workflow；
- MCP；
- 子 Agent；
- 向量数据库和 Embedding；
- 自动写入 Apple Calendar、Google Calendar 等真实外部服务；
- Shell 工具和外部消息发送；
- LLM-as-Judge 与完整 LLM Ops 平台。

这不代表这些功能没有价值，而是当前阶段优先验证最核心的 Harness 闭环。只有先能可靠解释模型怎样获得上下文、怎样调用工具、怎样保存记忆、怎样追踪执行，后续的全栈界面和复杂编排才有稳定基础。

## 17. 下一阶段可以怎样发展

建议按下面的顺序扩展：

1. **稳定 CLI Core**：继续用真实对话验证时间理解、工具参数和记忆准确率；
2. **消除命令冲突**：增加 `lsm-harness` 命令，保留或逐步弃用 `lsm`；
3. **受控外部工具**：增加 Apple Calendar 适配器，并设计授权、确认和幂等机制；
4. **可视化 Dashboard**：展示对话、检索结果、工具调用、Trace 和 Token；
5. **Graph 编排**：只有出现明确的分支、并行或状态机需求时再引入；
6. **RAG 对比实验**：用同一测试集比较文本检索与向量检索；
7. **Evaluation / LLM Ops**：对检索准确率、工具成功率、回答质量、延迟和成本建立指标。

## 18. 如何用一句话介绍这个项目

> LSM 的个人 Harness 是一个中文优先的本地 Agent 运行框架：它用中立模型接口接入 DeepSeek，通过可追踪的 Agent Loop 调用受控本地工具，并用 Working、Semantic、Episodic 和 Procedural Memory 让 Agent 能够在进程重启后继续理解用户与项目。

## 19. 核心源码阅读顺序

如果想真正理解代码，建议按一次请求的执行方向阅读：

1. [`gateway/cli.py`](../src/lsm_harness/gateway/cli.py)：输入从哪里进入；
2. [`app.py`](../src/lsm_harness/app.py)：各个组件如何组装；
3. [`runtime.py`](../src/lsm_harness/runtime.py)：上下文如何构造；
4. [`memory/retrieval.py`](../src/lsm_harness/memory/retrieval.py)：Gate 怎样决定是否检索；
5. [`memory/stores.py`](../src/lsm_harness/memory/stores.py)：中文记忆怎样搜索和排序；
6. [`loop/agent.py`](../src/lsm_harness/loop/agent.py)：模型与工具怎样循环；
7. [`tools/registry.py`](../src/lsm_harness/tools/registry.py)：工具边界怎样控制；
8. [`memory/consolidation.py`](../src/lsm_harness/memory/consolidation.py)：原始对话怎样变成长时记忆；
9. [`ops/tracing.py`](../src/lsm_harness/ops/tracing.py)：整个过程怎样留下证据。

按这个顺序读完，你看到的就不再是几个孤立模块，而是一条完整的 Harness 执行链。
