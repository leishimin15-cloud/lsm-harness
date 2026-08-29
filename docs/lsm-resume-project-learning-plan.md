# LSM Agent 简历项目与 Pi 架构学习计划

> 文档用途：在时间有限的情况下，控制项目范围、补齐核心理解，并形成可以演示和讲解的简历项目。  
> 项目定位：受 Pi 架构启发、结合个人记忆能力和 Waku 风格可观测界面的本地 Coding Agent Harness。  
> 执行原则：先理解并稳定核心闭环，再补记忆和 Web；不以逐项复刻 Pi 为目标。

## 1. 当前问题

LSM Agent 已经积累了 Agent Loop、Provider、工具、事件、Session、Memory、RAG、Web 和 Subagent 等代码，但当前存在三个风险：

1. 功能扩张速度超过对架构的理解速度；
2. 同时追求 Pi 完整对齐、记忆系统和 Web 产品，范围过大；
3. 自动化测试通过，但项目作者还不能稳定解释核心数据流和设计取舍。

因此，下一阶段不再以“继续增加功能”为主要目标，而以以下结果为目标：

```text
能运行
→ 能追踪
→ 能解释
→ 能演示
→ 能写进简历并应对追问
```

## 2. 项目统一定位

推荐项目名称与描述：

> LSM Agent 是一个受 Pi 架构启发的本地 Coding Agent Harness。它通过 Provider 中立的流式模型接口驱动多 Turn Agent Loop，使用受控文件与 Shell 工具完成编码任务，并通过 Session、滚动摘要和长期事实记忆维持跨运行上下文；CLI、TUI 和 Waku 风格 Web 控制台共享同一个 Harness Core。

定位边界：

- Pi 是架构和核心机制参考；
- Waku 是个人记忆思路和桌面 Web 交互参考；
- LSM Agent 是独立的 Python 实现和产品组合；
- 不在简历中声称完整复刻 Pi；
- 可以明确说明与 Pi 不同的简化设计及其原因。

## 3. 简历版 MVP 范围

### 3.1 必须完成：Coding Agent 核心闭环

系统必须真实完成：

```text
用户编码任务
→ 模型生成 Tool Call
→ read_file / write_file / exec
→ Tool Result 返回模型
→ 模型继续运行
→ 最终回答
```

验收场景：

```text
用户要求读取一个 Python 文件、定位一个确定性问题、修改文件并运行指定测试；
系统至少形成三个 Turn，并能在 Trace 中看到模型、工具和最终回答的完整顺序。
```

### 3.2 必须掌握：三层架构

```text
ai
  Model、标准消息、Provider translator、流事件

agent
  Agent Loop、AgentMessage、AgentTool、typed event、运行策略

coding_agent
  Session、上下文组装、具体工具、CLI/TUI/Web 产品接线
```

架构要求：

- `ai` 不依赖 `agent` 或 `coding_agent`；
- `agent` 不依赖 `coding_agent`、Memory 或具体产品工具；
- `coding_agent` 负责组装模型、Loop、Session、Memory 和工具；
- CLI、TUI、Web 不能各自实现一套 Agent Loop。

### 3.3 必须完成：最小记忆系统

简历主线只保留三种记忆：

| 类型 | 内容 | 主要作用 |
|---|---|---|
| Working Memory | 当前 Session 的近期原始消息 | 保持短期连续性 |
| Rolling Summary | 较早历史的结构化摘要 | 延长长会话 |
| Semantic Memory | SQLite 中的长期事实 | 跨 Session 找回稳定信息 |

以下能力属于可选扩展，不作为 MVP 阻塞项：

- Episodic Memory；
- Procedural Memory；
- 向量 RAG；
- 自动记忆冲突消解；
- 复杂记忆评分和遗忘算法。

### 3.4 必须完成：基本 Session 持久化

最低要求：

- 重启后能恢复会话；
- 能区分 user、assistant、tool call 和 tool result；
- 能查看当前 Session 的消息路径；
- 压缩摘要不会删除原始历史；
- Session 与长期 Memory 的职责明确分开。

完整的九种 Entry、assistant 切割点、turnPrefix 和复杂分支摘要属于架构增强项，不阻塞简历 MVP。

### 3.5 必须完成：可观测 Web 第一版

Waku 风格 Web 第一版只包含：

```text
Chat
Trace 时间线
Tool Call 与结果
Session 列表和恢复
Memory 只读查看
```

所有入口共享：

```text
CLI ─┐
TUI ─┼→ Harness.respond() → 同一 Agent Core
Web ─┘
```

第一版 Web 不做：

- 公网部署；
- 多用户权限；
- Graph Workflow 编辑器；
- 任意数据库管理；
- 另一套独立 Agent Loop；
- 为展示而重复实现核心状态。

## 4. 立即冻结的范围

当前 Pi 消息与 Session 改造完成后，暂停新增以下内容：

- 新 Provider；
- 新 MCP Server；
- 更多工具种类；
- Graph Workflow；
- 更复杂的多 Agent 编排；
- Web 公网和多用户能力；
- 新的记忆分类；
- 与简历演示无关的 UI 页面。

任何新需求进入待办列表，只有在 MVP 验收后才能开始。

## 5. 自动改造代码的验收门

当前外部编码 Agent 完成第 6–10 章改造后，不立即继续开发，先执行以下审查：

1. 查看完整 `git diff`，列出修改文件；
2. 检查是否破坏 `coding_agent → agent → ai` 依赖方向；
3. 检查第 3、5、7 章稳定行为是否保留；
4. 运行章节定向测试；
5. 运行 Agent Loop、Provider、Session 和 Memory 回归；
6. 用脚本模型执行一次真实多 Turn 工具链；
7. 用自己的话解释每一个重要修改；
8. 无法解释的抽象暂不作为简历亮点；
9. 审查通过前不继续增加新功能。

审查结论分为：

```text
保留：设计正确且能够解释
简化：功能正确，但不需要保持 Pi 的完整复杂度
返工：破坏边界、无法恢复或存在真实数据错误
删除候选：没有进入生产链路的装饰性抽象
```

## 6. 必须掌握的六个核心判断

学习结束后，必须能不用文档解释：

1. 模型只产生下一步决策，不会自己执行工具或再次调用自己；
2. Agent Loop 负责反复调用模型、执行工具并返回观察结果；
3. AgentMessage 是系统内部完整事实，标准 Message 是给模型的投影；
4. Event 让状态、CLI、Trace 和 Session 观察同一次运行；
5. Context Engineering 决定每次模型调用看到哪些规则、历史和工具结果；
6. Session 保存对话过程，Memory 提炼未来仍值得使用的信息。

如果无法解释其中任何一项，应返回对应章节学习，而不是通过继续写代码绕开。

## 7. 六次学习计划

每次学习采用相同流程：

```text
文档概念
→ 手动画调用链
→ 对照 LSM 源码
→ 运行最小测试
→ 自己复述
→ 记录仍不理解的问题
```

### 第一次：第 1、2 章——系统身份和三层骨架

学习目标：

- 分清 Model、Agent、Harness、Coding Agent 和 TUI；
- 理解 Pi 的三种身份；
- 理解 `coding_agent → agent → ai`；
- 理解上层类型为什么比下层类型更丰富；
- 能把 LSM 目录映射到三层结构。

代码阅读：

```text
src/lsm_harness/ai/
src/lsm_harness/agent/
src/lsm_harness/coding_agent/
tests/test_architecture.py
```

练习：

- 将 `Model`、`AgentLoopConfig`、`ToolDefinition`、`SessionEntry`、CLI renderer 分配到正确层；
- 画出 LSM 的依赖图；
- 找出一个“导入方向正确但概念放错层”的例子。

学习产出：

- 一张三层架构图；
- 一段两分钟项目架构口述。

### 第二次：第 3 章——Agent Loop

学习目标：

- 区分 Trace、Turn 和模型调用；
- 理解内层工具循环和外层 follow-up 循环；
- 理解 `tool_calls / stop / length / error / aborted`；
- 理解 steering、follow-up、prepareNextTurn 和 shouldStopAfterTurn；
- 理解一个 Turn 为什么必须包括完整工具批次。

代码阅读：

```text
src/lsm_harness/agent/agent_loop.py
src/lsm_harness/agent/runtime.py
src/lsm_harness/agent/pending.py
tests/test_loop.py
```

练习：

- 手工画出“模型先读文件、再写文件、最后回答”的三个 Turn；
- 分别写出五种 StopReason 的退出行为；
- 解释 steering 和 follow-up 的时间位置。

学习产出：

- 一张 Trace/Turn 嵌套图；
- 能逐行口述 Agent Loop 主干，而不是背代码。

### 第三次：第 4、5 章——模型与工具

学习目标：

- 理解 Model、AIContext、StreamOptions 和 StreamFunction；
- 理解 Provider registry 和 translator；
- 理解十二种 AssistantMessageEvent；
- 理解 Tool、AgentTool、ToolDefinition；
- 理解工具五步管道、错误消息化和批次调度。

代码阅读：

```text
src/lsm_harness/ai/types.py
src/lsm_harness/ai/stream.py
src/lsm_harness/ai/registry.py
src/lsm_harness/ai/api/
src/lsm_harness/agent/tools.py
src/lsm_harness/coding_agent/tools.py
```

练习：

- 跟踪一个 `read_file` 从 ToolDefinition 到 Provider Schema；
- 跟踪模型返回的 Tool Call 怎样进入五步执行管道；
- 解释为什么只有 execute 阶段并行；
- 解释错误为什么要作为 ToolResult 返回模型。

学习产出：

- 一张 Model → Translator → Event → Loop 图；
- 一张 ToolDefinition → AgentTool → Tool 图。

### 第四次：第 6、7 章——消息与事件

学习目标：

- 理解 AgentMessage 与标准 Message；
- 区分 transformContext 和 convertToLlm；
- 理解 content blocks 和 thinking signature；
- 理解十种 Agent 内核事件；
- 理解 EventSink 的同步屏障和状态所有权。

代码阅读：

```text
src/lsm_harness/ai/messages.py
src/lsm_harness/agent/messages.py
src/lsm_harness/agent/events.py
src/lsm_harness/coding_agent/messages.py
```

练习：

- 跟踪一条 ToolResult 怎样进入 Agent 状态、模型请求、Trace、UI 和 Session；
- 解释为什么 Session 保存 AgentMessage，而不是 Provider JSON；
- 解释为什么状态必须在 listener 之前更新。

学习产出：

- 一张“完整事实 → 模型投影 → Provider 协议”图；
- 一张四层事件生命周期图。

### 第五次：第 8、9、10 章——长期运行

学习目标：

- 理解工具截断和完整输出逃生通道；
- 理解项目规则递归和 Skills 懒加载；
- 理解压缩红线、切割点、摘要和 CompactionEntry；
- 理解 Session Tree、append-only、leaf、回退和分支；
- 理解 Session 与 Memory 的区别。

代码阅读：

```text
src/lsm_harness/tools/truncate.py
src/lsm_harness/coding_agent/resources.py
src/lsm_harness/coding_agent/compaction.py
src/lsm_harness/coding_agent/session.py
src/lsm_harness/ops/session_store.py
```

练习：

- 手工选择一组消息的合法压缩切割点；
- 画出 A/B 两个会话分支；
- 说明为什么 tool result 不能作为保留区起点；
- 说明为什么压缩必须只读取当前树路径；
- 说明 Working Memory、Summary 和 Semantic Memory 的数据来源。

学习产出：

- 一张 Context 漏斗图；
- 一张 Session Tree 和 Compaction 关系图。

### 第六次：简历、演示与面试

学习目标：

- 用三分钟介绍项目；
- 用十分钟演示真实 Coding Agent 任务；
- 讲清楚一个技术难点和一个设计取舍；
- 区分已经实现、简化实现和未来工作；
- 应对面试官的追问。

学习产出：

- 简历项目描述；
- 三分钟项目介绍稿；
- 一套确定性演示脚本；
- 架构图；
- 面试问答清单。

## 8. 时间紧张时的压缩版本

如果只能投入较短时间，按以下优先级：

| 优先级 | 内容 | 目标 |
|---|---|---|
| P0 | 第 1–3 章 | 能解释架构和 Loop |
| P0 | 第 5–7 章 | 能解释工具、消息、事件 |
| P1 | 第 4 章 | 能解释 Provider 统一方式 |
| P1 | 第 8–10 章主干 | 能解释上下文、压缩和 Session |
| P2 | Pi 完整边角语义 | 作为后续深入内容 |

可以暂缓深入：

- 每个 Provider 的全部缓存参数；
- Pi 的全部 SessionEntry 类型；
- turnPrefix 的全部边界条件；
- 完整扩展系统；
- 多 Agent 调度细节。

不能跳过：

- 三层边界；
- Trace 与 Turn；
- 工具调用闭环；
- AgentMessage 与 Message；
- EventSink；
- Session 与 Memory 的区别。

## 9. 项目实施顺序

### 阶段 A：冻结并审查当前改造

- 等待当前消息/Session 改造结束；
- 审查 diff 和真实调用链；
- 修复回归；
- 不增加新功能。

完成标准：第 3、5、7 章行为不回归，一次多 Turn 工具请求可追踪和恢复。

### 阶段 B：完成最小记忆闭环

```text
成功对话
→ chat/session projection
→ 结构化摘要
→ 长期事实提取
→ 下一次相关问题检索
→ 注入上下文
```

完成标准：保存一个稳定事实，重启进程后能够通过检索找回，并在 Trace 中看到检索证据。

### 阶段 C：稳定 CLI 演示

- 一条命令启动；
- 流式回答；
- 工具进度可见；
- Stop 可用；
- Session 可恢复；
- 错误有清晰提示。

完成标准：确定性脚本和一次真实模型演示均能完成。

### 阶段 D：完成 Web 可观测第一版

- Web 复用 Harness Core；
- Chat 支持流式响应；
- 展示 Trace、Turn、Tool；
- 查看 Session 和 Memory；
- 保持本机单用户边界。

完成标准：同一个任务可从 CLI 或 Web 运行，核心 Trace 结构一致。

### 阶段 E：简历材料

- 更新 README；
- 画最终架构图；
- 准备演示数据；
- 写简历项目描述；
- 记录性能、测试数量和已知限制；
- 准备面试追问。

## 10. 简历演示脚本

建议固定使用一个小型、安全、可重复的项目：

```text
1. 用户要求定位一个测试失败；
2. Agent 读取测试和实现文件；
3. Agent 调用受控 Shell 运行单测；
4. Agent 修改一个文件；
5. Agent 再次运行单测；
6. Agent 返回修改结论；
7. Web 展示三个 Turn、工具结果和文件变化；
8. 退出并重启；
9. 恢复 Session；
10. 询问之前修改了什么，展示 Session/Memory 恢复。
```

演示不得依赖：

- 不稳定网络搜索；
- 大型仓库；
- 随机生成结果；
- 外部服务写入；
- 无法重复的多 Agent 行为。

## 11. 面试时必须能解释的设计取舍

至少准备以下问题：

1. 为什么不把所有逻辑写进一个 `respond()`？
2. 为什么需要 AgentMessage 和标准 Message 两层？
3. 为什么 Tool 错误返回模型，而不是抛出 Loop？
4. 为什么只有工具 execute 阶段并行？
5. 为什么 EventSink 先更新状态再通知 listener？
6. 为什么 Session 和 Memory 不是同一个概念？
7. 为什么长会话需要摘要而不是简单删除旧消息？
8. 为什么 Web 必须复用 Harness Core？
9. 项目在哪些地方参考了 Pi，哪些地方是自己的实现？
10. 为了按时完成，你主动放弃了哪些复杂设计？

## 12. 每日工作约束

每次开发开始前写明：

```text
今天只解决什么？
它属于哪一章、哪一层？
完成标准是什么？
哪些内容明确不做？
```

每次开发结束后记录：

```text
修改了哪些文件？
真实调用链发生了什么变化？
运行了哪些测试？
我能否不看代码解释这次修改？
是否增加了新的范围？
```

如果无法解释本次修改，下一次工作优先学习和复盘，不继续叠加功能。

## 13. 最终完成定义

项目可以进入简历，需要同时满足：

- 核心编码任务闭环稳定；
- 三层依赖清楚并有测试；
- 至少一种真实 Provider 可运行；
- 文件与 Shell 工具有工作区和副作用边界；
- Session 可以恢复；
- Working Memory、Rolling Summary、Semantic Memory 能形成最小闭环；
- CLI 可稳定演示；
- Web 能展示同一 Harness 的 Chat 和 Trace；
- 完整离线回归通过；
- README 与实际代码一致；
- 项目作者能解释完整调用链和关键取舍；
- 简历只描述实际验证过的能力。

## 14. 当前下一步

当前只执行两件事：

1. 按顺序学习第 1、2 章，先建立三层架构心智模型；
2. 等待当前自动改造结束后，冻结代码并进行一次完整审查。

在这两件事完成前，不启动新的 Memory 或 Web 大改。
