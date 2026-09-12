# Pi 第九+十章对照（Compaction 对齐 + 会话树）

这份文档把教学版第九章（Compaction）和第十章（Session Tree）一起对应到
LSM Harness。两章合在一起做的原因：第九章的"选择性跳过"（压缩后如何重
建上下文）天然依赖第十章的会话树遍历——分开落地会写出一次性中间态。
第八章留下的历史侧两层防线（③ Compaction、④ 分支摘要）至此全部落地。

## 1. 第九章：Compaction 对齐

### 1.1 算法抽离：`coding_agent/compaction.py`

旧实现把判定、切割、摘要全揉在 `Session._do_compress` 里。本章把**纯决
策**抽成独立模块（Session 只管持久化和 LLM 调用），与 Pi 的
`compaction.ts` 对齐：

| Pi 设计 | 实现 | 位置 |
|---|---|---|
| `shouldCompact`：tokens > contextWindow − reserveTokens | `should_compact(context, budget, reserve)` | `compaction.py:38` |
| 默认 keepRecentTokens 20K | `Settings.context_keep_recent_tokens`（默认 6000，我们的预算整体小一个量级） | `config.py` |
| `findCutPoint` 倒序累积 token | `find_cut_point(rows, keep_recent_tokens, estimate)` | `compaction.py:47` |
| 切割点合法校验（toolResult 不可切） | user 边界切割（见 1.2 分歧） | 同上 |
| `<read-files>`/`<modified-files>` 追踪 | `extract_file_operations` / `parse_file_tags` / `format_file_operations` / `merge_file_lists` | `compaction.py:80-143` |

红线语义完全一致：`estimated > budget − reserve` 才触发。Session 侧
`context_compression_tokens` 扮演触发线，`reserve = budget − trigger`
（`session.py:_compress_if_due`）。

### 1.2 find_cut_point：token 预算 + user 边界（含边界案例）

算法与 Pi 相同：从最新行**倒序**累积 `4 + estimate(content)`，达到
keep_recent_tokens 即停——新历史比旧历史值钱，保哪段由 token 预算决定，
不再由"轮数"决定（旧的 `context_recent_turns` 退役）。

切割点调整是有意分歧：

- **Pi** 允许切在 assistant 行（配合 `turnPrefix` 把被切开的半个 turn
  前缀补回上下文）；
- **我们**的 chat_log 把一个 turn 融合成 user/assistant 一对（工具调用
  内嵌在 assistant 记录里），切开 turn 在结构上不可能，所以切割点
  **一律前移到最近的 user 边界**，turn 永远完整。

前移带来两个边界案例，都有测试锁定：

1. 前移后保留 token 可能略低于预算（Pi 切 assistant 不会）——语义可接
   受，差额小于一个 turn（`test_find_cut_point_keeps_recent_token_budget`）；
2. 极小预算下停点落在最后一个 turn 的 assistant 行，前移会跑出数组—
   此时**回退钳制到最后一个 user 行**，保住最后完整 turn；若连一个完
   整 turn 都容不下（`cut == 0`），返回 0 表示"无可压缩"，`_do_compress`
   直接放弃（`test_find_cut_point_never_splits_a_pair`）。

### 1.3 六段摘要模板 + 增量合并

`SUMMARY_SECTIONS` 从自建 7 段换成 Pi 的 6 段：**Goal / Constraints &
Preferences / Progress / Key Decisions / Next Steps / Critical Context**，
其中 Progress 固定三个子标题 `### Done / ### In Progress / ### Blocked`
——分类存放让模型续写时不用猜"做完的"和"卡住的"。

增量路径保留：已有摘要时用 `UPDATE_SUMMARY_PROMPT`（带上旧摘要要求合
并），首次用 `SUMMARY_PROMPT`。

### 1.4 文件操作追踪（跨压缩累积）

Pi 在摘要尾部追加 `<read-files>` / `<modified-files>` 标签，让"读过什
么、改过什么"在反复压缩中不丢失。落地方式：

1. `extract_file_operations(entries, from_chat_id, to_chat_id)`：扫 JSONL
   树中 chat_id 落在压缩区间的消息条目，从 `tool_calls` 的参数里收集
   `read_file → read-files`、`write_file → modified-files`；
2. 新一轮压缩先 `parse_file_tags(旧摘要)` 取回历史清单，再
   `merge_file_lists` 做有序并集——**上一轮压缩覆盖的文件不会因为原
   始消息被压掉而消失**；
3. 合并结果 `format_file_operations` 追加在摘要末尾，同时以结构化字段
   存进 `CompactionEntry.read_files / modified_files`（机器可读）。

集成测试 `test_compaction_appends_and_accumulates_file_tags` 验证：第二
轮压缩的摘要里，a.py（来自旧摘要标签）和 c.py（本轮新范围）同时在列。

### 1.5 摘要注入位置迁移：system prompt → 首条 custom 消息

旧实现把摘要拼进 system prompt（"当前会话历史摘要：..."）。本章按 Pi
改为**首条消息**注入，落地第六+七章留下的尾巴：

- 新增 `compaction_summary_message(summary, through_chat_id=..., version=...)`
  （`coding_agent/messages.py`），`role="custom"`、
  `custom_type="compaction_summary"`；
- `to_llm` 翻译成 user 消息，带 Pi 的 preamble：

  ```text
  The conversation history before this point was compacted into the following summary:

  <summary>
  ...
  </summary>
  ```

- `prepare_context` 的预算丢弃循环**保护首条摘要**：只有 `body`（摘要
  之后的消息）参与按对丢弃，摘要永远在。

理由与 Pi 相同：摘要是"对话的一部分"而不是"行为规范"，放 system 里既
污染提示词又让预算控制变复杂。

## 2. 第十章：会话树

### 2.1 核心观点：存储介质 ≠ 数据结构

教程强调：JSONL 文件是**存储介质**，树是**数据结构**，两者正交。LSM
以 JSONL Session Tree 作为权威会话记录；`chat_log`（SQLite）只是供检索、
记忆整合与旧版兼容使用的派生投影。上下文构建走当前树路径，只有旧会话
尚未回填 JSONL 时才回退 `chat_log`。

树原语新增在 `ops/session_store.py`：

```python
build_by_id(entries)                                # id → entry
path_to_leaf(entries, leaf_id)                      # leaf→root 沿 parent 走，反转
collect_abandoned_branch(entries, old_leaf, new_leaf)  # 旧路径减新路径（LCA 不含）
```

**追加式 + 认父不认子**：条目只追加、带 `parent_id` 单指针；分支就是移
动 `_last_entry_id`（O(1)），被弃分支一行不删——这正是"分支便宜"的根
本原因。

### 2.2 树上下文构建 + 压缩的选择性跳过

`_tree_context_messages()` 沿 leaf→root 路径按类型 dispatch：

| 条目类型 | 处理 |
|---|---|
| `message` | 进入上下文（带 `meta.chat_id` 供压缩比对） |
| `compaction` | **选择性跳过**：`chat_id <= through_chat_id` 的消息全部滤除；被更新的压缩取代的旧 `compaction_summary` custom 消息也滤除；`branch_summary` 保留；最后在头部插入最新摘要消息 |
| `branch_summary` | 追加 `branch_summary_message`（翻译时带分支 preamble） |

Pi 用 `firstKeptEntryId` 做选择性跳过；我们的切割点天然落在 user 边界，
用 chat_log 自增 id（`through_chat_id`）做区间判定，语义等价且更直观。
`MessageEntry.meta.chat_id` 是这次为压缩比对专门写入的。

### 2.3 一次性回填迁移

`_backfill_jsonl_from_chat_log()`：JSONL 里没有任何 message 条目但
chat_log 有记录时（老会话），把 chat_log 行逐条镜像成链式
`MessageEntry`（`meta={"chat_id": ..., "backfilled": True}`），再补
`session_summaries` 里已有的 `CompactionEntry`。幂等——重建 Session 不
会重复回填（`test_backfill_mirrors_chat_log_into_jsonl` 锁定）。

### 2.4 branch / branch_with_summary

- `Session.branch(entry_ref, emit)`：支持精确 id 或唯一前缀（`_resolve_
  entry_ref`），移动 leaf、按新路径重建 `self.history`，发
  `session.branched`；
- `Session.branch_with_summary(entry_ref, emit)`：
  1. `collect_abandoned_branch` 收集被弃分支（旧 leaf → LCA，不含 LCA）；
  2. 用 `BRANCH_SUMMARY_PROMPT`（5 段，比压缩模板少 Critical Context）
     让 LLM 摘要，`max_tokens=2048`（Pi 同款上限）；
  3. 追加文件操作标签（被弃分支里读过/改过什么不能丢）；
  4. 写 `BranchSummaryEntry`（`parent=分叉点`、`from_id=被弃 leaf`——
     挂在分叉点上，指向被弃分支）；
  5. 发 `session.branch_summary.started/completed/failed` + `session.branched`；
  6. **LLM 失败则整体不动**：leaf 保持原位，返回 None
     （`test_branch_with_summary_failure_keeps_state`）。

分支摘要进上下文时的 preamble（Pi 原文）：

```text
The user explored a different conversation branch before returning here.
Summary of that exploration:
```

## 3. 有意分歧与已知限制（汇总）

1. **turnPrefix 不适用**：chat_log 融合 turn，切割点必在 user 边界，Pi
   的"切在 assistant 再补前缀"分支结构性不存在；
2. **chat_log 保持线性**:chat_log 是 SQLite 查询投影,永远按插入
   顺序线性追加;权威读取一律走树。`_do_compress` 的压缩段已按树路
   径取(可靠性批次 1 修复,段起点 = 上一轮 `first_kept_entry_id`);
   `history` 展示历史自阶段 4 批 1 起也从树当前路径重建
   (`_history_from_tree`),chat_log 仅作 legacy 回退;
3. **事件命名**:`context.compression.*` 保留(TUI 已在消费),树操作
   新增 `session.branched` / `session.branch_summary.*`;
4. **Skills 推模式**(第八章已记):不重复展开。

## 3.1 阶段 4 对齐:运行状态恢复时机与重启恢复(2026-09-07)

以本地 Pi 为准核对后的行为(变更处为行为对齐,非纠错):

- **会话 open/switch 即时恢复**:Pi `createAgentSession` 在打开会话时
  用 `buildSessionContext` 的 model/thinkingLevel 恢复运行时。对齐前
  我们在 respond 前懒恢复(状态栏在切换后到首次运行前显示旧模型);
  现在 `switch_session` 与 Harness 启动路径即时恢复,失败兜底保留当
  前模型并记录 `session.runtime_state_restore_failed`(类比 Pi 的
  modelFallbackMessage)。
- **会话内 branch 不改模型**:Pi `navigateTree` 只替换
  `agent.state.messages`,不重新派生模型/thinking。对齐前懒恢复会在
  branch 后的下一次 respond 把模型回退到路径基线;现在 branch 只更
  新消息,用户显式切换的模型保持不变(取代 code-review issue 二的
  旧契约,测试已同步改写)。
- **重启恢复 = 文件末行**:Pi `_buildIndex` 把 leaf 重置为文件最后一
  个条目,分支选择本身不持久化;branch 后追加过新消息,重启才落在
  新分支。我们与之完全一致,由
  `test_restart_without_new_messages_lands_on_file_last_line` 钉住,
  防止以后误把分支指针持久化。

## 4. 测试

新增两个文件 16 个测试；更新 `test_context.py`、`test_chapter6_messages.py`
配合行为迁移（轮数保留 → token 保留、system 注入 → 消息注入、7 段 → 6
段）：

- `test_chapter9_compaction.py`（8 个）：红线判定、token 预算切割、永
  不拆 turn（含极小预算钳制）、全保留返回 0、文件标签抽取/往返/合并、
  跨压缩累积集成、摘要消息注入位置；
- `test_chapter10_session_tree.py`（8 个）：path_to_leaf 排除被弃分支、
  collect_abandoned_branch 止于 LCA、branch 移 leaf 重建历史 + 被弃分
  支不可见、未知 ref 拒绝、branch_with_summary 落条目 + preamble 翻译、
  LLM 失败状态不变、树上下文的选择性跳过、回填迁移幂等。

全量 225 个测试通过（208 → 225）。
