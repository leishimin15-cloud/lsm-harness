# Pi 第八章上下文工程对照

这份文档把教学版第八章"上下文工程"逐项对应到 LSM Harness。第八章的核心
观点是：上下文压缩只是冰山一角，Pi 在**输入侧**（工具输出截断、系统提示
词组装）和**历史侧**（Compaction、分支摘要）各布置了两层防线，组合成一
道漏斗。本章改造落地了输入侧的两层；历史侧分别留给第九章（Compaction 对
齐）和第十章（会话树 + 分支摘要）。

## 1. 四层防线的落地状态

| 层 | Pi 机制 | LSM Harness 现状 | 本章动作 |
|---|---|---|---|
| ① 工具输出截断 | `truncateHead` / `truncateTail` 双限制 | 新增 `tools/truncate.py` | **落地** |
| ② 系统提示词组装 | CLAUDE.md 向上递归 + XML 包装 + Skills 懒加载 | 新增 `coding_agent/resources.py`；Skills 维持记忆匹配推送 | **落地（Skills 有意分歧）** |
| ③ Compaction | 阈值触发 + 切割点 + 结构化摘要 | 已有自建滚动摘要（`session.py`） | 第 9 章对齐 |
| ④ 分支摘要 | LCA + 5 section 摘要 | 无会话树，不适用 | 第 10 章再说 |

## 2. ① 工具输出截断：`tools/truncate.py`

### 2.1 四件套对照

| Pi 设计 | 实现 | 位置 |
|---|---|---|
| 双限制（2000 行 / 50KB，先触者胜） | `DEFAULT_MAX_LINES` / `DEFAULT_MAX_BYTES` | `truncate.py:17-19` |
| 双向策略（read 留头 / bash 留尾） | `truncate_head` / `truncate_tail` | `truncate.py:83,119` |
| UTF-8 边界安全（不切开多字节字符） | `_take_from_start` / `_take_from_end` 逐字符累加字节 | `truncate.py:46-70` |
| 单行超限兜底（不能返回空） | `first_line_partial` / `last_line_partial` 标志 | `truncate.py:100-105` |
| 截断元信息（诚实告知模型） | `TruncationResult`（truncated_by、total/output lines/bytes） | `truncate.py:22-34` |
| grep 单行限长 500 + 标记 | `truncate_line`（`... [truncated]` 后缀） | `truncate.py:160-164` |

Python 的 `str` 切片本身按码点切、不会切碎 UTF-8，所以边界安全的重点
不在"切"，而在**预算按字节计**——`len(ch.encode("utf-8"))` 逐字符累加，
一个 4 字节 emoji 要么完整保留要么完全不要（测试
`test_truncation_never_splits_multibyte_characters` 锁定）。

### 2.2 exec（shell）：方向反转 + 逃生通道

旧实现按 16000 **字符**切**头部**——方向与 bash 语义相反（错误堆栈在末
尾），且无字节限制、无完整输出落盘。新实现（`shell.py:_format_output`）：

1. stdout + stderr 合并后过 `truncate_tail`（2000 行 / 50KB 先到为准）；
2. 截断时完整输出写入 `tempfile`（`lsm-exec-*.log`），末尾追加逃生提示：

   ```text
   [Showing lines 1001–3000 of 3000. Full output: /tmp/lsm-exec-xxx.log]
   ```

   这行字进 LLM 上下文——模型想看全量可自行 `read_file`；
3. 单行就超 50KB 时走 `last_line_partial` 分支，提示
   `[Showing last 49.5KB of line 1 (line is 92.3KB). Full output: ...]`；
4. 工具描述写明契约："输出截断为最后 2000 行或 50KB（先到为准）；截断
   时完整输出会保存到临时文件"——与 Pi 的 `bash.ts` 描述一致，"last"
   是契约的一部分。

与 Pi 的差异：Pi 有 `OutputAccumulator` 做流式累积期间的内存控制；我们
的 `ShellOperations.run` 是一次性 `subprocess.run` 收集，流式累积属于
工程细节，教程也未展开，暂不引入。

### 2.3 read_file / grep

- `read_file` 保留 offset/limit 分页，新增**字节上限**：选中行段过
  `truncate_head`（防 minified 文件几行就撑爆预算），截断时头部注明
  `[Truncated by bytes: showing N of M requested lines. Use offset/limit
  to read further.]`；
- **顺带修复第五章遗留问题**：offset 越界不再静默回退到最后一页，返回
  具体错误 `Error: offset N is beyond end of file (M lines total).`（教
  程第五章正面示例）；
- `grep` 单行从裸 `[:200]` 改为 `truncate_line`（500 字符 +
  `... [truncated]` 标记），与 Pi 的 `GREP_MAX_LINE_LENGTH` 对齐。

## 3. ② 系统提示词组装：`coding_agent/resources.py`

### 3.1 多层 CLAUDE.md 向上递归

`load_project_context_files(cwd, agent_dir)` 按 Pi 的 `resource-loader.ts`
语义收集，顺序**从外到内**：

```text
1. agent_dir/CLAUDE.md   ← 全局（用户级，对应 ~/.pi/）
2. 祖先目录              ← 从 / 到 cwd 上一层
3. cwd                   ← 当前项目，最具体，放最后
```

每目录按 `CLAUDE.md → AGENTS.md → claude.md → agents.md` 顺序取第一个命
中；去重（agent_dir 与 cwd 相同时不重复收集）；单文件上限 64KB，空内容
跳过。`format_project_context` 用 XML 包装：

```xml
<project_context>

Project-specific instructions and guidelines:

<project_instructions path="/org/CLAUDE.md">
...
</project_instructions>

</project_context>
```

XML 的理由与 Pi 相同：边界明确 + `path` 属性让模型区分规范层级。接入点
在 `Session.build_system`——插在 SOUL/回复规范之后、时间/模型元数据之
前（对应 Pi 骨架中 project_context 在 Current date 之前的位置）。

### 3.2 Skills：有意维持推模式（分歧记录）

Pi 的 Skills 是文件系统里的 `SKILL.md`：prompt 只放清单（name +
description + location），模型按需 read——**拉模式**。LSM 的 skills 是
记忆系统产物（`memory.matching_skills` 按用户消息匹配后推送全文），创建
路径、存储、召回方式都不同。强改拉模式要重写 memory/skills 架构，超出本
章范围，**记为有意分歧**：当前推模式只注入"命中的"skill，开销已经受
控；若未来 skill 数量膨胀，再考虑清单化 + read 按需加载。

## 4. 与教程全景链路的对应

教程 §七的完整链路，本章后的覆盖情况：

```text
[1] 系统提示词组装          ✅ 本章（resources.py）
[6] 工具输出截断            ✅ 本章（truncate.py，exec/read/grep 接线）
[9] shouldCompact 判定      ◐ 已有自建实现，第 9 章对齐 findCutPoint 等
[10] Branch Summarization   ✗ 依赖会话树，第 10 章
```

## 5. 测试

`tests/test_chapter8_context.py` 新增 19 个测试：

- 截断算法 8 个：双限制两个方向、单行超限两个方向、UTF-8 不切字符、
  未超限时元信息为零、grep 标记；
- 工具接线 7 个：exec 尾部保留 + 落盘 + 提示行、小输出原样、单行超限提
  示、read offset 越界报错、read 字节上限、行号格式保持、grep 长行截
  断；
- 上下文文件 4 个：全局→祖先→cwd 顺序与去重、缺失/空文件跳过、XML 包
  装、`build_system` 集成（位置在"当前时间"之前）。

全量 208 个测试通过（189 → 208）。
