# Tau TUI 边界对照(阶段 5,2026-09-08)

阶段 5 把 `gateway/tui.py`(单文件 332 行)按 Tau(`~/tau-main`,
`tau_coding/tui/`)的适配边界重组为 `gateway/tui/` 包。原则不变:
**Tau 只提供 Python/Textual 的组织方式参考;同步生成器 + 线程池 +
threading.Event 的运行模型保留**,不抄 Tau 的全 asyncio 结构。

## 1. 分层映射

| Tau | LSM | 职责 |
|---|---|---|
| `tui/state.py` (921 行) | `tui/state.py` | 纯显示状态,零 Textual 导入;`TuiState` 是 transcript 事实来源,`render_lines()` 全量渲染(纯测用) |
| `tui/adapter.py` (157 行) | `tui/adapter.py` | 事件→状态翻译;持有 `TurnProjection`(阶段 3),`feed()` 返回增量行(纯测用;App 只消费归约后的 state) |
| `tui/app.py` (7846 行) | `tui/app.py` | Textual 壳:组装、事件分发、条目控件同步、worker 线程、生命周期 |
| — (内联在 app.py) | `tui/commands.py` | 斜杠命令 + /model /sessions /tree 选择器回调(CommandsMixin) |
| — (内联在 app.py) | `tui/screens.py` | 通用 PickerScreen(ModalScreen + OptionList) |
| — (内联在 app.py) | `tui/bindings.py` | 按键绑定表 APP_BINDINGS |
| `tui/widgets.py` (2871 行) | `tui/widgets.py` | 纯渲染:MessageWidget / ToolCallWidget / TranscriptView / StatusBar / QueueBar |
| `tests/test_tui_adapter.py` | `tests/test_tui_state.py` | state/adapter 纯测(无 Textual) |
| — | `tests/test_tui_widgets.py` | 控件纯渲染测(无 running App) |
| `tests/test_tui_app.py` (Pilot + FakeSession) | `tests/test_tui_app.py` | 真实 App + 假模型(QueueClient),`asyncio.run()` 包 `run_test()`,不加 pytest 插件 |

规模差距是有意的:Tau 的 TranscriptView 窗口化/MarkdownStream/扩展桥
等属于它的扩展系统与富渲染。阶段二(typed-event 改造)后本项目用「每条
transcript 一个条目控件」替代 RichLog + 清屏重放:assistant 流式
原位重渲染、工具单击折叠。阶段三又补齐三件:assistant 正文 Markdown
渲染(节流重解析)、TranscriptView 窗口化(挂载上限 + 顶部边界占位)、
@文件/斜杠命令 autocomplete(autocomplete.py + ChatInput)。

## 2. 有意分歧

- **线程模型**:Tau 全 asyncio 单循环(无 call_from_thread);我们保留
  worker 线程跑 `Harness.respond`,**所有** UI 更新经 `call_from_thread`
  编组回 UI 线程(TuiState 只在 UI 线程变更,免锁)。修了旧实现的两个
  真 bug:observer 直写 RichLog(线程不安全)、默认 conn
  `check_same_thread=True` 被 worker 跨线程使用(现以
  `check_same_thread=False` 创建,`sqlite3.threadsafety == 3` 钉住)。
- **流式**:Tau 的 AssistantMessageWidget 逐 token 挂载;我们保留
  AssistantWidget 的 Markdown 控件,App 在每个 `MessageUpdateEvent` 把
  ``MessageView`` 的 text 同步进去,**节流**调 ``Markdown.update``
  (全量重解析 O(n),~80ms 节流;落定强制一次)——增量即时可见
  (阶段二批 ④ 真流式 + 阶段三 markdown)。
- **工具折叠**:Tau 每条目一个 widget 可原地折叠;阶段二后本项目同样
  「每条 ToolView 一个 ToolCallWidget」——单击折叠该项(`collapsed`),
  Ctrl+O 全局折叠(`show_tool_results`),两者叠加;折叠规则:**工具只留
  调用行、隐藏结果预览,错误永远显示**(与 `state.render_lines()` 同规则,
  测试锁定)。不再需要 RichLog 的清屏 + 全量重放。
- **状态栏**:补 retry(`retry n/m`)、compaction(`compacting…`)与累积
  usage(`Σ ↑in ↓out`)显示(阶段二批 ⑥)。
- **键位**:Escape=中断(`harness.abort()`),Ctrl+C=退出(`run_tui`
  finally → `shutdown()`:abort + wait_for_idle + close);Tau 是
  Esc=cancel / Ctrl+D=quit / Ctrl+C=清输入。
- **运行中输入**:Enter=steer(Tau 同款),`/follow <文本>` 排队
  follow-up(Tau 用 Alt+Enter,单行 Input 没有该键位,改用命令)。
- **选择器**:统一 `PickerScreen`(ModalScreen + OptionList),
  /model、/sessions、/tree 共用;`/tree` 列当前路径条目,选中即
  `session.branch`(Pi navigateTree 语义:只换消息,不动模型)。

## 3. 验收

`tests/test_tui_state.py`(state/adapter 纯测)+ `tests/test_tui_widgets.py`
(控件纯渲染)+ `tests/test_tui_app.py`(Pilot 冒烟,假模型,不调付费 API):
一轮问答/真流式/中断/退出清理/steer/follow-up/三个选择器/Ctrl+O 全局折叠
/单击单项折叠/状态栏 retry+compaction+usage。每批反向变异均如期 FAIL。
