"""Built-in Eval 2.0 core scenarios.

Five deterministic scenarios cover the coding-agent lifecycle:
  1. ``read_and_answer``           — read a file, answer correctly.
  2. ``calc_bug``                  — fix a bug and pass a test.
  3. ``self_correct``              — recover from a bad tool argument.
  4. ``abort_restart_continue``    — abort → restart → continue.
  5. ``compaction_remembers_goal`` — remember a goal across compaction.

Each scenario scripts the model explicitly (the script is an independent
fixture, never derived from the assertions) and asserts on observable
effects: tool calls, workspace files, session tree, and the final answer.
"""

from __future__ import annotations

from lsm_harness.ai.types import ModelResponse, ToolCall, Usage

from lsm_harness.ops.eval.fixture import EvalFixture
from lsm_harness.ops.eval.scenario import (
    EvalScenario,
    abort,
    compact,
    continue_,
    prompt,
    reload,
)
from lsm_harness.ops.eval.task import EvalTask
from lsm_harness.ops.eval.assertions import (
    expect_tool,
    file_equals,
    reply_contains,
    session_tree_has,
    summary_contains,
    tool_output_contains,
    context_user_once,
)


def _reply(text: str) -> ModelResponse:
    return ModelResponse(text=text, usage=Usage(input_tokens=100, output_tokens=20))


def _tool_call(tool: str, args: dict, call_id: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(call_id, tool, args)],
        stop_reason="tool_calls",
        usage=Usage(input_tokens=100, output_tokens=20),
    )


# calc-bug fixture contents
_CALC_BUGGY = "def add(a, b):\n    return a - b\n"
_CALC_FIXED = "def add(a, b):\n    return a + b\n"
_CALC_TEST = (
    "from calc import add\n\n"
    "assert add(2, 3) == 5, f'add(2, 3) == {add(2, 3)}, expected 5'\n"
    "assert add(-1, 1) == 0, f'add(-1, 1) == {add(-1, 1)}, expected 0'\n"
    "print('OK')\n"
)


def core_scenarios() -> list[EvalScenario]:
    """The five core scenarios, in a stable order."""
    return [
        _read_and_answer(),
        _calc_bug(),
        _self_correct(),
        _abort_restart_continue(),
        _compaction_remembers_goal(),
    ]


def core_tasks() -> list[EvalTask]:
    """The core non-scripted tasks (real agent decides its own path).

    These are the *unscripted* counterparts of the deterministic scenarios:
    same fixture + assertions, but no scripted model responses — the agent
    must read, edit and run on its own, and correctness comes from the
    verify command + file effects.
    """
    return [
        EvalTask(
            name="fix_calc_add",
            description="修复 calc.py 的 add bug 并通过 test_calc.py（非脚本）",
            prompt="calc.py 的 add 函数有 bug，请修复它，然后运行 test_calc.py 确认通过。",
            fixture=EvalFixture.inline({
                "calc.py": _CALC_BUGGY,
                "test_calc.py": _CALC_TEST,
            }),
            verify_commands=["python test_calc.py"],
            assertions=[
                file_equals("calc.py", _CALC_FIXED),
                expect_tool("exec"),
                tool_output_contains("exec", "OK"),
            ],
        ),
        EvalTask(
            name="report_answer_value",
            description="读取 answer.txt 并汇报数值（非脚本）",
            prompt="读取 answer.txt，告诉我里面的数字是多少。",
            fixture=EvalFixture.inline({"answer.txt": "42\n"}),
            assertions=[
                expect_tool("read_file"),
                reply_contains("42"),
            ],
        ),
    ]


def _read_and_answer() -> EvalScenario:
    return EvalScenario(
        name="read_and_answer",
        description="读取文件并正确回答其中的数字",
        fixture=EvalFixture.inline({"answer.txt": "42\n"}),
        steps=[prompt("读取 answer.txt，告诉我里面的数字是多少")],
        scripted_responses=[
            _tool_call("read_file", {"path": "answer.txt"}, "r1"),
            _reply("42"),
        ],
        assertions=[
            expect_tool("read_file"),
            reply_contains("42"),
        ],
    )


def _calc_bug() -> EvalScenario:
    return EvalScenario(
        name="calc_bug",
        description="修复 calc.py 的 add bug 并通过 test_calc.py",
        fixture=EvalFixture.inline({
            "calc.py": _CALC_BUGGY,
            "test_calc.py": _CALC_TEST,
        }),
        steps=[prompt("calc.py 的 add 函数有 bug，请修复它并通过 test_calc.py")],
        scripted_responses=[
            _tool_call("read_file", {"path": "calc.py"}, "r1"),
            _tool_call("write_file", {"path": "calc.py", "content": _CALC_FIXED}, "w1"),
            _tool_call("exec", {"command": "python test_calc.py"}, "e1"),
            _reply("已修复，测试通过。"),
        ],
        assertions=[
            expect_tool("read_file"),
            expect_tool("write_file"),
            expect_tool("exec"),
            file_equals("calc.py", _CALC_FIXED),
            tool_output_contains("exec", "OK"),
        ],
    )


def _self_correct() -> EvalScenario:
    return EvalScenario(
        name="self_correct",
        description="工具参数错误后自行纠正（先读错路径，再读对）",
        fixture=EvalFixture.inline({"data.txt": "secret-value\n"}),
        steps=[prompt("读取 data.txt 的内容并告诉我")],
        scripted_responses=[
            _tool_call("read_file", {"path": "data.txtx"}, "r1"),
            _tool_call("read_file", {"path": "data.txt"}, "r2"),
            _reply("secret-value"),
        ],
        assertions=[
            expect_tool("read_file"),
            tool_output_contains("read_file", "not found"),
            tool_output_contains("read_file", "secret-value"),
            reply_contains("secret-value"),
        ],
    )


def _abort_restart_continue() -> EvalScenario:
    return EvalScenario(
        name="abort_restart_continue",
        description="中断 → 重启 → 继续回答被中断的问题",
        steps=[
            prompt("2+2 等于几？", park=True),
            abort(),
            reload(),
            continue_(),
        ],
        scripted_responses=[_reply("4")],
        assertions=[
            reply_contains("4"),
            session_tree_has("message"),
            context_user_once("2+2 等于几？"),
        ],
    )


def _compaction_remembers_goal() -> EvalScenario:
    return EvalScenario(
        name="compaction_remembers_goal",
        description="长上下文压缩后仍记得目标编号",
        settings_overrides={"context_keep_recent_tokens": 1},
        steps=[
            prompt("记住：我们的目标是 TARGET-42。"),
            prompt("好的，请再复述一遍目标编号。"),
            compact(),
            prompt("目标编号是多少？"),
        ],
        scripted_responses=[
            _reply("已记住目标。"),
            _reply("目标是 TARGET-42。"),
            _reply("TARGET-42"),
        ],
        summary_text="目标是 TARGET-42",
        assertions=[
            session_tree_has("compaction"),
            summary_contains("TARGET-42"),
            reply_contains("TARGET-42"),
        ],
    )
