"""Delegate coding tasks to pi (sub-agent).

When the user asks lsm-harness to do something that involves writing or
editing code, this tool shells out to pi in non-interactive mode, passes
the task, and captures the result.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from lsm_harness.tools.registry import Tool

# ── helpers ───────────────────────────────────────────────────────


def _find_pi() -> str | None:
    """Locate the pi binary."""
    pi = shutil.which("pi")
    if pi:
        return pi
    # Check common install paths
    for candidate in [
        Path.home() / ".local/bin/pi",
        Path.home() / "node_modules/.bin/pi",
        Path("/usr/local/bin/pi"),
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def _delegate_to_pi(task: str, cwd: str = "", _on_update=None) -> str:
    """Run pi in non-interactive mode for a coding task.

    Uses ``pi --print`` which does one turn and exits, returning the
    assistant's response (including any file edits made).
    """
    pi = _find_pi()
    if not pi:
        return (
            "Error: pi not found. Install pi first:\n"
            "  npm install -g @anthropic-ai/pi\n"
            "Or set PI_PATH in your .env."
        )

    working_dir = cwd or os.getcwd()
    notify = _on_update or (lambda msg: None)

    notify(f"Delegating to pi in {working_dir}…")

    try:
        proc = subprocess.Popen(
            [pi, "--print", task],
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PI_NO_COLOR": "1"},
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        # Read stdout line by line, forwarding progress
        for line in proc.stdout:  # type: ignore[union-attr]
            stripped = line.rstrip()
            if stripped:
                stdout_lines.append(stripped)
                # Forward substantial lines as progress updates
                if len(stripped) > 5:
                    notify(stripped[:120])

        proc.wait(timeout=300)  # 5 min timeout

        # Collect any stderr
        for line in proc.stderr:  # type: ignore[union-attr]
            stderr_lines.append(line.rstrip())

        result = "\n".join(stdout_lines)
        if not result.strip():
            err_text = "\n".join(stderr_lines[-10:]) if stderr_lines else "(no output)"
            return f"Pi completed but produced no output. Stderr: {err_text}"

        if proc.returncode != 0:
            err_text = "\n".join(stderr_lines[-10:]) if stderr_lines else ""
            return f"Pi exited with code {proc.returncode}.\n{err_text}\n\nOutput:\n{result}"

        return result

    except subprocess.TimeoutExpired:
        return "Error: pi timed out after 5 minutes."
    except FileNotFoundError:
        return f"Error: pi binary not found at '{pi}'."
    except Exception as exc:
        return f"Error running pi: {type(exc).__name__}: {exc}"


# ── tool definition ──────────────────────────────────────────────


def make_tool(home: Path | None = None) -> Tool:
    """Create the delegate_code tool.

    Args:
        home: lsm-harness home directory (for resolving relative paths).
    """

    def delegate_code(
        task: str,
        working_dir: str = "",
        _on_update=None,
    ) -> str:
        """Delegate a coding or file-editing task to pi.

        Use this when the user asks you to:
        - Write, read, or edit files
        - Run shell commands
        - Search codebases
        - Anything involving code or file manipulation

        Args:
            task: Clear description of what to do. Include file paths,
                  constraints, and expected outputs.
            working_dir: Directory to run pi in. Defaults to current
                         working directory.
        """
        cwd = working_dir or os.getcwd()
        return _delegate_to_pi(task, cwd, _on_update)

    return Tool(
        name="delegate_code",
        description=(
            "将编码或文件操作任务委托给 pi Agent 执行。"
            "当用户需要编写、编辑、读取文件，运行命令，或搜索代码时使用此工具。"
            "pi 会在指定的工作目录中执行任务并返回结果。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "任务描述，包含文件路径、约束和期望输出。",
                },
                "working_dir": {
                    "type": "string",
                    "description": "pi 的工作目录，默认当前目录。",
                },
            },
            "required": ["task"],
        },
        fn=delegate_code,
        effect="external_write",  # pi can modify files
        timeout=300.0,  # 5 minutes
    )
