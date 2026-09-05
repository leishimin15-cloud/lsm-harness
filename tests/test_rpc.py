"""RPC mode (``lsm rpc``): JSONL command protocol over stdin/stdout."""

from __future__ import annotations

import json
import threading
import time
from io import StringIO

from lsm_harness.ai.api.common import snapshot
from lsm_harness.ai.types import AssistantMessageEvent, ModelResponse, Usage
from lsm_harness.coding_agent.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.gateway.rpc import MAX_RECORD_BYTES, run_rpc

from helpers import QueueClient, client_stream_fn


def _run(tmp_path, input_lines, *responses, stream_fn=None):
    """Serve the given JSONL input against a scripted Harness.

    Returns (exit_code, [parsed output lines]).
    """
    settings = Settings(api_key="scripted", home=tmp_path, sandbox_enabled=False)
    conn = connect(tmp_path, check_same_thread=False)
    if stream_fn is None:
        client = QueueClient(*responses)
        stream_fn = client_stream_fn(client)
    else:
        client = QueueClient()
    app = Harness(settings=settings, client=client, conn=conn, stream_fn=stream_fn)
    stdin = StringIO("".join(line + "\n" for line in input_lines))
    stdout = StringIO()
    rc = run_rpc(app=app, stdin=stdin, stdout=stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    return rc, lines


def _responses(lines):
    return [line for line in lines if line["type"] == "response"]


def _events(lines):
    return [line["event"] for line in lines if line["type"] == "event"]


def test_get_state_reports_runtime(tmp_path):
    rc, lines = _run(tmp_path, ['{"id":"1","type":"get_state"}'])
    assert rc == 0
    (answer,) = _responses(lines)
    assert answer["id"] == "1"
    assert answer["command"] == "get_state"
    assert answer["success"] is True
    assert answer["is_running"] is False
    assert answer["session_id"]


def test_prompt_streams_events_and_answers(tmp_path):
    rc, lines = _run(
        tmp_path,
        ['{"id":"1","type":"prompt","message":"你好"}'],
        ModelResponse(text="你好呀", usage=Usage(4, 4)),
    )
    assert rc == 0
    (answer,) = _responses(lines)
    assert answer["command"] == "prompt" and answer["success"] is True
    types = [event["type"] for event in _events(lines)]
    assert "llm.text.delta" in types
    assert "trace.completed" in types
    deltas = [
        event["data"]["text"] for event in _events(lines)
        if event["type"] == "llm.text.delta"
    ]
    assert "".join(deltas) == "你好呀"


def test_prompt_while_running_steers(tmp_path):
    started = threading.Event()

    def blocking_stream(_model, _context, options):
        started.set()
        while not (options.interrupt and options.interrupt.is_set()):
            time.sleep(0.005)
        yield AssistantMessageEvent(
            "error",
            snapshot(text="", thinking="", pending={}, stop_reason="aborted"),
            error_category="aborted",
        )

    rc, lines = _run(
        tmp_path,
        [
            '{"id":"1","type":"prompt","message":"先跑着"}',
            '{"id":"2","type":"prompt","message":"插一句","streamingBehavior":"steer"}',
            '{"id":"3","type":"abort"}',
        ],
        stream_fn=blocking_stream,
    )
    assert rc == 0
    answers = _responses(lines)
    assert answers[0]["command"] == "prompt" and answers[0]["success"] is True
    assert answers[1]["queued_as"] == "steer"
    assert answers[1]["success"] is True
    assert answers[2]["command"] == "abort" and answers[2]["success"] is True


def test_unknown_command_and_invalid_json(tmp_path):
    rc, lines = _run(
        tmp_path,
        ['{"id":"1","type":"explode"}', "not json at all"],
    )
    assert rc == 0
    answers = _responses(lines)
    assert answers[0]["success"] is False
    assert "unknown command" in answers[0]["error"]
    assert answers[1]["success"] is False
    assert "invalid JSON" in answers[1]["error"]


def test_oversized_record_is_rejected(tmp_path):
    big = "x" * (MAX_RECORD_BYTES + 1)
    rc, lines = _run(tmp_path, [big])
    assert rc == 0
    (answer,) = _responses(lines)
    assert answer["success"] is False
    assert "16 MiB" in answer["error"]


def test_session_and_thinking_commands(tmp_path):
    rc, lines = _run(
        tmp_path,
        [
            '{"id":"1","type":"get_state"}',
            '{"id":"2","type":"cycle_thinking_level"}',
            '{"id":"3","type":"set_thinking_level","level":"enabled"}',
            '{"id":"4","type":"set_thinking_level","level":"bogus"}',
            '{"id":"5","type":"new_session"}',
            '{"id":"6","type":"switch_session","session_id":"不存在"}',
        ],
    )
    assert rc == 0
    answers = _responses(lines)
    original_session = answers[0]["session_id"]
    assert answers[1]["success"] is True and answers[1]["level"] == "auto"
    assert answers[2]["success"] is True and answers[2]["level"] == "enabled"
    assert answers[3]["success"] is False
    assert answers[4]["success"] is True
    assert answers[4]["session_id"] != original_session
    assert answers[5]["success"] is False


def test_get_messages_after_run(tmp_path):
    rc, lines = _run(
        tmp_path,
        [
            '{"id":"1","type":"prompt","message":"记一句"}',
            '{"id":"2","type":"get_messages"}',
        ],
        ModelResponse(text="记住了", usage=Usage(4, 4)),
    )
    assert rc == 0
    answers = _responses(lines)
    assert answers[1]["success"] is True
    contents = [m.get("content") for m in answers[1]["messages"]]
    assert "记一句" in contents
    assert "记住了" in contents


def test_compact_command(tmp_path):
    rc, lines = _run(tmp_path, ['{"id":"1","type":"compact"}'])
    assert rc == 0
    (answer,) = _responses(lines)
    # Empty session: nothing to compact, but the command round-trips.
    assert answer["command"] == "compact"
    assert answer["success"] is False


def test_eof_closes_cleanly(tmp_path):
    rc, lines = _run(tmp_path, [])
    assert rc == 0
    assert lines == []
