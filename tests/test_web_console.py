from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from lsm_harness.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.events import make_event
from lsm_harness.gateway.http_api import create_server
from lsm_harness.ops.approval import ApprovalBroker
from lsm_harness.smoke import ScriptedClient
from lsm_harness.types import TurnResult


def _app(tmp_path):
    settings = Settings(
        home=tmp_path,
        api_key="secret-never-render",
        model="scripted",
        small_model="scripted",
        sandbox_enabled=False,
        rag_enabled=False,
        mcp_enabled=False,
        consolidate_every=99,
    )
    return Harness(
        settings=settings,
        client=ScriptedClient(),
        conn=connect(tmp_path, check_same_thread=False),
    )


def _request(url, token="", method="GET", body=None, origin=""):
    headers = {}
    if token:
        headers["X-LSM-Web-Token"] = token
    if origin:
        headers["Origin"] = origin
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = Request(url, headers=headers, data=data, method=method)
    with urlopen(req, timeout=3) as response:
        return response.status, response.headers, response.read()


def test_web_console_local_token_static_and_topology(tmp_path):
    app = _app(tmp_path)
    server = create_server(app=app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert server.server_address[0] == "127.0.0.1"
        status, headers, body = _request(base + "/")
        assert status == 200
        assert b"__LSM_WEB_TOKEN__" not in body
        assert server.runtime.token.encode() in body
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        assert b"secret-never-render" not in body

        try:
            _request(base + "/api/bootstrap")
        except HTTPError as exc:
            assert exc.code == 401
        else:
            raise AssertionError("API accepted a request without its Web token")

        status, _, body = _request(base + "/api/topology", server.runtime.token)
        data = json.loads(body)
        assert status == 200
        assert {"gateway", "memory_gate", "approval", "subagents", "trace"} <= {
            node["id"] for node in data["nodes"]
        }

        try:
            _request(
                base + "/api/bootstrap",
                server.runtime.token,
                origin="https://evil.example",
            )
        except HTTPError as exc:
            assert exc.code == 403
        else:
            raise AssertionError("cross-origin request was not rejected")
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_approval_broker_is_serializable_and_resolvable():
    broker = ApprovalBroker(timeout=2)
    emitted = []
    result = []

    thread = threading.Thread(
        target=lambda: result.append(
            broker.request(
                turn_id="turn-1",
                session_id="session-1",
                tool_name="write_file",
                effect="local_write",
                arguments={"path": "demo.txt"},
                sandboxed=False,
                emit=lambda kind, data: emitted.append((kind, data)),
            )
        )
    )
    thread.start()
    while not broker.snapshot()["pending"]:
        thread.join(0.01)
    pending = broker.snapshot()["pending"][0]
    json.dumps(pending)
    assert "_event" not in pending
    assert broker.resolve(pending["id"], "approve")
    thread.join(2)
    assert result == [(True, "approved")]
    assert [item[0] for item in emitted] == [
        "tool.approval.required",
        "tool.approval.resolved",
    ]


def test_event_protocol_and_multimodal_jsonl_chain(tmp_path):
    event = make_event("tool.approval.required", "turn-1", {"tool_name": "write_file"}, session_id="s", sequence=4)
    payload = event.as_dict()
    assert payload["event_id"] == "turn-1:4"
    assert payload["trace_id"] == "turn-1"
    assert event.trace_id == "turn-1"
    assert payload["flow"]["active_nodes"] == ["approval"]
    assert payload["flow"]["state"] == "waiting"

    app = _app(tmp_path)
    try:
        result = TurnResult(reply="看到了图片", iterations=1)
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "检查这张图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
        # Simulate the run: the recorder persists per-message entries
        # (image payloads stripped), the projector writes SQLite.
        from lsm_harness.agent.messages import AssistantMessage
        app.session.recorder.record(
            app.session.persistable_user_message(user_message), source="test"
        )
        app.session.recorder.record(AssistantMessage(text=result.reply), source="test")
        status = app.session.add_exchange(user_message, result, "test")
        assert status == {"sqlite": "ok", "jsonl": "ok", "multimodal": True}
        rows = app.conn.execute(
            "SELECT content FROM chat_log WHERE session_id=? ORDER BY id",
            (app.session.session_id,),
        ).fetchall()
        assert rows[0]["content"] == "检查这张图\n[image]"
        assert "base64" not in app.session.jsonl_path.read_text(encoding="utf-8")
        entries = [json.loads(line) for line in app.session.jsonl_path.read_text(encoding="utf-8").splitlines()]
        assert entries[1]["parent_id"] == entries[0]["id"]
        assert entries[2]["parent_id"] == entries[1]["id"]
        assert len({entry["id"] for entry in entries}) == len(entries)
    finally:
        app.close()
