"""Secure localhost Web console and backwards-compatible HTTP gateway."""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import re
import sqlite3
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from lsm_harness.app import Harness
from lsm_harness.config import Settings
from lsm_harness.db import connect
from lsm_harness.events import HarnessEvent
from lsm_harness.gateway.web_runtime import BusyError, TurnCoordinator, aggregate_turns, trace_slice, turn_events
from lsm_harness.models import PROVIDERS
from lsm_harness.ops.flow import topology_snapshot


MAX_BODY = 2 * 1024 * 1024
STATIC_ROOT = Path(__file__).with_name("static")
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/static/styles.css": "styles.css", "/static/app.js": "app.js"}


class LocalWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, *, runtime: TurnCoordinator, owns_app: bool):
        super().__init__(address, handler)
        self.runtime = runtime
        self.owns_app = owns_app


def create_server(settings: Settings | None = None, *, port: int = 8910, app: Harness | None = None) -> LocalWebServer:
    """Create a localhost-only server without starting its main loop."""
    owns_app = app is None
    if app is None:
        settings = settings or Settings()
        app = Harness(settings, conn=connect(settings.home, check_same_thread=False))
    runtime = TurnCoordinator(app)
    if app.sandbox is None and not app.settings.web_allow_host_shell:
        app.tools = app.tools.filter([name for name in app.tools.tool_names() if name != "exec"])
    return LocalWebServer(("127.0.0.1", port), _make_handler(runtime), runtime=runtime, owns_app=owns_app)


def run_server(settings: Settings | None = None, port: int = 8910, host: str = "127.0.0.1", *, open_browser: bool = True) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("lsm web only listens on localhost")
    server = create_server(settings, port=port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"LSM Web Console → {url}")
    print(f"  model: {server.runtime.app.settings.model}  session: {server.runtime.app.session.session_id[:8]}")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if server.owns_app:
            server.runtime.app.close()


def _make_handler(runtime: TurnCoordinator) -> type[BaseHTTPRequestHandler]:
    app = runtime.app

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "LSMWeb/2.2"

        def do_GET(self) -> None:
            if not self._valid_host_origin():
                return
            parsed = urlparse(self.path)
            if parsed.path in STATIC_FILES:
                self._static(parsed.path)
                return
            if parsed.path == "/health":
                self._json(200, {"status": "ok", **runtime.state()})
                return
            if not self._authorized():
                return
            try:
                self._api_get(parsed.path, parse_qs(parsed.query))
            except Exception as exc:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:
            if not self._valid_host_origin() or not self._authorized():
                return
            try:
                self._api_post(urlparse(self.path).path, self._read_body())
            except BusyError as exc:
                self._json(409, {"error": str(exc), **runtime.state()})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def _api_get(self, path: str, query: dict[str, list[str]]) -> None:
            if path == "/api/bootstrap":
                data = runtime.bootstrap()
                data["turns"] = aggregate_turns(app.settings.home)[:30]
                self._json(200, data)
            elif path == "/api/topology":
                self._json(200, topology_snapshot(app))
            elif path == "/api/events":
                self._json(200, trace_slice(app.settings.home, int(query.get("cursor", ["0"])[0])))
            elif path == "/api/turns":
                self._json(200, {"turns": aggregate_turns(app.settings.home)[:200]})
            elif re.fullmatch(r"/api/turns/[^/]+/events", path):
                turn_id = path.split("/")[3]
                self._json(200, {"turn_id": turn_id, "events": turn_events(app.settings.home, turn_id)})
            elif re.fullmatch(r"/api/turns/[^/]+", path):
                turn_id = path.split("/")[3]
                events = turn_events(app.settings.home, turn_id)
                self._json(200 if events else 404, {"turn_id": turn_id, "events": events})
            elif path in {"/api/sessions", "/sessions"}:
                self._json(200, {"sessions": _sessions_for(app)})
            elif re.fullmatch(r"/api/sessions/[^/]+/messages", path):
                session_id = path.split("/")[3]
                self._json(200, {"session_id": session_id, "messages": _messages_for(app, session_id)})
            elif path == "/api/memory":
                self._json(200, _memory_for(app))
            elif path == "/api/rag":
                self._json(200, _rag_for(app))
            elif path == "/api/tools":
                self._json(200, {"tools": [{"name": tool.name, "description": tool.description, "effect": tool.effect, "sandboxed": bool(app.sandbox and tool.name == "exec")} for tool in app.tools._tools.values()], "approvals": runtime.approvals.snapshot()})
            elif path == "/api/subagents":
                self._json(200, {"subagents": app.subagents.snapshot()})
            elif path == "/api/files":
                self._json(200, {"modified_files": app.file_state.modified_files, "current_summary": app.file_state.summary(), "last_summary": app.file_state.last_turn_summary, "diff": app.file_state.all_diffs()})
            elif path == "/api/database":
                self._json(200, {"tables": _database_status_for(app)})
            elif path == "/api/ops":
                all_events = trace_slice(app.settings.home, 0, 2000)["events"]
                self._json(200, {"runtime": runtime.state(), "usage": app.tracer.usage_summary(), "events": all_events[-100:]})
            elif path == "/api/settings":
                self._json(200, _safe_settings(app))
            else:
                self._json(404, {"error": "not found"})

        # POST, streaming, security and storage helpers are appended below.

    return _finish_handler(Handler, runtime)


def _finish_handler(handler: type[BaseHTTPRequestHandler], runtime: TurnCoordinator) -> type[BaseHTTPRequestHandler]:
    """Attach helpers while keeping the main request router readable."""
    app = runtime.app

    def api_post(self, path: str, body: dict[str, Any]) -> None:
        if path in {"/api/turns/stream", "/respond/stream"}:
            self._stream_turn(body.get("message", ""))
        elif path == "/respond":
            turn_id = str(uuid4())
            runtime.claim(turn_id, "http")
            try:
                result = app.respond(body.get("message", ""), source="http", turn_id=turn_id, approval_broker=runtime.approvals)
                self._json(200, {"turn_id": turn_id, "reply": result.reply, "aborted": result.aborted})
            finally:
                runtime.release(turn_id)
        elif re.fullmatch(r"/api/turns/[^/]+/abort", path) or path == "/abort":
            turn_id = path.split("/")[3] if path.startswith("/api/") else runtime.active_turn_id
            ok = runtime.abort(turn_id)
            self._json(200 if ok else 404, {"aborted": ok, "turn_id": turn_id})
        elif re.fullmatch(r"/api/turns/[^/]+/steer", path) or path == "/steer":
            turn_id = path.split("/")[3] if path.startswith("/api/") else runtime.active_turn_id
            ok = runtime.steer(turn_id, str(body.get("message", "")))
            self._json(200 if ok else 404, {"steered": ok, "turn_id": turn_id})
        elif re.fullmatch(r"/api/approvals/[^/]+", path):
            request_id = path.split("/")[3]
            decision = str(body.get("decision", "reject"))
            if decision not in {"approve", "reject"}:
                raise ValueError("decision must be approve or reject")
            ok = runtime.approvals.resolve(request_id, decision, reason=str(body.get("reason", "")))
            self._json(200 if ok else 404, {"resolved": ok, "id": request_id})
        elif path == "/api/sessions":
            runtime.ensure_idle()
            requested = str(body.get("session_id", ""))
            session_id = app.session.resume(requested) if requested else app.session.start_new()
            self._json(200 if session_id else 404, {"session_id": session_id} if session_id else {"error": "session not found"})
        elif path == "/api/settings/model":
            runtime.ensure_idle()
            provider = str(body.get("provider", ""))
            if provider not in PROVIDERS:
                raise ValueError("unknown provider")
            app.switch_model(provider, model=str(body.get("model", "")))
            self._json(200, _safe_settings(app))
        elif path == "/api/settings/thinking":
            runtime.ensure_idle()
            thinking = str(body.get("thinking", "disabled"))
            if thinking not in {"disabled", "enabled", "auto"}:
                raise ValueError("invalid thinking mode")
            app.settings.thinking = thinking
            app.session.record_thinking_change(thinking)
            self._json(200, _safe_settings(app))
        else:
            self._json(404, {"error": "not found"})

    def stream_turn(self, message: str | dict) -> None:
        if not message:
            raise ValueError("message is required")
        turn_id = str(uuid4())
        runtime.claim(turn_id, "web")
        event_queue: queue.Queue[HarnessEvent | dict | None] = queue.Queue()

        def worker() -> None:
            try:
                app.respond(message, observer=event_queue.put, source="web", turn_id=turn_id, approval_broker=runtime.approvals)
            except Exception as exc:
                event_queue.put({"type": "transport.error", "turn_id": turn_id, "data": {"error": str(exc)}})
            finally:
                runtime.release(turn_id)
                event_queue.put(None)

        threading.Thread(target=worker, name=f"lsm-turn-{turn_id[:8]}", daemon=True).start()
        self.send_response(200)
        self._security_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-LSM-Turn-ID", turn_id)
        self.end_headers()
        try:
            self._sse({"type": "transport.ready", "turn_id": turn_id, "data": {}})
            while True:
                try:
                    item = event_queue.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    continue
                if item is None:
                    break
                self._sse(item.as_dict() if isinstance(item, HarnessEvent) else item)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            runtime.abort(turn_id)

    def sse(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.flush()

    def valid_host_origin(self) -> bool:
        host = self.headers.get("Host", "")
        host_name = host.rsplit(":", 1)[0].strip("[]").lower()
        if host_name not in {"127.0.0.1", "localhost", "::1"}:
            self._json(403, {"error": "invalid host"})
            return False
        origin = self.headers.get("Origin", "")
        if origin:
            parsed = urlparse(origin)
            if parsed.scheme != "http" or (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
                self._json(403, {"error": "invalid origin"})
                return False
        return True

    def authorized(self) -> bool:
        if self.headers.get("X-LSM-Web-Token", "") != runtime.token:
            self._json(401, {"error": "unauthorized"})
            return False
        return True

    def read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > MAX_BODY:
            raise ValueError("request body too large")
        if length <= 0:
            return {}
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def static(self, path: str) -> None:
        file_name = STATIC_FILES[path]
        target = STATIC_ROOT / file_name
        if not target.is_file():
            self._json(404, {"error": "static asset missing"})
            return
        body = target.read_bytes()
        if file_name == "index.html":
            body = body.replace(b"__LSM_WEB_TOKEN__", runtime.token.encode("ascii"))
        content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        self.send_response(200)
        self._security_headers()
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") or content_type == "application/javascript" else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if file_name == "index.html" else "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def security_headers(self) -> None:
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")

    def json_response(self, code: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    handler._api_post = api_post
    handler._stream_turn = stream_turn
    handler._sse = sse
    handler._valid_host_origin = valid_host_origin
    handler._authorized = authorized
    handler._read_body = read_body
    handler._static = static
    handler._security_headers = security_headers
    handler._json = json_response
    handler.log_message = lambda self, fmt, *args: None
    return handler


def _open_db(app: Harness) -> sqlite3.Connection:
    conn = sqlite3.connect(app.settings.home / "state.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=3000")
    return conn


def _sessions_for(app: Harness) -> list[dict[str, Any]]:
    conn = _open_db(app)
    try:
        rows = conn.execute("SELECT s.id,s.title,s.created_at,s.updated_at,COUNT(c.id) message_count FROM sessions s LEFT JOIN chat_log c ON c.session_id=s.id GROUP BY s.id ORDER BY s.updated_at DESC LIMIT 200").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _messages_for(app: Harness, session_id: str) -> list[dict[str, Any]]:
    conn = _open_db(app)
    try:
        rows = conn.execute("SELECT id,role,content,source,meta,created_at FROM chat_log WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _database_status_for(app: Harness) -> list[dict[str, Any]]:
    conn = _open_db(app)
    try:
        names = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]
        result = []
        for name in names:
            if re.fullmatch(r"[A-Za-z0-9_]+", name):
                result.append({"name": name, "rows": conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]})
        return result
    finally:
        conn.close()


def _memory_for(app: Harness) -> dict[str, Any]:
    conn = _open_db(app)
    try:
        facts = [dict(row) for row in conn.execute(
            "SELECT id,subject,content,source,created_at FROM facts ORDER BY id DESC"
        ).fetchall()]
        episodes = [dict(row) for row in conn.execute(
            "SELECT id,happened_at,summary,created_at FROM episodes ORDER BY id DESC"
        ).fetchall()]
        row = conn.execute(
            "SELECT version,summary,through_chat_id,source_message_count,created_at "
            "FROM session_summaries WHERE session_id=? ORDER BY version DESC LIMIT 1",
            (app.session.session_id,),
        ).fetchone()
        return {"facts": facts, "episodes": episodes, "summary": dict(row) if row else None}
    finally:
        conn.close()


def _rag_for(app: Harness) -> dict[str, Any]:
    if app.rag is None:
        return {"enabled": False, "stats": {}, "documents": []}
    conn = _open_db(app)
    try:
        docs = [dict(row) for row in conn.execute(
            "SELECT id,path,title,ingested_at,chunk_count FROM rag_docs ORDER BY ingested_at DESC"
        ).fetchall()]
        chunks, tokens = conn.execute(
            "SELECT COUNT(*),COALESCE(SUM(token_count),0) FROM rag_chunks"
        ).fetchone()
        return {
            "enabled": True,
            "stats": {"documents": len(docs), "chunks": chunks, "total_tokens": tokens},
            "documents": docs,
        }
    finally:
        conn.close()


def _safe_settings(app: Harness) -> dict[str, Any]:
    return {
        "provider": app.settings.provider,
        "model": app.settings.model,
        "small_model": app.settings.small_model,
        "thinking": app.settings.thinking,
        "providers": [{"id": name, "model": value.model, "small_model": value.small_model, "configured": bool(os.getenv(value.key_env))} for name, value in PROVIDERS.items()],
        "web_allow_host_shell": app.settings.web_allow_host_shell,
        "sandbox": app.sandbox is not None,
        "rag": app.rag is not None,
        "mcp": app.mcp is not None,
    }
