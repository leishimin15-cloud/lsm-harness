"""Minimal HTTP API — one file, zero dependencies beyond stdlib.

Start with::

    lsm serve              # default http://0.0.0.0:8910
    lsm serve --port 9000  # custom port

Endpoints:

    GET  /health               → {"status": "ok", "model": "...", "session": "..."}
    GET  /sessions             → list of session summaries
    POST /respond              → {"message": "..."} → {"reply": "...", ...}
    POST /respond/stream       → SSE stream of events
    POST /abort                → cancel the current turn
    POST /steer                → inject a message into the running loop
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from lsm_harness.app import Harness
from lsm_harness.config import Settings
from lsm_harness.events import HarnessEvent


def run_server(settings: Settings | None = None, port: int = 8910, host: str = "0.0.0.0") -> None:
    """Start the HTTP API server (blocking)."""
    app = Harness(settings)
    handler = _make_handler(app)
    server = HTTPServer((host, port), handler)
    print(f"LSM HTTP API → http://{host}:{port}")
    print(f"  model: {app.settings.model}  session: {app.session.session_id[:8]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


def _make_handler(app: Harness) -> type[BaseHTTPRequestHandler]:
    """Closure to inject app into the handler class."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._json(200, {
                    "status": "ok",
                    "model": app.settings.model,
                    "session": app.session.session_id[:8],
                    "running": app.is_running,
                })
            elif self.path == "/sessions":
                sessions = app.session.list_sessions()
                self._json(200, {"sessions": sessions})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            body = self._read_body()

            if self.path == "/respond":
                result = app.respond(body.get("message", ""), source="http")
                self._json(200, {
                    "reply": result.reply,
                    "iterations": result.iterations,
                    "tools": [t["tool"] for t in result.tool_calls],
                    "aborted": result.aborted,
                })

            elif self.path == "/respond/stream":
                self._stream_respond(body.get("message", ""))

            elif self.path == "/abort":
                app.abort()
                self._json(200, {"aborted": True})

            elif self.path == "/steer":
                app.steer(body.get("message", ""))
                self._json(200, {"steered": True})

            else:
                self._json(404, {"error": "not found"})

        def _stream_respond(self, message: str):
            """Server-Sent Events stream of the agent's response."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            events: list[HarnessEvent] = []

            def on_event(event: HarnessEvent):
                events.append(event)
                data = json.dumps(event.as_dict(), ensure_ascii=False, default=str)
                try:
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            try:
                app.respond(message, observer=on_event, source="http")
            except Exception as exc:
                self.wfile.write(
                    f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
                )
                self.wfile.flush()

        # ── helpers ──────────────────────────────────────────

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"message": raw.decode("utf-8", errors="replace")}

        def _json(self, code: int, data: dict[str, Any]):
            body = json.dumps(data, ensure_ascii=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return _Handler
