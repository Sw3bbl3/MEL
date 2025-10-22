"""Reference HTTP server exposing the MEL runtime."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Optional

from .runtime import RouterRuntime


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8089


class RouterHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, *, runtime: RouterRuntime):
        super().__init__(server_address, RequestHandlerClass)
        self.runtime = runtime


class RouterRequestHandler(BaseHTTPRequestHandler):
    server: RouterHTTPServer  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - verbosity control
        if os.environ.get("MEL_QUIET", "0") == "1":
            return
        super().log_message(fmt, *args)

    # ------------------------------------------------------------------
    # GET endpoints
    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/state":
            self._send_json(200, self.server.runtime.export_state())
        else:
            self._send_json(404, {"error": "not_found"})

    # ------------------------------------------------------------------
    # POST endpoint for tasks
    # ------------------------------------------------------------------
    def do_POST(self) -> None:
        if self.path not in {"/", "/v1/task"}:
            self._send_json(404, {"error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid_json"})
            return

        try:
            result = self.server.runtime.handle(payload)
        except ValueError as exc:
            self._send_json(422, {"error": "invalid_request", "detail": str(exc)})
            return

        self._send_json(200, result.to_dict())

    # ------------------------------------------------------------------
    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    runtime: Optional[RouterRuntime] = None,
) -> RouterHTTPServer:
    runtime = runtime or RouterRuntime.with_defaults()
    return RouterHTTPServer((host, port), RouterRequestHandler, runtime=runtime)


def main(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    runtime: Optional[RouterRuntime] = None,
    config: Optional[str] = None,
) -> None:
    if runtime is None:
        runtime = RouterRuntime.from_config_file(config) if config else RouterRuntime.with_defaults()
    server = create_server(host, port, runtime=runtime)
    print(f"MEL router listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        pass


__all__ = [
    "create_server",
    "main",
    "RouterHTTPServer",
    "RouterRequestHandler",
]

