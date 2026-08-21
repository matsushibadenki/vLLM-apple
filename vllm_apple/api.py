from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .backend import BackendHTTPError
from .service import InferenceUnavailableError, RuntimeService
from .version import API_VERSION, MINIMUM_CLIENT_VERSION, SCHEMA_VERSION, __version__


MAX_REQUEST_BYTES = 4 * 1024 * 1024


def _metadata() -> dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "runtime_version": __version__,
        "minimum_client_version": MINIMUM_CLIENT_VERSION,
    }


class RuntimeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(
        self,
        address: tuple[str, int],
        service: RuntimeService,
        max_concurrent_requests: int = 32,
        socket_timeout: float = 30.0,
    ):
        if max_concurrent_requests <= 0 or socket_timeout <= 0:
            raise ValueError("server limits must be positive")
        self.service = service
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)
        self._socket_timeout = socket_timeout
        super().__init__(address, RuntimeRequestHandler)

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        request, address = super().get_request()
        request.settimeout(self._socket_timeout)
        return request, address

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        if not self._request_slots.acquire(blocking=False):
            payload = b'{"error":{"code":"server_busy","message":"runtime is busy"}}'
            response = (
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Type: application/json\r\n"
                b"Connection: close\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                + payload
            )
            try:
                request.sendall(response)
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def process_request_thread(
        self, request: socket.socket, client_address: tuple[str, int]
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


class RuntimeRequestHandler(BaseHTTPRequestHandler):
    server: RuntimeHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def _control_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        return {**_metadata(), **data}

    def _error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._send(
            status,
            {
                "error": {
                    "message": message,
                    "type": "vllm_apple_error",
                    "param": None,
                    "code": code,
                }
            },
        )

    def _read_json(self) -> dict[str, Any] | None:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._error(HTTPStatus.LENGTH_REQUIRED, "content_length_required", "missing Content-Length")
            return None
        try:
            length = int(content_length)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_content_length", "invalid Content-Length")
            return None
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large", "request is too large")
            return None
        try:
            decoded = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "invalid_json", "request body must be valid JSON")
            return None
        if not isinstance(decoded, dict):
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "request body must be an object")
            return None
        return decoded

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        snapshot = self.server.service.snapshot()
        if path == "/health":
            self._send(
                HTTPStatus.OK,
                self._control_payload(
                    {
                        "status": snapshot.state.value,
                        "control_ready": snapshot.control_ready,
                        "inference_ready": snapshot.inference_ready,
                    }
                ),
            )
        elif path == "/ready":
            status = HTTPStatus.OK if snapshot.control_ready else HTTPStatus.SERVICE_UNAVAILABLE
            self._send(status, self._control_payload({"ready": snapshot.control_ready}))
        elif path == "/v1/runtime":
            self._send(HTTPStatus.OK, self._control_payload(snapshot.to_dict()))
        elif path == "/v1/hardware":
            self._send(
                HTTPStatus.OK,
                self._control_payload({"hardware": snapshot.profile.hardware.to_dict()}),
            )
        elif path == "/v1/profiles":
            self._send(
                HTTPStatus.OK,
                self._control_payload({"profiles": [snapshot.profile.to_dict()]}),
            )
        elif path == "/v1/models":
            try:
                models = self.server.service.engine.models()
            except BackendHTTPError as error:
                self._backend_error(error)
                return
            self._send(HTTPStatus.OK, {"object": "list", "data": models})
        else:
            self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")
            return
        request = self._read_json()
        if request is None:
            return
        if request.get("stream") is True:
            self._stream_chat(request)
            return
        started = time.monotonic()
        try:
            response = self.server.service.engine.chat_completions(request)
        except InferenceUnavailableError as error:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "backend_unavailable", str(error))
            return
        except BackendHTTPError as error:
            self._backend_error(error)
            return
        response.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
        response.setdefault("object", "chat.completion")
        response.setdefault("created", int(time.time()))
        response.setdefault("runtime_ms", round((time.monotonic() - started) * 1000, 3))
        self._send(HTTPStatus.OK, response)

    def _backend_error(self, error: BackendHTTPError) -> None:
        try:
            status = HTTPStatus(error.status)
        except ValueError:
            status = HTTPStatus.BAD_GATEWAY
        self._error(status, error.code or "backend_error", str(error))

    def _stream_chat(self, request: dict[str, Any]) -> None:
        try:
            upstream_context = self.server.service.engine.open_chat_stream(request)
        except InferenceUnavailableError as error:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "backend_unavailable", str(error))
            return
        except BackendHTTPError as error:
            self._backend_error(error)
            return

        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            with upstream_context as upstream:
                read = getattr(upstream, "read1", upstream.read)
                while True:
                    chunk = read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            # Client cancellation must promptly close the upstream response and
            # release the bounded request slot without turning into a daemon error.
            return


def create_server(
    host: str,
    port: int,
    service: RuntimeService | None = None,
    max_concurrent_requests: int = 32,
) -> RuntimeHTTPServer:
    return RuntimeHTTPServer(
        (host, port), service or RuntimeService(), max_concurrent_requests=max_concurrent_requests
    )
