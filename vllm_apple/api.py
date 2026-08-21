from __future__ import annotations

import json
import os
import socket
import socketserver
import stat
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .auth import SessionAuthenticator
from .backend import BackendHTTPError
from .events import RuntimeEvent, SubscriptionLimitError
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


class _BoundedRuntimeServerMixin:
    daemon_threads = True
    request_queue_size = 64

    def _initialize_runtime(
        self,
        service: RuntimeService,
        max_concurrent_requests: int,
        socket_timeout: float,
        session_token: str | None,
    ) -> None:
        if max_concurrent_requests <= 0 or socket_timeout <= 0:
            raise ValueError("server limits must be positive")
        self.service = service
        self.authenticator = SessionAuthenticator(session_token)
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)
        self._socket_timeout = socket_timeout
        self._request_metrics_lock = threading.Lock()
        self._active_requests = 0
        self._peak_active_requests = 0
        self._completed_requests = 0
        self._rejected_requests = 0

    def request_metrics(self) -> dict[str, int]:
        with self._request_metrics_lock:
            return {
                "active_requests": self._active_requests,
                "peak_active_requests": self._peak_active_requests,
                "completed_requests": self._completed_requests,
                "rejected_requests": self._rejected_requests,
            }

    def _record_request_admitted(self) -> None:
        with self._request_metrics_lock:
            self._active_requests += 1
            self._peak_active_requests = max(
                self._peak_active_requests, self._active_requests
            )

    def _release_request_slot(self, *, completed: bool) -> None:
        with self._request_metrics_lock:
            self._active_requests -= 1
            if self._active_requests < 0:
                raise RuntimeError("request slot accounting underflow")
            if completed:
                self._completed_requests += 1
        self._request_slots.release()

    def get_request(self) -> tuple[socket.socket, Any]:
        request, address = super().get_request()
        request.settimeout(self._socket_timeout)
        return request, address

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        if not self._request_slots.acquire(blocking=False):
            with self._request_metrics_lock:
                self._rejected_requests += 1
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
        self._record_request_admitted()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release_request_slot(completed=False)
            raise

    def process_request_thread(
        self, request: socket.socket, client_address: Any
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release_request_slot(completed=True)


class RuntimeHTTPServer(_BoundedRuntimeServerMixin, ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        service: RuntimeService,
        max_concurrent_requests: int = 32,
        socket_timeout: float = 30.0,
        session_token: str | None = None,
    ):
        self._initialize_runtime(
            service, max_concurrent_requests, socket_timeout, session_token
        )
        super().__init__(address, RuntimeRequestHandler)


class RuntimeUnixHTTPServer(
    _BoundedRuntimeServerMixin,
    socketserver.ThreadingMixIn,
    socketserver.UnixStreamServer,
):
    daemon_threads = True

    def __init__(
        self,
        socket_path: str,
        service: RuntimeService,
        max_concurrent_requests: int = 32,
        socket_timeout: float = 30.0,
        session_token: str | None = None,
    ):
        self.socket_path = socket_path
        if len(os.fsencode(socket_path)) >= 104:
            raise ValueError("Unix Domain Socket path must be shorter than 104 bytes on macOS")
        path_stat = None
        try:
            path_stat = os.lstat(socket_path)
        except FileNotFoundError:
            pass
        if path_stat is not None:
            if not stat.S_ISSOCK(path_stat.st_mode) or path_stat.st_uid != os.getuid():
                raise ValueError("refusing to replace a non-socket or foreign-owned UDS path")
            os.unlink(socket_path)
        self._initialize_runtime(
            service, max_concurrent_requests, socket_timeout, session_token
        )
        super().__init__(socket_path, RuntimeRequestHandler)
        os.chmod(socket_path, 0o600)

    def server_close(self) -> None:
        super().server_close()
        try:
            path_stat = os.lstat(self.socket_path)
            if stat.S_ISSOCK(path_stat.st_mode) and path_stat.st_uid == os.getuid():
                os.unlink(self.socket_path)
        except FileNotFoundError:
            pass


class RuntimeRequestHandler(BaseHTTPRequestHandler):
    server: RuntimeHTTPServer | RuntimeUnixHTTPServer
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

    def _authorize(self) -> bool:
        if self.server.authenticator.authorize(self.headers.get("Authorization")):
            return True
        payload = b'{"error":{"message":"authentication required","type":"vllm_apple_error","param":null,"code":"unauthorized"}}'
        self.send_response(HTTPStatus.UNAUTHORIZED.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)
        return False

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
        if not self._authorize():
            return
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
        elif path == "/v1/events":
            self._stream_events()
        else:
            self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")

    def do_POST(self) -> None:
        if not self._authorize():
            return
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

    def _write_event(self, event: RuntimeEvent | None) -> None:
        if event is None:
            self.wfile.write(b": heartbeat\n\n")
        else:
            data = json.dumps(
                event.to_dict(), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self.wfile.write(f"id: {event.event_id}\n".encode("ascii"))
            self.wfile.write(f"event: {event.type}\n".encode("utf-8"))
            self.wfile.write(b"data: " + data + b"\n\n")
        self.wfile.flush()

    def _stream_events(self) -> None:
        raw_after = self.headers.get("Last-Event-ID", "0")
        try:
            after_sequence = int(raw_after)
            subscription = self.server.service.events.subscribe(after_sequence=after_sequence)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_event_id", "Last-Event-ID must be an integer")
            return
        except SubscriptionLimitError as error:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "subscriber_limit", str(error))
            return

        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for event in subscription:
                self._write_event(event)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            subscription.close()


def create_server(
    host: str,
    port: int,
    service: RuntimeService | None = None,
    max_concurrent_requests: int = 32,
    session_token: str | None = None,
) -> RuntimeHTTPServer:
    return RuntimeHTTPServer(
        (host, port),
        service or RuntimeService(),
        max_concurrent_requests=max_concurrent_requests,
        session_token=session_token,
    )


def create_unix_server(
    socket_path: str,
    service: RuntimeService,
    max_concurrent_requests: int = 32,
    session_token: str | None = None,
) -> RuntimeUnixHTTPServer:
    return RuntimeUnixHTTPServer(
        socket_path,
        service,
        max_concurrent_requests=max_concurrent_requests,
        session_token=session_token,
    )
