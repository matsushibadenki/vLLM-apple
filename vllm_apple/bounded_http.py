"""Bound accepted HTTP connections before allocating request threads."""
from __future__ import annotations

import socket
import threading
from http.server import ThreadingHTTPServer
from typing import Any


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16
    maximum_connections = 16
    header_deadline_seconds = 5.0
    io_timeout_seconds = 30.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._connections: set[socket.socket] = set()
        self._connection_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.maximum_connections)
        self._peak_connections = 0
        self._rejections = 0
        self._header_expirations = 0
        super().__init__(*args, **kwargs)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            with self._connection_lock:
                self._rejections += 1
            # Do not block the accept loop on an overloaded client.
            try:
                request.setblocking(False)
                request.send(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n'
                             b'Retry-After: 1\r\nConnection: close\r\n\r\n')
            except OSError:
                pass
            self.shutdown_request(request)
            return
        with self._connection_lock:
            self._connections.add(request)
            self._peak_connections = max(self._peak_connections, len(self._connections))
        request.settimeout(self.header_deadline_seconds)
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release_connection(request)
            self.shutdown_request(request)
            raise

    def _release_connection(self, request: socket.socket) -> None:
        with self._connection_lock:
            if request not in self._connections:
                return
            self._connections.remove(request)
        self._slots.release()

    def process_request_thread(self, request: socket.socket, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release_connection(request)

    def resource_snapshot(self) -> dict[str, int]:
        with self._connection_lock:
            return dict(limit=self.maximum_connections, active=len(self._connections),
                        peak=self._peak_connections, rejected=self._rejections,
                        header_expirations=self._header_expirations)


class HeaderDeadlineMixin:
    """Absolute header deadline, including request-line trickle, per request."""

    def handle_one_request(self) -> None:
        expired = threading.Event()

        def expire() -> None:
            expired.set()
            with self.server._connection_lock:
                self.server._header_expirations += 1
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        self._header_expired = expired
        self._header_timer = threading.Timer(self.server.header_deadline_seconds, expire)
        self._header_timer.daemon = True
        self.connection.settimeout(self.server.header_deadline_seconds)
        self._header_timer.start()
        try:
            super().handle_one_request()
        finally:
            self._header_timer.cancel()
            self._header_timer.join()

    def parse_request(self) -> bool:
        parsed = super().parse_request()
        self._header_timer.cancel()
        self._header_timer.join()
        if self._header_expired.is_set():
            self.close_connection = True
            return False
        self.connection.settimeout(self.server.io_timeout_seconds)
        return parsed
