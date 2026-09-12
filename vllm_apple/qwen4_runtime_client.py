"""Bounded same-user client for the local Qwen4 runtime socket."""
from __future__ import annotations

import json
import os
import secrets
import socket
import stat
from pathlib import Path

from .qwen4_runtime_protocol import (
    QWEN4_RUNTIME_ABI_VERSION,
    build_qwen4_numeric_runtime_request,
    build_qwen4_numeric_streaming_runtime_request,
    parse_qwen4_runtime_response,
)
from .qwen4_runtime_transport import receive_qwen4_runtime_frame, send_qwen4_runtime_frame


MAX_SESSION_FILE_BYTES = 4096


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Qwen4 runtime session contains a duplicate JSON key")
        result[key] = value
    return result


def _identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


class Qwen4RuntimeClient:
    def __init__(
        self, socket_path: str | Path, session_file: str | Path, *, timeout_seconds: float = 10.0,
    ) -> None:
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
            raise ValueError("Qwen4 runtime client timeout is invalid")
        if not 0 < timeout_seconds <= 300:
            raise ValueError("Qwen4 runtime client timeout is invalid")
        self.socket_path = Path(socket_path).expanduser().absolute()
        self.session_file = Path(session_file).expanduser().absolute()
        self.timeout_seconds = float(timeout_seconds)

    def load_numeric(
        self,
        *,
        sequence: int,
        artifact_name: str,
        artifact_digest: str,
        target_dtype: str,
        scratch_bytes: int = 0,
    ) -> dict[str, object]:
        session_id = self._session_id()
        request = build_qwen4_numeric_runtime_request(
            session_id=session_id, sequence=sequence, request_id=secrets.token_hex(16),
            artifact_name=artifact_name, artifact_digest=artifact_digest,
            target_dtype=target_dtype, scratch_bytes=scratch_bytes)
        return self._request(request)

    def unload(self, *, sequence: int, handle: str) -> dict[str, object]:
        return self._simple_request(sequence, "unload", handle=handle)

    def load_numeric_streaming(
        self,
        *,
        sequence: int,
        artifact_name: str,
        artifact_digest: str,
        target_dtype: str,
        tile_bytes: int,
        buffer_count: int = 2,
        scratch_bytes: int = 0,
    ) -> dict[str, object]:
        request = build_qwen4_numeric_streaming_runtime_request(
            session_id=self._session_id(),
            sequence=sequence,
            request_id=secrets.token_hex(16),
            artifact_name=artifact_name,
            artifact_digest=artifact_digest,
            target_dtype=target_dtype,
            tile_bytes=tile_bytes,
            buffer_count=buffer_count,
            scratch_bytes=scratch_bytes,
        )
        return self._request(request)

    def status(self, *, sequence: int) -> dict[str, object]:
        return self._simple_request(sequence, "status")

    def shutdown(self, *, sequence: int) -> dict[str, object]:
        return self._simple_request(sequence, "shutdown")

    def _simple_request(self, sequence: int, operation: str, **values: object) -> dict[str, object]:
        request = {
            "abi_version": QWEN4_RUNTIME_ABI_VERSION,
            "session_id": self._session_id(),
            "sequence": sequence,
            "request_id": secrets.token_hex(16),
            "operation": operation,
            **values,
        }
        return self._request(request)

    def _session_id(self) -> str:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.session_file, flags)
        try:
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                    or stat.S_IMODE(before.st_mode) & 0o077
                    or not 1 <= before.st_size <= MAX_SESSION_FILE_BYTES):
                raise ValueError("Qwen4 runtime session file is unsafe")
            chunks = bytearray()
            while len(chunks) <= MAX_SESSION_FILE_BYTES:
                chunk = os.read(
                    descriptor, min(4096, MAX_SESSION_FILE_BYTES + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
            raw = bytes(chunks)
            after = os.fstat(descriptor)
            if (len(raw) != before.st_size
                    or (before.st_dev, before.st_ino, before.st_size)
                    != (after.st_dev, after.st_ino, after.st_size)):
                raise ValueError("Qwen4 runtime session file changed while reading")
        finally:
            os.close(descriptor)
        try:
            payload = json.loads(raw, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Qwen4 runtime session file is invalid") from error
        if (not isinstance(payload, dict) or set(payload) != {"schema_version", "session_id"}
                or payload["schema_version"] != 1 or not _identifier(payload["session_id"])):
            raise ValueError("Qwen4 runtime session identity is invalid")
        return payload["session_id"]

    def _request(self, request: dict[str, object]) -> dict[str, object]:
        info = self.socket_path.lstat()
        if (not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise ValueError("Qwen4 runtime socket is unsafe")
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(self.timeout_seconds)
            connection.connect(str(self.socket_path))
            send_qwen4_runtime_frame(connection, request)
            response = parse_qwen4_runtime_response(receive_qwen4_runtime_frame(connection))
        finally:
            connection.close()
        for name in ("session_id", "sequence", "request_id", "operation"):
            if response[name] != request[name]:
                raise ValueError("Qwen4 runtime response is not bound to its request")
        return response
