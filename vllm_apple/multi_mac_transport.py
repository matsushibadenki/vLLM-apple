"""Bounded authenticated stream framing for qualified Multi-Mac links."""
from __future__ import annotations

import hashlib
import json
import socket
import ssl
import struct
from dataclasses import dataclass
from typing import BinaryIO

from .multi_mac import FabricTransfer

FABRIC_FRAME_MAGIC = b"VLAF"
FABRIC_FRAME_VERSION = 1
MAX_FABRIC_HEADER_BYTES = 4096
MAX_FABRIC_PAYLOAD_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class FabricFrame:
    sequence: int
    transfer: FabricTransfer
    payload: bytes

    def __post_init__(self) -> None:
        if not 0 <= self.sequence < 2**63:
            raise ValueError("invalid fabric frame sequence")
        if not isinstance(self.transfer, FabricTransfer):
            raise ValueError("invalid fabric frame transfer")
        if not isinstance(self.payload, bytes) or len(self.payload) > MAX_FABRIC_PAYLOAD_BYTES:
            raise ValueError("invalid fabric frame payload")
        if len(self.payload) != self.transfer.bytes:
            raise ValueError("fabric frame payload does not match planned transfer")


class FabricStream:
    """Strict in-order frame exchange over an already authenticated byte stream."""

    def __init__(
        self,
        stream: BinaryIO | socket.socket,
        *,
        source_node: str,
        destination_node: str,
        authenticated_peer: bool,
        maximum_payload_bytes: int = MAX_FABRIC_PAYLOAD_BYTES,
    ) -> None:
        if not authenticated_peer:
            raise ValueError("fabric stream peer is not authenticated")
        if not 1 <= maximum_payload_bytes <= MAX_FABRIC_PAYLOAD_BYTES:
            raise ValueError("invalid fabric stream payload limit")
        self._stream = stream
        self.source_node = source_node
        self.destination_node = destination_node
        self.maximum_payload_bytes = maximum_payload_bytes
        self._send_sequence = 0
        self._receive_sequence = 0

    def send(self, transfer: FabricTransfer, payload: bytes) -> None:
        self._validate_transfer(transfer)
        if not isinstance(payload, bytes) or len(payload) > self.maximum_payload_bytes:
            raise ValueError("fabric payload exceeds stream limit")
        frame = FabricFrame(self._send_sequence, transfer, payload)
        header = _header(frame)
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > MAX_FABRIC_HEADER_BYTES:
            raise ValueError("fabric frame header exceeds limit")
        prefix = FABRIC_FRAME_MAGIC + struct.pack(">HII", FABRIC_FRAME_VERSION, len(encoded), len(payload))
        _write_all(self._stream, prefix)
        _write_all(self._stream, encoded)
        _write_all(self._stream, payload)
        self._send_sequence += 1

    def receive(self, expected_transfer: FabricTransfer) -> FabricFrame:
        self._validate_transfer(expected_transfer)
        prefix = _read_exact(self._stream, 14)
        if prefix[:4] != FABRIC_FRAME_MAGIC:
            raise ValueError("invalid fabric frame magic")
        version, header_size, payload_size = struct.unpack(">HII", prefix[4:])
        if (
            version != FABRIC_FRAME_VERSION
            or not 1 <= header_size <= MAX_FABRIC_HEADER_BYTES
            or payload_size > self.maximum_payload_bytes
            or payload_size != expected_transfer.bytes
        ):
            raise ValueError("invalid fabric frame bounds")
        try:
            header = json.loads(_read_exact(self._stream, header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid fabric frame header") from error
        expected = _transfer_header(expected_transfer)
        if (
            not isinstance(header, dict)
            or set(header) != set(expected) | {"sequence", "payload_sha256"}
            or header.get("sequence") != self._receive_sequence
            or any(header.get(key) != value for key, value in expected.items())
            or not _digest(header.get("payload_sha256"))
        ):
            raise ValueError("fabric frame is not bound to its plan")
        payload = _read_exact(self._stream, payload_size)
        if hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
            raise ValueError("fabric frame payload digest mismatch")
        frame = FabricFrame(self._receive_sequence, expected_transfer, payload)
        self._receive_sequence += 1
        return frame

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()

    def finish_sending(self) -> None:
        shutdown = getattr(self._stream, "shutdown", None)
        if not callable(shutdown):
            raise OSError("fabric stream does not support half-close")
        shutdown(socket.SHUT_WR)

    def _validate_transfer(self, transfer: FabricTransfer) -> None:
        if (
            not isinstance(transfer, FabricTransfer)
            or transfer.source_node != self.source_node
            or transfer.destination_node != self.destination_node
            or transfer.bytes > self.maximum_payload_bytes
        ):
            raise ValueError("fabric transfer does not match stream endpoints")


def open_ethernet_fabric_stream(
    host: str,
    port: int,
    *,
    source_node: str,
    destination_node: str,
    tls_context: ssl.SSLContext,
    expected_peer_certificate_sha256: str,
    timeout_seconds: float = 10.0,
    maximum_payload_bytes: int = MAX_FABRIC_PAYLOAD_BYTES,
) -> FabricStream:
    """Open a mutually authenticated Ethernet stream with certificate pinning."""
    if (
        not isinstance(host, str)
        or not 1 <= len(host) <= 253
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in host)
        or not isinstance(port, int)
        or isinstance(port, bool)
        or not 1 <= port <= 65535
        or not isinstance(tls_context, ssl.SSLContext)
        or tls_context.verify_mode != ssl.CERT_REQUIRED
        or not tls_context.check_hostname
        or not _digest(expected_peer_certificate_sha256)
        or not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not 0 < timeout_seconds <= 300
    ):
        raise ValueError("invalid authenticated Ethernet fabric configuration")
    raw = socket.create_connection((host, port), timeout=float(timeout_seconds))
    secure = None
    try:
        secure = tls_context.wrap_socket(raw, server_hostname=host)
        certificate = secure.getpeercert(binary_form=True)
        if (
            not isinstance(certificate, bytes)
            or not certificate
            or hashlib.sha256(certificate).hexdigest()
            != expected_peer_certificate_sha256
        ):
            raise ValueError("Ethernet fabric peer certificate mismatch")
        secure.settimeout(float(timeout_seconds))
        return FabricStream(
            secure,
            source_node=source_node,
            destination_node=destination_node,
            authenticated_peer=True,
            maximum_payload_bytes=maximum_payload_bytes,
        )
    except BaseException:
        (secure if secure is not None else raw).close()
        raise


def accept_ethernet_fabric_stream(
    listener: socket.socket,
    *,
    source_node: str,
    destination_node: str,
    tls_context: ssl.SSLContext,
    expected_peer_certificate_sha256: str,
    timeout_seconds: float = 10.0,
    maximum_payload_bytes: int = MAX_FABRIC_PAYLOAD_BYTES,
) -> FabricStream:
    """Accept one mutually authenticated Ethernet peer with certificate pinning."""
    if (
        not isinstance(listener, socket.socket)
        or not isinstance(tls_context, ssl.SSLContext)
        or tls_context.verify_mode != ssl.CERT_REQUIRED
        or not _digest(expected_peer_certificate_sha256)
        or not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not 0 < timeout_seconds <= 300
    ):
        raise ValueError("invalid authenticated Ethernet fabric listener")
    raw, _ = listener.accept()
    secure = None
    try:
        raw.settimeout(float(timeout_seconds))
        secure = tls_context.wrap_socket(raw, server_side=True)
        certificate = secure.getpeercert(binary_form=True)
        if (
            not isinstance(certificate, bytes)
            or not certificate
            or hashlib.sha256(certificate).hexdigest()
            != expected_peer_certificate_sha256
        ):
            raise ValueError("Ethernet fabric peer certificate mismatch")
        secure.settimeout(float(timeout_seconds))
        return FabricStream(
            secure,
            source_node=source_node,
            destination_node=destination_node,
            authenticated_peer=True,
            maximum_payload_bytes=maximum_payload_bytes,
        )
    except BaseException:
        (secure if secure is not None else raw).close()
        raise


def _header(frame: FabricFrame) -> dict[str, object]:
    return {
        **_transfer_header(frame.transfer),
        "sequence": frame.sequence,
        "payload_sha256": hashlib.sha256(frame.payload).hexdigest(),
    }


def _transfer_header(transfer: FabricTransfer) -> dict[str, object]:
    return {
        "source_stage": transfer.source_stage,
        "destination_stage": transfer.destination_stage,
        "source_node": transfer.source_node,
        "destination_node": transfer.destination_node,
        "transport": transfer.transport.value,
        "bytes": transfer.bytes,
        "measurement_id": transfer.measurement_id,
    }


def _digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_all(stream: BinaryIO | socket.socket, payload: bytes) -> None:
    if isinstance(stream, socket.socket):
        stream.sendall(payload)
        return
    written = stream.write(payload)
    if written is not None and written != len(payload):
        raise OSError("fabric stream short write")
    flush = getattr(stream, "flush", None)
    if callable(flush):
        flush()


def _read_exact(stream: BinaryIO | socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        if isinstance(stream, socket.socket):
            chunk = stream.recv(size - len(result))
        else:
            chunk = stream.read(size - len(result))
        if not chunk:
            raise EOFError("fabric stream ended before frame completion")
        result.extend(chunk)
    return bytes(result)
