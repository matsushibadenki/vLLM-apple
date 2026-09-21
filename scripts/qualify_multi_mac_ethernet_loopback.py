#!/usr/bin/env python3
"""Qualify the authenticated Multi-Mac Ethernet framing over real loopback TLS."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import ssl
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_apple.multi_mac import FabricTransfer, FabricTransport  # noqa: E402
from vllm_apple.multi_mac_transport import (  # noqa: E402
    accept_ethernet_fabric_stream,
    open_ethernet_fabric_stream,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402


def _certificate_digest(path: Path) -> str:
    text = path.read_text(encoding="ascii")
    der = ssl.PEM_cert_to_DER_cert(text)
    return hashlib.sha256(der).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ca", type=Path, required=True)
    parser.add_argument("--server-cert", type=Path, required=True)
    parser.add_argument("--server-key", type=Path, required=True)
    parser.add_argument("--client-cert", type=Path, required=True)
    parser.add_argument("--client-key", type=Path, required=True)
    parser.add_argument("--payload-bytes", type=int, default=1_048_576)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    if not 1 <= arguments.payload_bytes <= 64 * 1024 * 1024:
        raise ValueError("payload bytes are outside qualification bounds")
    if not 1 <= arguments.frames <= 256:
        raise ValueError("frame count is outside qualification bounds")

    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.minimum_version = ssl.TLSVersion.TLSv1_2
    server_context.verify_mode = ssl.CERT_REQUIRED
    server_context.load_cert_chain(arguments.server_cert, arguments.server_key)
    server_context.load_verify_locations(cafile=str(arguments.ca))
    client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client_context.minimum_version = ssl.TLSVersion.TLSv1_2
    client_context.verify_mode = ssl.CERT_REQUIRED
    client_context.check_hostname = True
    client_context.load_cert_chain(arguments.client_cert, arguments.client_key)
    client_context.load_verify_locations(cafile=str(arguments.ca))
    server_digest = _certificate_digest(arguments.server_cert)
    client_digest = _certificate_digest(arguments.client_cert)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    payload = bytes((index * 17 + 3) & 0xFF for index in range(arguments.payload_bytes))
    payload_digest = hashlib.sha256(payload).hexdigest()
    transfer = FabricTransfer(
        "stage-a", "stage-b", "mac-a", "mac-b", FabricTransport.ETHERNET,
        len(payload), 1, "loopback-mtls",
    )
    received: list[str] = []
    server_error: list[str] = []

    def serve() -> None:
        try:
            stream = accept_ethernet_fabric_stream(
                listener,
                source_node="mac-a",
                destination_node="mac-b",
                tls_context=server_context,
                expected_peer_certificate_sha256=client_digest,
                timeout_seconds=30,
            )
            try:
                for _ in range(arguments.frames):
                    frame = stream.receive(transfer)
                    received.append(hashlib.sha256(frame.payload).hexdigest())
            finally:
                stream.close()
        except BaseException as error:  # report the bounded qualification failure
            server_error.append(f"{type(error).__name__}:{error}")

    thread = threading.Thread(target=serve, name="fabric-loopback-server")
    thread.start()
    started = time.monotonic_ns()
    client = None
    try:
        client = open_ethernet_fabric_stream(
            "localhost",
            port,
            source_node="mac-a",
            destination_node="mac-b",
            tls_context=client_context,
            expected_peer_certificate_sha256=server_digest,
            timeout_seconds=30,
        )
        for _ in range(arguments.frames):
            client.send(transfer, payload)
        client.finish_sending()
    finally:
        listener.close()
    thread.join(timeout=30)
    if client is not None:
        client.close()
    elapsed_ns = time.monotonic_ns() - started
    total_bytes = arguments.payload_bytes * arguments.frames
    report = {
        "schema_version": 1,
        "scope": "multi_mac_ethernet_loopback_mtls",
        "transport": "ethernet",
        "loopback": True,
        "tls_version_minimum": "TLSv1.2",
        "mutual_tls": True,
        "hostname_verified": True,
        "server_certificate_sha256": server_digest,
        "client_certificate_sha256": client_digest,
        "payload_sha256": payload_digest,
        "payload_bytes": arguments.payload_bytes,
        "frames": arguments.frames,
        "total_bytes": total_bytes,
        "elapsed_nanoseconds": elapsed_ns,
        "throughput_bytes_per_second": int(total_bytes * 1_000_000_000 / elapsed_ns),
        "received_frames": len(received),
        "digest_mismatches": sum(value != payload_digest for value in received),
        "server_error": server_error[0] if server_error else None,
        "thread_stopped": not thread.is_alive(),
        "passed": (
            not thread.is_alive()
            and not server_error
            and len(received) == arguments.frames
            and all(value == payload_digest for value in received)
        ),
        "limitations": [
            "single-Mac loopback qualification",
            "does not measure physical Ethernet bandwidth or latency",
            "does not qualify Thunderbolt transport",
        ],
    }
    if arguments.report is not None:
        save_qualification_report(report, arguments.report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
