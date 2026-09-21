import hashlib
import io
import json
import socket
import ssl
import struct
import unittest
from unittest.mock import Mock, patch

from vllm_apple.multi_mac import FabricTransfer, FabricTransport
from vllm_apple.multi_mac_transport import (
    FABRIC_FRAME_MAGIC,
    FABRIC_FRAME_VERSION,
    FabricStream,
    accept_ethernet_fabric_stream,
    open_ethernet_fabric_stream,
)


def transfer(size=4, measurement="tb-a-b"):
    return FabricTransfer(
        "stage-a", "stage-b", "mac-a", "mac-b", FabricTransport.THUNDERBOLT,
        size, 50, measurement,
    )


class MultiMacTransportTests(unittest.TestCase):
    def test_frame_round_trip_is_plan_bound_and_ordered(self):
        wire = io.BytesIO()
        sender = FabricStream(
            wire, source_node="mac-a", destination_node="mac-b",
            authenticated_peer=True, maximum_payload_bytes=16,
        )
        sender.send(transfer(), b"data")
        sender.send(transfer(), b"next")
        wire.seek(0)
        receiver = FabricStream(
            wire, source_node="mac-a", destination_node="mac-b",
            authenticated_peer=True, maximum_payload_bytes=16,
        )
        self.assertEqual(receiver.receive(transfer()).payload, b"data")
        self.assertEqual(receiver.receive(transfer()).payload, b"next")

    def test_finish_sending_uses_half_close(self):
        stream_socket = Mock()
        stream = FabricStream(
            stream_socket, source_node="mac-a", destination_node="mac-b",
            authenticated_peer=True,
        )
        stream.finish_sending()
        stream_socket.shutdown.assert_called_once_with(socket.SHUT_WR)

    def test_peer_endpoint_and_payload_bounds_fail_before_io(self):
        with self.assertRaisesRegex(ValueError, "not authenticated"):
            FabricStream(io.BytesIO(), source_node="a", destination_node="b",
                         authenticated_peer=False)
        stream = FabricStream(
            io.BytesIO(), source_node="mac-a", destination_node="mac-b",
            authenticated_peer=True, maximum_payload_bytes=4,
        )
        with self.assertRaisesRegex(ValueError, "stream endpoints"):
            stream.send(FabricTransfer(
                "stage-a", "stage-b", "mac-x", "mac-b", FabricTransport.ETHERNET,
                4, 1, "eth-x-b"), b"data")
        with self.assertRaisesRegex(ValueError, "payload"):
            stream.send(transfer(), b"five!")

    def test_digest_tamper_and_plan_substitution_are_rejected(self):
        wire = io.BytesIO()
        FabricStream(
            wire, source_node="mac-a", destination_node="mac-b",
            authenticated_peer=True, maximum_payload_bytes=16,
        ).send(transfer(), b"data")
        encoded = bytearray(wire.getvalue())
        encoded[-1] ^= 1
        receiver = FabricStream(
            io.BytesIO(encoded), source_node="mac-a", destination_node="mac-b",
            authenticated_peer=True, maximum_payload_bytes=16,
        )
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            receiver.receive(transfer())

        wire.seek(0)
        receiver = FabricStream(
            wire, source_node="mac-a", destination_node="mac-b",
            authenticated_peer=True, maximum_payload_bytes=16,
        )
        with self.assertRaisesRegex(ValueError, "not bound"):
            receiver.receive(transfer(measurement="replacement"))

    def test_replayed_sequence_and_oversized_prefix_are_rejected(self):
        header = {
            "source_stage": "stage-a", "destination_stage": "stage-b",
            "source_node": "mac-a", "destination_node": "mac-b",
            "transport": "thunderbolt", "bytes": 4, "measurement_id": "tb-a-b",
            "sequence": 1, "payload_sha256": "a" * 64,
        }
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        wire = io.BytesIO(
            FABRIC_FRAME_MAGIC
            + struct.pack(">HII", FABRIC_FRAME_VERSION, len(encoded), 4)
            + encoded + b"data"
        )
        receiver = FabricStream(
            wire, source_node="mac-a", destination_node="mac-b",
            authenticated_peer=True, maximum_payload_bytes=4,
        )
        with self.assertRaisesRegex(ValueError, "not bound"):
            receiver.receive(transfer())

        oversized = io.BytesIO(
            FABRIC_FRAME_MAGIC + struct.pack(">HII", FABRIC_FRAME_VERSION, 4097, 0)
        )
        receiver = FabricStream(
            oversized, source_node="mac-a", destination_node="mac-b",
            authenticated_peer=True, maximum_payload_bytes=4,
        )
        with self.assertRaisesRegex(ValueError, "bounds"):
            receiver.receive(transfer())

    def test_ethernet_client_requires_mtls_hostname_and_pinned_peer(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        certificate = b"peer-certificate"
        digest = hashlib.sha256(certificate).hexdigest()
        raw = Mock()
        secure = Mock()
        secure.getpeercert.return_value = certificate
        with (
            patch("vllm_apple.multi_mac_transport.socket.create_connection", return_value=raw),
            patch.object(ssl.SSLContext, "wrap_socket", return_value=secure) as wrap,
        ):
            stream = open_ethernet_fabric_stream(
                "mac-b.local", 9443, source_node="mac-a", destination_node="mac-b",
                tls_context=context, expected_peer_certificate_sha256=digest,
            )
        self.assertIs(stream._stream, secure)
        wrap.assert_called_once_with(raw, server_hostname="mac-b.local")
        secure.settimeout.assert_called_once_with(10.0)

        secure = Mock()
        secure.getpeercert.return_value = b"replacement"
        with (
            patch("vllm_apple.multi_mac_transport.socket.create_connection", return_value=raw),
            patch.object(ssl.SSLContext, "wrap_socket", return_value=secure),
            self.assertRaisesRegex(ValueError, "certificate mismatch"),
        ):
            open_ethernet_fabric_stream(
                "mac-b.local", 9443, source_node="mac-a", destination_node="mac-b",
                tls_context=context, expected_peer_certificate_sha256=digest,
            )
        secure.close.assert_called_once()

    def test_ethernet_client_rejects_server_auth_only_context_before_connect(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with (
            patch("vllm_apple.multi_mac_transport.socket.create_connection") as connect,
            self.assertRaisesRegex(ValueError, "authenticated Ethernet"),
        ):
            open_ethernet_fabric_stream(
                "mac-b.local", 9443, source_node="mac-a", destination_node="mac-b",
                tls_context=context, expected_peer_certificate_sha256="a" * 64,
            )
        connect.assert_not_called()

    def test_ethernet_server_requires_mtls_and_pinned_client(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.verify_mode = ssl.CERT_REQUIRED
        certificate = b"client-certificate"
        digest = hashlib.sha256(certificate).hexdigest()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client = socket.create_connection(listener.getsockname())
        secure = Mock()
        secure.getpeercert.return_value = certificate
        try:
            with patch.object(ssl.SSLContext, "wrap_socket", return_value=secure) as wrap:
                stream = accept_ethernet_fabric_stream(
                    listener, source_node="mac-a", destination_node="mac-b",
                    tls_context=context, expected_peer_certificate_sha256=digest,
                )
            self.assertIs(stream._stream, secure)
            self.assertTrue(wrap.call_args.kwargs["server_side"])
            secure.settimeout.assert_called_once_with(10.0)
        finally:
            client.close()
            listener.close()


if __name__ == "__main__":
    unittest.main()
