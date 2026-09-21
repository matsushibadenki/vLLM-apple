import tempfile
import unittest
import ssl
from pathlib import Path
from unittest.mock import Mock, patch

from vllm_apple.daemon import _is_loopback_host, _server_tls_context, serve


class DaemonRemoteTLSTests(unittest.TestCase):
    def test_loopback_classification_is_explicit(self):
        for host in ("localhost", "127.0.0.1", "127.4.5.6", "::1"):
            self.assertTrue(_is_loopback_host(host))
        for host in ("0.0.0.0", "::", "192.168.1.2", "runtime.example"):
            self.assertFalse(_is_loopback_host(host))

    def test_remote_bind_fails_before_server_without_explicit_secure_contract(self):
        with self.assertRaisesRegex(ValueError, "allow-remote"):
            serve(host="0.0.0.0", enable_runtime_probes=False)
        with self.assertRaisesRegex(ValueError, "TLS and bearer"):
            serve(host="0.0.0.0", allow_remote=True, session_token="secret",
                  enable_runtime_probes=False)
        with self.assertRaisesRegex(ValueError, "provided together"):
            serve(host="127.0.0.1", tls_cert=Path("certificate.pem"),
                  enable_runtime_probes=False)
        with self.assertRaisesRegex(ValueError, "client CA"):
            serve(host="127.0.0.1", tls_client_ca=Path("ca.pem"),
                  enable_runtime_probes=False)
        with self.assertRaisesRegex(ValueError, "client policy"):
            serve(
                host="127.0.0.1",
                tls_cert=Path("certificate.pem"),
                tls_key=Path("private-key.pem"),
                tls_client_policy=Path("policy.json"),
                enable_runtime_probes=False,
            )

    def test_tls_context_rejects_public_key_and_loads_private_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certificate, key = root / "certificate.pem", root / "private-key.pem"
            certificate.write_text("certificate")
            key.write_text("key")
            certificate.chmod(0o644)
            key.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                _server_tls_context(certificate, key)
            key.chmod(0o600)
            context = Mock()
            context.minimum_version = None
            with patch("vllm_apple.daemon.ssl.SSLContext", return_value=context):
                self.assertIs(_server_tls_context(certificate, key), context)
            context.load_cert_chain.assert_called_once_with(certificate, key)
            self.assertIsNotNone(context.minimum_version)

    def test_client_ca_enables_required_mutual_tls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certificate, key, ca = (
                root / "certificate.pem", root / "private-key.pem", root / "client-ca.pem")
            for path in (certificate, key, ca):
                path.write_text(path.name)
            key.chmod(0o600)
            context = Mock()
            context.minimum_version = None
            context.verify_mode = None
            with patch("vllm_apple.daemon.ssl.SSLContext", return_value=context):
                _server_tls_context(certificate, key, client_ca=ca)
            context.load_verify_locations.assert_called_once_with(cafile=str(ca))
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)


if __name__ == "__main__":
    unittest.main()
