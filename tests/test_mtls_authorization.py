import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from vllm_apple.api import create_server
from vllm_apple.mtls_authorization import (
    ClientCertificatePolicy,
    ClientCertificatePolicyStore,
)
from vllm_apple.service import RuntimeService


def certificate(*, serial="01", common_name="client", sans=()):
    return {
        "serialNumber": serial,
        "subject": ((('commonName', common_name),),),
        "subjectAltName": tuple(sans),
    }


def payload(*, subjects=None, sans=None, revoked=None):
    return {
        "policy_version": 1,
        "allowed_subjects": subjects or [],
        "allowed_sans": sans or [],
        "revoked_serial_numbers": revoked or [],
    }


class ClientCertificateAuthorizationTests(unittest.TestCase):
    def test_subject_san_and_revocation_are_fail_closed(self):
        policy = ClientCertificatePolicy.from_dict(payload(
            subjects=["commonName=client"],
            sans=["DNS:worker.example"],
            revoked=["DEAD"],
        ))
        self.assertTrue(policy.authorize(certificate()))
        self.assertTrue(policy.authorize(certificate(
            common_name="other", sans=(("DNS", "worker.example"),)
        )))
        self.assertFalse(policy.authorize(certificate(serial="dead")))
        self.assertFalse(policy.authorize(certificate(common_name="other")))
        self.assertFalse(policy.authorize(None))

    def test_atomic_policy_rotation_and_invalid_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(payload(subjects=["commonName=first"])))
            path.chmod(0o600)
            store = ClientCertificatePolicyStore(path)
            self.assertTrue(store.authorize(certificate(common_name="first")))
            replacement = path.with_suffix(".new")
            replacement.write_text(json.dumps(payload(
                sans=["URI:spiffe://example/worker"]
            )))
            replacement.chmod(0o600)
            replacement.replace(path)
            self.assertFalse(store.authorize(certificate(common_name="first")))
            self.assertTrue(store.authorize(certificate(
                common_name="other",
                sans=(("URI", "spiffe://example/worker"),),
            )))
            path.write_text("not-json")
            self.assertFalse(store.authorize(certificate(
                common_name="other",
                sans=(("URI", "spiffe://example/worker"),),
            )))
            self.assertFalse(store.snapshot()["valid"])

    def test_http_boundary_requires_peer_certificate_when_policy_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(payload(subjects=["commonName=client"])))
            path.chmod(0o600)
            server = create_server(
                "127.0.0.1", 0, RuntimeService(),
                client_certificate_policy=ClientCertificatePolicyStore(path),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{server.server_port}/health", timeout=2
                    )
                self.assertEqual(raised.exception.code, 403)
                self.assertEqual(
                    json.load(raised.exception)["error"]["code"],
                    "client_certificate_forbidden",
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
