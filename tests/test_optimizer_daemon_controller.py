from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from vllm_apple.api import create_server
from vllm_apple.optimizer.daemon_controller import OptimizerDaemonController
from vllm_apple.service import RuntimeService


class _EchoOptimizerController:
    def handle(self, operation: str, payload: bytes) -> bytes:
        if operation != "plan":
            raise ValueError("operation disabled")
        return b'{"dry_run":true}' + payload


class OptimizerDaemonAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(
            "127.0.0.1", 0, RuntimeService(), optimizer_controller=_EchoOptimizerController()
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/v1/optimizer"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _post(self, body: dict[str, object]) -> dict[str, object]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            return json.load(response)

    def test_versioned_optimizer_envelope_round_trips_bounded_payload(self) -> None:
        request_id = str(uuid.uuid4())
        result = self._post({
            "schema_version": 1,
            "request_id": request_id,
            "operation": "plan",
            "payload": base64.b64encode(b"{}").decode(),
        })
        self.assertEqual(result["request_id"], request_id)
        self.assertTrue(result["succeeded"])
        self.assertEqual(base64.b64decode(result["payload"]), b'{"dry_run":true}{}')

    def test_optimizer_envelope_rejects_unknown_fields_and_operation(self) -> None:
        for body in (
            {
                "schema_version": 1, "request_id": str(uuid.uuid4()),
                "operation": "delete", "payload": None,
            },
            {
                "schema_version": 1, "request_id": str(uuid.uuid4()),
                "operation": "plan", "payload": None, "extra": "secret",
            },
        ):
            request = urllib.request.Request(
                self.endpoint, data=json.dumps(body).encode(), method="POST"
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=2)
            self.assertEqual(raised.exception.code, 400)


class OptimizerDaemonControllerTests(unittest.TestCase):
    def test_plan_builds_real_dry_run_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            models = root / "models"
            output = root / "output"
            model = models / "tiny"
            model.mkdir(parents=True)
            output.mkdir()
            (model / "config.json").write_text(json.dumps({
                "model_type": "gpt2",
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "hidden_size": 8,
                "max_position_embeddings": 128,
                "torch_dtype": "float32",
            }))
            header = json.dumps({
                "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
            }).encode()
            (model / "model.safetensors").write_bytes(
                len(header).to_bytes(8, "little") + header + bytes(4)
            )
            destination = output / "artifact"
            controller = OptimizerDaemonController((models,), (output,))
            request = {
                "model_path": str(model),
                "output_path": str(destination),
                "objective": "balanced",
                "maximum_memory_bytes": 1_073_741_824,
                "maximum_disk_bytes": 1_073_741_824,
                "maximum_duration_seconds": 60,
                "license": "test-only",
            }
            plan = json.loads(controller.handle("plan", json.dumps(request).encode()))
            self.assertEqual(plan["schema_version"], 1)
            self.assertTrue(plan["dry_run"])
            self.assertEqual(plan["output_path"], str(destination.resolve()))
            self.assertFalse(destination.exists())

    def test_roots_must_be_owned_real_directories(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            link = root / "link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                OptimizerDaemonController((link,), (root,))

    def test_plan_rejects_paths_outside_allowlist_before_model_inspection(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            models = root / "models"
            output = root / "output"
            models.mkdir()
            output.mkdir()
            controller = OptimizerDaemonController((models,), (output,))
            request = {
                "model_path": "/tmp",
                "output_path": str(output / "artifact"),
                "objective": "balanced",
                "maximum_memory_bytes": 1,
                "maximum_disk_bytes": 1,
                "maximum_duration_seconds": None,
                "license": None,
            }
            with self.assertRaisesRegex(ValueError, "outside allowed roots"):
                controller.handle("plan", json.dumps(request).encode())

    def test_output_root_must_be_writable(self) -> None:
        if os.getuid() == 0:
            self.skipTest("root bypasses mode-based write checks")
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            models = root / "models"
            output = root / "output"
            models.mkdir()
            output.mkdir(mode=0o500)
            try:
                with self.assertRaises(ValueError):
                    OptimizerDaemonController((models,), (output,))
            finally:
                output.chmod(0o700)


if __name__ == "__main__":
    unittest.main()
