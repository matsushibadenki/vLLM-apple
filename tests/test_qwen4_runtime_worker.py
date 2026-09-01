import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_qwen4_adapter_loader as loader_tests
from tests import test_qwen4_resident_store as resident_tests
from vllm_apple.qwen4_runtime_worker import Qwen4RuntimeWorker
from vllm_apple.qwen4_shard_stager import stage_qwen4_shards


class Qwen4RuntimeWorkerTests(unittest.TestCase):
    def worker(self, root: Path) -> Qwen4RuntimeWorker:
        source = loader_tests.Qwen4AdapterLoaderTests().source(root)
        stage = root / "stage"
        stage_qwen4_shards(source, stage, maximum_output_bytes=65536)
        private = root / "private"
        private.mkdir(mode=0o700)
        return Qwen4RuntimeWorker(
            stage_root=stage,
            socket_path=private / "runtime.sock",
            session_file=private / "session.json",
            maximum_artifact_bytes=65536,
            memory_capacity_bytes=32,
            backend=resident_tests.FakeResidentBackend(),
        )

    def test_composes_verified_store_and_private_session_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = self.worker(root)
            with patch.object(worker.server, "start"):
                worker.start()
            payload = json.loads(worker.session_file.read_text())
            self.assertEqual(payload["session_id"], worker.session_id)
            self.assertEqual(worker.session_file.stat().st_mode & 0o777, 0o600)
            self.assertEqual(worker.service.handle(self._status(worker, 1))["result"]["resident_tensors"], 0)
            with patch.object(worker.server, "close"):
                snapshot = worker.close()
            self.assertEqual(snapshot["resident_tensors"], 0)
            self.assertFalse(worker.session_file.exists())

    def test_does_not_remove_replaced_session_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(Path(directory))
            with patch.object(worker.server, "start"):
                worker.start()
            worker.session_file.unlink()
            worker.session_file.write_text("replacement", encoding="utf-8")
            with patch.object(worker.server, "close"):
                worker.close()
            self.assertEqual(worker.session_file.read_text(), "replacement")

    @staticmethod
    def _status(worker: Qwen4RuntimeWorker, sequence: int) -> dict[str, object]:
        return {
            "abi_version": 1,
            "session_id": worker.session_id,
            "sequence": sequence,
            "request_id": f"{sequence:032x}",
            "operation": "status",
        }


if __name__ == "__main__":
    unittest.main()
