import io
import json
import unittest
from unittest.mock import Mock, patch

from vllm_apple.coreml_worker import CoreMLPersistentWorker


class CoreMLPersistentWorkerTests(unittest.TestCase):
    def worker(self, lines):
        process = Mock()
        process.stdin = io.BytesIO()
        process.stdout = io.BytesIO(b"".join(
            (json.dumps(value) + "\n").encode() for value in lines
        ))
        process.wait.return_value = 0
        with patch("vllm_apple.coreml_worker.subprocess.Popen", return_value=process), patch(
            "vllm_apple.coreml_worker.select.select",
            side_effect=lambda streams, _w, _x, _timeout: (streams, (), ()),
        ):
            instance = CoreMLPersistentWorker.__new__(CoreMLPersistentWorker)
            instance._timeout = 1
            instance._maximum_output = 4096
            instance._expected_count = 2
            import threading
            instance._lock = threading.RLock()
            instance._closed = False
            instance._process = process
            return instance

    def test_predict_decodes_bounded_response_and_reuses_process(self):
        worker = self.worker([
            {"status": "ok", "latency_nanoseconds": 12, "output_values": [2, 4]},
            {"status": "ok", "latency_nanoseconds": 10, "output_values": [6, 8]},
        ])
        with patch(
            "vllm_apple.coreml_worker.select.select",
            side_effect=lambda streams, _w, _x, _timeout: (streams, (), ()),
        ):
            self.assertEqual(worker.predict((1.0, 2.0)).values, (2, 4))
            self.assertEqual(worker.predict((3.0, 4.0)).values, (6, 8))
        self.assertEqual(worker._process.stdin.getvalue().count(b"\n"), 2)

    def test_timeout_closes_worker(self):
        worker = self.worker([])
        with patch("vllm_apple.coreml_worker.select.select", return_value=((), (), ())):
            with self.assertRaises(TimeoutError):
                worker.predict((1.0, 2.0))
        self.assertTrue(worker._closed)
