import unittest
from unittest.mock import Mock, patch

from vllm_apple.qwen3_vl_persistent_worker import (
    Qwen3VLPersistentWorker,
    _VISION_WORKER_PROGRAM,
)


class Qwen3VLPersistentWorkerTests(unittest.TestCase):
    def test_worker_loads_once_and_has_bounded_protocol(self):
        self.assertIn("for index in 1...5", _VISION_WORKER_PROGRAM)
        self.assertIn('operation == "shutdown"', _VISION_WORKER_PROGRAM)
        self.assertIn('operation == "predict"', _VISION_WORKER_PROGRAM)
        self.assertIn('"peak_rss_bytes"', _VISION_WORKER_PROGRAM)
        self.assertIn("autoreleasepool", _VISION_WORKER_PROGRAM)

    def test_invalid_json_stops_worker_before_error(self):
        worker = object.__new__(Qwen3VLPersistentWorker)
        worker._timeout = 1
        worker._process = Mock()
        worker._process.stdout.readline.return_value = b"not-json\n"
        worker._stop = Mock()
        with patch("select.select", return_value=([worker._process.stdout], [], [])):
            with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                worker._read()
        worker._stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
