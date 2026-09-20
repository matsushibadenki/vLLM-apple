import unittest

from vllm_apple.qwen3_vl_persistent_worker import _VISION_WORKER_PROGRAM


class Qwen3VLPersistentWorkerTests(unittest.TestCase):
    def test_worker_loads_once_and_has_bounded_protocol(self):
        self.assertIn("for index in 1...5", _VISION_WORKER_PROGRAM)
        self.assertIn('operation == "shutdown"', _VISION_WORKER_PROGRAM)
        self.assertIn('operation == "predict"', _VISION_WORKER_PROGRAM)
        self.assertIn('"peak_rss_bytes"', _VISION_WORKER_PROGRAM)
        self.assertIn("autoreleasepool", _VISION_WORKER_PROGRAM)


if __name__ == "__main__":
    unittest.main()
