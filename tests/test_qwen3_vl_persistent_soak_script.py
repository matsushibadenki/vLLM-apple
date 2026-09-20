import ast
import unittest
from pathlib import Path


class Qwen3VLPersistentSoakScriptTests(unittest.TestCase):
    def test_script_is_bounded_and_requires_clean_shutdown(self):
        path = Path(__file__).parents[1] / "scripts" / "soak_qwen3_vl_persistent_worker.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("arguments.duration <= 7200", source)
        self.assertIn("arguments.duration < 1800", source)
        self.assertIn("len(latency_samples) > 256", source)
        self.assertIn("shutdown_clean = worker.close()", source)
        self.assertIn("worker.restart_count == 0", source)


if __name__ == "__main__":
    unittest.main()
