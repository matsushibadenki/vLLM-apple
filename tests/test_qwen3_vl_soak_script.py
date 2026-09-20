import ast
import unittest
from pathlib import Path


class Qwen3VLSoakScriptTests(unittest.TestCase):
    def test_soak_script_is_bounded_and_atomic(self):
        path = Path(__file__).parents[1] / "scripts" / "soak_qwen3_vl_coreml.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("arguments.duration <= 7200", source)
        self.assertIn("arguments.duration < 1800", source)
        self.assertIn("repetitions=10", source)
        self.assertIn("os.replace(temporary, path)", source)
        self.assertIn('"digest_mismatches"', source)
        self.assertIn('"shutdown_clean"', source)


if __name__ == "__main__":
    unittest.main()
