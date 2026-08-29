import json
import stat
import tempfile
import unittest
from pathlib import Path

from vllm_apple.long_context import save_long_context_report


class LongContextPersistenceTests(unittest.TestCase):
    def test_saves_private_atomic_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "private"
            output = parent / "report.json"
            save_long_context_report({"schema_version": 1}, output)
            self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(json.loads(output.read_text()), {"schema_version": 1})
            self.assertEqual(list(parent.glob(".report.json.*")), [])

    def test_rejects_shared_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "shared"
            parent.mkdir(mode=0o755)
            with self.assertRaisesRegex(ValueError, "directory must be private"):
                save_long_context_report({}, parent / "report.json")


if __name__ == "__main__":
    unittest.main()
