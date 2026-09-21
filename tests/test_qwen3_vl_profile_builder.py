import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_PATH = Path(__file__).parents[1] / "scripts" / "build_qwen3_vl_coreml_profile.py"
_SPEC = importlib.util.spec_from_file_location("qwen3_vl_profile_builder", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


class Qwen3VLProfileBuilderTests(unittest.TestCase):
    def test_destination_must_be_new(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "destination"):
                module.build_profile(model, "a" * 40, output)

    def test_failure_removes_atomic_staging_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            output = root / "output"
            with patch.object(module, "inspect_qwen3_vl_vision_for_ane", return_value=object()), patch.object(
                module, "build_qwen3_vl_coreml_conversion_plan", side_effect=RuntimeError("stop")
            ):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    module.build_profile(model, "a" * 40, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.*")), [])


if __name__ == "__main__":
    unittest.main()
