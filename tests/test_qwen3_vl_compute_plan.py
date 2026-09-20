import tempfile
import unittest
from pathlib import Path

from vllm_apple.qwen3_vl_compute_plan import (
    _COMPUTE_PLAN_PROGRAM,
    inspect_qwen3_vl_coreml_compute_plans,
)


class Qwen3VLComputePlanTests(unittest.TestCase):
    def test_program_uses_public_compute_plan_device_usage(self):
        self.assertIn("MLComputePlan.load", _COMPUTE_PLAN_PROGRAM)
        self.assertIn("plan.deviceUsage(for: operation)", _COMPUTE_PLAN_PROGRAM)
        self.assertIn('case .neuralEngine: return "neural_engine"', _COMPUTE_PLAN_PROGRAM)

    def test_rejects_non_compiled_model_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model.txt"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "compiled model"):
                inspect_qwen3_vl_coreml_compute_plans((root,))


if __name__ == "__main__":
    unittest.main()
