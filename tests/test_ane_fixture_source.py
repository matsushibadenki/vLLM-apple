import ast
import unittest
from pathlib import Path


class ANEFixtureSourceTests(unittest.TestCase):
    def test_generator_is_deterministic_scale_only_model(self):
        path = Path(__file__).resolve().parents[1] / "scripts/generate_ane_probe_fixture.py"
        source = path.read_text()
        ast.parse(source)
        self.assertIn('name="double"', source)
        self.assertIn("np.array([2.0]", source)
        self.assertIn('datatypes.Array(4)', source)
        self.assertNotIn("random", source)

    def test_generator_has_bounded_representative_encoder_preset(self):
        path = Path(__file__).resolve().parents[1] / "scripts/generate_ane_probe_fixture.py"
        source = path.read_text()
        ast.parse(source)
        self.assertIn('choices=("scale", "encoder")', source)
        self.assertIn("add_inner_product", source)
        self.assertIn('non_linearity="RELU"', source)
        self.assertIn("MAX_ENCODER_WIDTH", source)

    def test_self_hosted_workflow_runs_fail_closed_heterogeneous_suite(self):
        path = (
            Path(__file__).resolve().parents[1]
            / ".github/workflows/ane-qualification.yml"
        )
        source = path.read_text()
        self.assertIn("mlx==0.31.2", source)
        self.assertIn("expected - passed", source)
        self.assertIn("maximum_slowdown_ratio=10_000", source)
        self.assertIn("run_device_benchmark_suite", source)
        self.assertIn("expected_values=config.expected_values", source)
        self.assertIn("save_device_benchmark", source)
        self.assertIn("actions/upload-artifact@v4", source)
        self.assertIn("retention-days: 14", source)
        self.assertIn("rm -rf qualification-private qualification-reports", source)


if __name__ == "__main__":
    unittest.main()
