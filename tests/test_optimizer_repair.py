import tempfile
import unittest
from pathlib import Path

from vllm_apple.optimizer.evaluation import PerplexityEvaluationReport, PerplexitySlice
from vllm_apple.optimizer.repair import (
    RepairArtifact,
    RepairMethod,
    RepairRequest,
    run_repair_and_evaluate,
)


def report(path, model_hash, perplexity):
    item = PerplexitySlice("general", "en", 1, 10, 1.0, perplexity)
    return PerplexityEvaluationReport(
        str(path), model_hash, "/tmp/data.jsonl", "d" * 64,
        1, 10, 1.0, perplexity, 1, 1, (item,),
    )


class Adapter:
    def __init__(self, artifact):
        self.artifact = artifact

    def repair(self, request):
        return self.artifact


class OptimizerRepairTests(unittest.TestCase):
    def test_repair_requires_matching_before_and_after_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repaired"
            path.mkdir()
            request = RepairRequest(
                RepairMethod.LORA, "a" * 64, "d" * 64, 100, 2, 1e-4, 8, 42
            )
            artifact = RepairArtifact(path, "b" * 64, "a" * 64, RepairMethod.LORA)
            result = run_repair_and_evaluate(
                Adapter(artifact), request, report(path, "a" * 64, 10.0),
                lambda candidate: report(candidate, "b" * 64, 9.0),
            )
            self.assertTrue(result.approved)
            self.assertAlmostEqual(result.relative_change, -.1)

    def test_regressing_repair_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repaired"
            path.mkdir()
            request = RepairRequest(
                RepairMethod.SFT, "a" * 64, "d" * 64, 100, 1, 1e-4, None, 0
            )
            artifact = RepairArtifact(path, "b" * 64, "a" * 64, RepairMethod.SFT)
            result = run_repair_and_evaluate(
                Adapter(artifact), request, report(path, "a" * 64, 10.0),
                lambda candidate: report(candidate, "b" * 64, 10.1),
            )
            self.assertFalse(result.approved)

    def test_invalid_method_parameters_fail_closed(self):
        with self.assertRaises(ValueError):
            RepairRequest(RepairMethod.LORA, "a" * 64, "d" * 64, 1, 1, 1e-4, None, 0)


if __name__ == "__main__":
    unittest.main()
