import unittest
from pathlib import Path


class MetalQualificationWorkflowTests(unittest.TestCase):
    def test_artifact_admission_precedes_backend_and_model_load(self) -> None:
        workflow = Path(".github/workflows/metal-qualification.yml").read_text()
        admission = workflow.index("python3 -m vllm_apple artifact-admission")
        preflight = workflow.index("python3 -m vllm_apple qualification-preflight")
        qualification = workflow.index("qualify-model \"$QUALIFICATION_MODEL\"")
        self.assertLess(admission, preflight)
        self.assertLess(preflight, qualification)
        self.assertIn("artifact-size-gib and estimated-resident-gib must be provided together", workflow)
        self.assertIn("> qualification-results/artifact-admission.json", workflow)
        self.assertIn("evidence+=(--require-artifact-admission)", workflow)


if __name__ == "__main__":
    unittest.main()
