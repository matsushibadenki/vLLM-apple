import unittest
from pathlib import Path


class MetalQualificationWorkflowTests(unittest.TestCase):
    def test_artifact_admission_precedes_backend_and_model_load(self) -> None:
        workflow = Path(".github/workflows/metal-qualification.yml").read_text()
        admission = workflow.index("python3 -m vllm_apple artifact-admission")
        preflight = workflow.index("qualification-preflight")
        qualification = workflow.index("qualify-model \"$QUALIFICATION_MODEL\"")
        self.assertLess(admission, preflight)
        self.assertLess(preflight, qualification)
        self.assertIn("artifact-size-bytes and estimated-resident-bytes must be provided together", workflow)
        self.assertIn("> qualification-results/artifact-admission.json", workflow)
        self.assertIn("evidence+=(--require-artifact-admission)", workflow)

    def test_candidate_promotion_requires_exact_stack_and_full_evidence(self) -> None:
        workflow = Path(".github/workflows/vllm-stack-promotion.yml").read_text()
        preflight = workflow.index("qualification-preflight")
        integrity_before = workflow.index("promotion model provenance before load")
        qualification = workflow.index("qualify-model \"$PROMOTION_MODEL\"")
        integrity_after = workflow.index("promotion model provenance after qualification")
        sdk_gate = workflow.index("VLLMAppleQualificationCheck")
        bundle = workflow.index("qualification-bundle")
        self.assertLess(preflight, qualification)
        self.assertLess(preflight, integrity_before)
        self.assertLess(integrity_before, qualification)
        self.assertLess(qualification, integrity_after)
        self.assertLess(integrity_after, sdk_gate)
        self.assertLess(qualification, sdk_gate)
        self.assertLess(sdk_gate, bundle)
        self.assertIn("--candidate-vllm-version", workflow)
        self.assertIn("--candidate-vllm-metal-version", workflow)
        self.assertIn("--candidate-transformers-version", workflow)
        self.assertIn("--duration 1800", workflow)
        self.assertIn("--require-text-only", workflow)
        self.assertIn("--require-backend-versions", workflow)
        self.assertIn('--expected-vllm-version="$PROMOTION_VLLM"', workflow)
        self.assertIn('--expected-vllm-metal-version="$PROMOTION_VLLM_METAL"', workflow)
        self.assertIn('--expected-transformers-version="$PROMOTION_TRANSFORMERS"', workflow)
        self.assertEqual(workflow.count("--expected-signer-sha256"), 2)
        self.assertIn("all signed integrity inputs must be provided together", workflow)
        self.assertIn("qualification-private/model-integrity-manifest.json", workflow)

    def test_qwen_workflow_fails_closed_before_large_model_load(self) -> None:
        workflow = Path(
            ".github/workflows/qwen-flash-next-qualification.yml"
        ).read_text()
        admission = workflow.index("python3 -m vllm_apple artifact-admission")
        weight_map = workflow.index("qwen4-weight-map-inspect")
        conversion_plan = workflow.index("qwen4-conversion-plan")
        cache_fixture = workflow.index("qwen4-cache-fixture")
        preflight = workflow.index("python3 -m vllm_apple qualification-preflight")
        signed_preflight = workflow.index("Verify signed artifact provenance before load")
        integrity_create = workflow.index("model-integrity-create")
        qualification = workflow.index("qualify-model \"$QWEN_MODEL\"")
        ephemeral_verify = workflow.index(
            "Verify the identical ephemeral model tree after qualification"
        )
        signed_reverify = workflow.index(
            "Reverify signed artifact provenance after qualification"
        )
        sdk_gate = workflow.index("VLLMAppleQualificationCheck")
        bundle = workflow.index("qualification-bundle")
        self.assertLess(admission, preflight)
        self.assertLess(admission, weight_map)
        self.assertLess(weight_map, conversion_plan)
        self.assertLess(conversion_plan, cache_fixture)
        self.assertLess(cache_fixture, preflight)
        self.assertLess(preflight, signed_preflight)
        self.assertLess(preflight, integrity_create)
        self.assertLess(integrity_create, qualification)
        self.assertLess(qualification, ephemeral_verify)
        self.assertLess(qualification, signed_reverify)
        self.assertLess(ephemeral_verify, sdk_gate)
        self.assertLess(signed_reverify, sdk_gate)
        self.assertLess(sdk_gate, bundle)
        self.assertIn("large-memory", workflow)
        self.assertIn("--mode text", workflow)
        self.assertIn('qwen4-weight-map.json', workflow)
        self.assertIn('qwen4-conversion-plan.json', workflow)
        self.assertIn('qwen4-cache-fixture.json', workflow)
        self.assertIn("--model \"$QWEN_MODEL\"", workflow)
        self.assertIn("--max-model-len \"$QWEN_MAX_MODEL_LEN\"", workflow)
        self.assertIn("--duration 1800", workflow)
        self.assertIn("--require-artifact-admission", workflow)
        self.assertIn("inputs['upload-report']", workflow)
        self.assertIn("qualification-private/model-integrity-manifest.json", workflow)
        self.assertEqual(workflow.count("path: qualification-results/*.json"), 1)
        self.assertEqual(workflow.count("--expected-signer-sha256"), 2)
        self.assertIn("all signed integrity inputs must be provided together", workflow)


if __name__ == "__main__":
    unittest.main()
