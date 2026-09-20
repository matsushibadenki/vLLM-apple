import unittest

from vllm_apple.qwen3_vl_pipeline_coreml import _PIPELINE_PROGRAM


class Qwen3VLPipelineCoreMLTests(unittest.TestCase):
    def test_pipeline_directly_hands_main_output_to_next_stage(self):
        self.assertIn("current = main", _PIPELINE_PROGRAM)
        self.assertIn("for index in 0..<4", _PIPELINE_PROGRAM)
        self.assertIn('index == 3 ? "final_hidden_states"', _PIPELINE_PROGRAM)
        self.assertIn("getrusage(RUSAGE_SELF", _PIPELINE_PROGRAM)
        self.assertIn("output_sha256", _PIPELINE_PROGRAM)
        self.assertIn('for: "patch_hidden_states"', _PIPELINE_PROGRAM)
        self.assertIn('"pixel_values": MLFeatureValue', _PIPELINE_PROGRAM)

    def test_pipeline_reuses_models_and_isolates_request_outputs(self):
        self.assertIn("for path in CommandLine.arguments[16...]", _PIPELINE_PROGRAM)
        self.assertIn("for (requestIndex, input) in inputs.enumerated()", _PIPELINE_PROGRAM)
        self.assertIn('"request_\\(requestIndex)"', _PIPELINE_PROGRAM)
        self.assertIn('"model_load_count"', _PIPELINE_PROGRAM)
        self.assertIn('"request_count"', _PIPELINE_PROGRAM)


if __name__ == "__main__":
    unittest.main()
