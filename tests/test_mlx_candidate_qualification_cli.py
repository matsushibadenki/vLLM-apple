import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from vllm_apple.cli import main
from vllm_apple.compat import MLXBackendCompatibility


class MLXCandidateQualificationTests(unittest.TestCase):
    def test_exact_candidate_only_relaxes_version_matrix_for_qualification(self):
        for requested, issues, success in (
            (None, ("mlx_lm_version_outside_verified_matrix",), False),
            ("0.32.0", ("mlx_lm_version_outside_verified_matrix",), True),
            ("0.32.1", ("mlx_lm_version_outside_verified_matrix",), False),
            ("0.32.0", ("mlx_lm_version_outside_verified_matrix", "probe_failed"), False),
        ):
            args = ["qualify-model", "fixture", "--backend-kind", "mlx_lm",
                    "--backend-executable", "/unused"]
            if requested:
                args += ["--candidate-mlx-lm-version", requested]
            with self.subTest(requested=requested, issues=issues), patch(
                "vllm_apple.cli.inspect_mlx_lm_backend",
                return_value=MLXBackendCompatibility("/unused", "0.32.0", False, issues),
            ), patch("vllm_apple.cli.qualify_model", return_value={"passed": True}) as runner, patch(
                "vllm_apple.cli.save_qualification_report"
            ), redirect_stdout(StringIO()):
                self.assertEqual(main(args), 0 if success else 2)
                self.assertEqual(runner.call_count, int(success))
                if success:
                    self.assertEqual(runner.call_args.args[0].backend_versions, {"mlx_lm": "0.32.0"})
