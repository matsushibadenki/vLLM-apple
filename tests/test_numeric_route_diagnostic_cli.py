import contextlib
import io
import json
import unittest

from vllm_apple.cli import main


class NumericRouteDiagnosticCLITests(unittest.TestCase):
    def run_cli(self, language):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main([
                "numeric-route-diagnostic",
                "--source-format", "nvfp4_e2m1",
                "--runtime-format", "int8",
                "--compute-format", "fp16",
                "--tensor-role", "weight",
                "--route", "cached_convert",
                "--absolute-error-budget", "0.02",
                "--rmse-budget", "0.01",
                "--fallback-reason", "probe_unavailable",
                "--language", language,
            ])
        return code, json.loads(output.getvalue())

    def test_three_language_machine_readable_diagnostics(self):
        messages = set()
        for language in ("en", "ja", "zh-Hans"):
            code, report = self.run_cli(language)
            self.assertEqual(code, 0)
            self.assertTrue(report["valid"])
            self.assertEqual(report["language"], language)
            self.assertEqual(report["source_format"], "nvfp4_e2m1")
            self.assertEqual(report["runtime_format"], "int8")
            self.assertEqual(report["compute_format"], "fp16")
            self.assertEqual(report["fallback_reason"], "probe_unavailable")
            messages.add(report["message"])
        self.assertEqual(len(messages), 3)

    def test_invalid_error_budget_fails_closed(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main([
                "numeric-route-diagnostic", "--source-format", "int8",
                "--runtime-format", "int8", "--compute-format", "int8",
                "--tensor-role", "kv_state", "--route", "load_convert",
                "--absolute-error-budget", "nan", "--rmse-budget", "0",
            ])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(output.getvalue())["valid"])


if __name__ == "__main__":
    unittest.main()
