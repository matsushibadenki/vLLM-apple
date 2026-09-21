import sys
import unittest

from vllm_apple.qwen_image_21_conversion_subprocess import (
    run_qwen_image_21_conversion_worker,
)


class QwenImage21ConversionSubprocessTests(unittest.TestCase):
    def test_bounded_worker_report_is_returned(self) -> None:
        report = {"passed": True, "artifact_root_sha256": "a" * 64}
        result = run_qwen_image_21_conversion_worker(
            sys.executable,
            "source",
            "output",
            timeout_seconds=5,
            command=(sys.executable, "-c", f"import json; print(json.dumps({report!r}))"),
        )
        self.assertEqual(result, report)

    def test_oversized_output_is_stopped(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "output exceeded limit"):
            run_qwen_image_21_conversion_worker(
                sys.executable,
                "source",
                "output",
                timeout_seconds=5,
                command=(sys.executable, "-c", "print('x' * 70000)"),
            )

    def test_nonzero_exit_surfaces_bounded_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "status 3"):
            run_qwen_image_21_conversion_worker(
                sys.executable,
                "source",
                "output",
                timeout_seconds=5,
                command=(sys.executable, "-c", "import sys; print('failed', file=sys.stderr); sys.exit(3)"),
            )
