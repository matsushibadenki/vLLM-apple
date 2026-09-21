import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.native_hardware_benchmark import (
    run_native_hardware_benchmarks,
    save_native_hardware_benchmarks,
)


class NativeHardwareBenchmarkTests(unittest.TestCase):
    def test_strict_response_is_normalized_and_saved_privately(self):
        reports = [{
            "operator": operator, "nanoseconds": [10] * 7,
            "work_items": 100, "bytes": 200, "digest": operator,
        } for operator in (
            "gpu_gemm", "gpu_gemv", "unified_memory_copy", "metal_launch",
            "attention", "int8_matmul", "packed_u4_unpack", "nf4_gemv_fused",
            "nf4_decode_gemv_two_pass",
            "nf4_gemm_fused", "nf4_decode_gemm_two_pass", "nf4_attention_fused",
            "nf4_decode_attention_three_pass",
        )]
        response = SimpleNamespace(returncode=0, stdout=json.dumps({
            "device": "Apple Test", "samples": 7, "reports": reports,
        }))
        with patch("subprocess.run", return_value=response):
            result = run_native_hardware_benchmarks()
        self.assertTrue(result["passed"])
        self.assertEqual(result["reports"][0]["median_nanoseconds"], 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private"
            path = save_native_hardware_benchmarks(result, root / "report.json")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_missing_or_duplicate_operator_fails_closed(self):
        report = {"operator": "gpu_gemm", "nanoseconds": [1] * 7,
                  "work_items": 1, "bytes": 0, "digest": "x"}
        response = SimpleNamespace(returncode=0, stdout=json.dumps({
            "device": "Apple Test", "samples": 7, "reports": [report] * 13,
        }))
        with patch("subprocess.run", return_value=response), self.assertRaises(ValueError):
            run_native_hardware_benchmarks()


if __name__ == "__main__":
    unittest.main()
