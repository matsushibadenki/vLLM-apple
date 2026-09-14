import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from vllm_apple.device_benchmark import (
    DeviceBenchmarkConfig,
    DeviceBenchmarkMeasurement,
    DeviceBenchmarkReport,
    save_device_benchmark,
)
from vllm_apple.device_placement import load_device_placement_plan
from vllm_apple.device_placement_cli import main
from vllm_apple.execution import ExecutionBackend, WorkloadPhase


def report(backend, latency, capability):
    config = DeviceBenchmarkConfig(
        "matmul", backend, WorkloadPhase.PREFILL, "fp32", (8, 8, 8), 1, 3
    )
    samples = tuple(
        DeviceBenchmarkMeasurement(
            latency, latency, 0, 0, 512, 100, None, "a" * 64
        )
        for _ in range(3)
    )
    return DeviceBenchmarkReport(
        "m4-test", "environment-test", capability, config, samples, None
    )


class DevicePlacementCLITests(unittest.TestCase):
    def test_promotes_strict_reports_to_current_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            baseline = save_device_benchmark(
                report(ExecutionBackend.CPU, 100, "1" * 24), root / "cpu.json"
            )
            candidate = save_device_benchmark(
                report(ExecutionBackend.NATIVE_MLX, 50, "2" * 24), root / "mlx.json"
            )
            current = root / "current.json"
            last_good = root / "last-good.json"
            output = io.StringIO()
            with redirect_stdout(output):
                result = main([
                    "--hardware-fingerprint", "m4-test",
                    "--environment-fingerprint", "environment-test",
                    "--baseline-report", str(baseline),
                    "--candidate-report", str(candidate),
                    "--current-plan", str(current),
                    "--last-known-good-plan", str(last_good),
                    "--ttl-seconds", "60",
                    "--language", "ja",
                ])
            self.assertEqual(result, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["selected_backend"], "native_mlx")
            self.assertEqual(payload["language"], "ja")
            self.assertEqual(payload["message"], "デバイス配置プランを昇格しました。")
            loaded = load_device_placement_plan(
                current,
                hardware_fingerprint="m4-test",
                environment_fingerprint="environment-test",
            )
            self.assertEqual(loaded.plan_id, payload["plan_id"])

    def test_rejects_non_cpu_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            baseline = save_device_benchmark(
                report(ExecutionBackend.NATIVE_MLX, 50, "2" * 24), root / "mlx.json"
            )
            candidate = save_device_benchmark(
                report(ExecutionBackend.CPU, 100, "1" * 24), root / "cpu.json"
            )
            with self.assertRaisesRegex(ValueError, "must use CPU"):
                main([
                    "--hardware-fingerprint", "m4-test",
                    "--environment-fingerprint", "environment-test",
                    "--baseline-report", str(baseline),
                    "--candidate-report", str(candidate),
                    "--current-plan", str(root / "current.json"),
                    "--last-known-good-plan", str(root / "last-good.json"),
                ])


if __name__ == "__main__":
    unittest.main()
