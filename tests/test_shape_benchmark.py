import hashlib
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tests.schema_validator import validate_instance
from vllm_apple.cli import main
from vllm_apple.execution import ExecutionBackend
from vllm_apple.kernel_probe import KernelMeasurement, KernelProbeConfig, run_kernel_probe
from vllm_apple.kernel_profile import ModelKernelShapeProfile, PagedAttentionShape
from vllm_apple.shape_benchmark import (
    default_metal_shape_benchmark_path,
    load_metal_shape_benchmark,
    run_metal_shape_benchmark,
    save_metal_shape_benchmark,
)


class FakeMetalShapeAdapter:
    def probe_shape_profile(self, profile, **options):
        digest = hashlib.sha256(b"same").hexdigest()
        shape = profile.shapes[0]
        return (
            run_kernel_probe(
                KernelProbeConfig(
                    options["hardware_fingerprint"],
                    options["environment_fingerprint"],
                    ExecutionBackend.NATIVE_METAL,
                    f"paged_attention:c{shape.context_tokens}:d{shape.head_dimension}:q4:kv2:b4",
                    samples=options["samples"],
                    maximum_slowdown_ratio=20,
                ),
                lambda: KernelMeasurement(digest, 100),
                lambda: KernelMeasurement(digest, 120),
            ),
        )


class MetalShapeBenchmarkTests(unittest.TestCase):
    def profile(self):
        shape = PagedAttentionShape(1024, 1, 4, 2, 8, 4, 256, 32768)
        return ModelKernelShapeProfile(1, "a" * 24, "model", "test", 2, 2, (shape,))

    def test_report_is_deterministic_profile_bound_and_schema_valid(self) -> None:
        benchmark = run_metal_shape_benchmark(
            self.profile(),
            FakeMetalShapeAdapter(),  # type: ignore[arg-type]
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            clock=lambda: 1000,
        )
        repeated = run_metal_shape_benchmark(
            self.profile(),
            FakeMetalShapeAdapter(),  # type: ignore[arg-type]
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            clock=lambda: 2000,
        )
        self.assertEqual(benchmark.benchmark_id, repeated.benchmark_id)
        self.assertEqual(benchmark.created_at_unix_seconds, 1000)
        schema = json.loads(
            Path("schemas/runtime/metal-shape-benchmark-v1.schema.json").read_text()
        )
        validate_instance(benchmark.to_dict(), schema)

    def test_save_is_private_atomic_and_rejects_public_parent(self) -> None:
        benchmark = run_metal_shape_benchmark(
            self.profile(),
            FakeMetalShapeAdapter(),  # type: ignore[arg-type]
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            clock=lambda: 1000,
        )
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory) / "private"
            destination = private / "benchmark.json"
            save_metal_shape_benchmark(benchmark, destination)
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            self.assertEqual(json.loads(destination.read_text()), benchmark.to_dict())
            loaded = load_metal_shape_benchmark(
                destination,
                profile_id=benchmark.profile_id,
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
            )
            self.assertEqual(loaded, benchmark)
            with self.assertRaises(ValueError):
                load_metal_shape_benchmark(
                    destination,
                    profile_id="b" * 24,
                    hardware_fingerprint="hardware",
                    environment_fingerprint="environment",
                )
            public = Path(directory) / "public"
            public.mkdir(mode=0o755)
            with self.assertRaises(ValueError):
                save_metal_shape_benchmark(benchmark, public / "benchmark.json")

    def test_default_path_is_profile_and_environment_scoped(self) -> None:
        benchmark = run_metal_shape_benchmark(
            self.profile(),
            FakeMetalShapeAdapter(),  # type: ignore[arg-type]
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            clock=lambda: 1000,
        )
        with patch(
            "vllm_apple.shape_benchmark.default_application_support",
            return_value=Path("/private/app-support"),
        ):
            path = default_metal_shape_benchmark_path(benchmark)
        self.assertEqual(
            path,
            Path("/private/app-support/profiles/metal-shapes/hardware/environment")
            / f"{benchmark.profile_id}-{benchmark.benchmark_id}.json",
        )

    def test_cli_builds_profile_and_emits_benchmark_without_saving(self) -> None:
        benchmark = run_metal_shape_benchmark(
            self.profile(),
            FakeMetalShapeAdapter(),  # type: ignore[arg-type]
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            clock=lambda: 1000,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "hidden_size": 32,
                    }
                )
            )
            (root / "model.safetensors").write_bytes(b"weights")
            output = StringIO()
            with (
                patch("vllm_apple.cli.detect_apple_chip_profile") as detect,
                patch("vllm_apple.cli.discover_runtime_versions") as versions,
                patch("vllm_apple.cli.run_metal_shape_benchmark", return_value=benchmark),
                redirect_stdout(output),
            ):
                detect.return_value.platform = "Darwin"
                detect.return_value.os_version = "macOS"
                detect.return_value.hardware_fingerprint = "hardware"
                versions.return_value.toolchain_version = "metal"
                versions.return_value.mlx_version = "mlx"
                versions.return_value.backend_version = "backend"
                status = main(
                    ["metal-shape-benchmark", str(root), "--contexts", "1024", "--stdout"]
                )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue()), benchmark.to_dict())


if __name__ == "__main__":
    unittest.main()
