import json
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.diffusers_generation_worker import WorkerTelemetry
from vllm_apple.mlx_gen_generation_worker import (
    LocalMLXGenImageRuntime,
    _BoundedProgressSink,
    _failure_code,
    execute_mlx_gen_image_request,
)
from vllm_apple.mlx_gen_memory_profile import MLXGenMemoryProfileError


class MLXGenGenerationWorkerTests(unittest.TestCase):
    def test_local_worker_routes_cli_in_process_and_removes_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            output = root / "output"
            model.mkdir()
            output.mkdir()
            seen = {}

            def backend_main():
                seen["argv"] = list(sys.argv)
                target = Path(sys.argv[sys.argv.index("--output") + 1])
                target.write_bytes(b"png")
                print(json.dumps({"event": "step", "step": 1}))

            runtime = LocalMLXGenImageRuntime(
                module_loader=lambda name: SimpleNamespace(main=backend_main)
            )
            request = {
                "candidate_id": "flux2-klein-9b-base",
                "modality": "image",
                "mode": "text-to-image",
                "model_root": str(model),
                "output_root": str(output),
                "prompt": "local test",
                "seed": 42,
                "width": 512,
                "height": 512,
                "steps": 20,
                "batch_size": 1,
                "sample_index": 0,
                "memory_hard_ceiling_bytes": 1024 * 1024,
            }
            events = []
            with patch("vllm_apple.mlx_gen_generation_worker.platform.system", return_value="Darwin"), patch(
                "vllm_apple.mlx_gen_generation_worker.platform.machine", return_value="arm64"
            ):
                execute_mlx_gen_image_request(
                    request,
                    runtime,
                    telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                    emit=events.append,
                )
            self.assertEqual(seen["argv"][0:2], ["mlxgen", "generate"])
            self.assertIn(str(model.resolve()), seen["argv"])
            self.assertIn("--low-ram", seen["argv"])
            self.assertNotIn("--vae-tiling", seen["argv"])
            self.assertFalse(list(output.glob("*.png")))
        self.assertEqual(
            [event.kind for event in events],
            ["started", "progress", "first_output", "completed"],
        )

    def test_local_worker_routes_z_image_through_the_bounded_generic_cli(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "z-image"
            output = root / "output"
            model.mkdir()
            output.mkdir()
            seen = {}

            def backend_main():
                seen["argv"] = list(sys.argv)
                Path(sys.argv[sys.argv.index("--output") + 1]).write_bytes(b"png")
                print(json.dumps({"event": "step", "step": 1}))

            runtime = LocalMLXGenImageRuntime(
                "z-image-turbo-mlx-4bit",
                module_loader=lambda name: SimpleNamespace(main=backend_main),
            )
            request = {
                "candidate_id": "z-image-turbo-mlx-4bit",
                "modality": "image",
                "mode": "text-to-image",
                "model_root": str(model),
                "output_root": str(output),
                "prompt": "local test",
                "seed": 42,
                "width": 512,
                "height": 512,
                "steps": 2,
                "batch_size": 1,
                "sample_index": 0,
                "memory_hard_ceiling_bytes": 1024 * 1024,
            }
            events = []
            with patch(
                "vllm_apple.mlx_gen_generation_worker.platform.system",
                return_value="Darwin",
            ), patch(
                "vllm_apple.mlx_gen_generation_worker.platform.machine",
                return_value="arm64",
            ):
                execute_mlx_gen_image_request(
                    request,
                    runtime,
                    telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                    emit=events.append,
                )
            self.assertEqual(runtime.pipeline_class, "MLXGenZImageTurbo")
            self.assertEqual(seen["argv"][0:2], ["mlxgen", "generate"])
            self.assertIn(str(model.resolve()), seen["argv"])
            self.assertEqual(
                seen["argv"][seen["argv"].index("--base-model") + 1],
                "Tongyi-MAI/Z-Image-Turbo",
            )
            self.assertEqual(seen["argv"][seen["argv"].index("--steps") + 1], "2")
            self.assertFalse(list(output.glob("*.png")))
        self.assertEqual(events[-1].kind, "completed")

    def test_low_cache_profile_forwards_the_bound_cache_limit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            output = root / "output"
            model.mkdir()
            output.mkdir()
            seen = {}

            def backend_main():
                seen["argv"] = list(sys.argv)
                Path(sys.argv[sys.argv.index("--output") + 1]).write_bytes(b"png")
                print(json.dumps({"event": "step", "step": 1}))

            runtime = LocalMLXGenImageRuntime(
                "flux2-klein-9b-base-low-cache",
                module_loader=lambda name: SimpleNamespace(main=backend_main),
            )
            request = {
                "candidate_id": "flux2-klein-9b-base-low-cache",
                "modality": "image",
                "mode": "text-to-image",
                "model_root": str(model),
                "output_root": str(output),
                "prompt": "local test",
                "seed": 42,
                "width": 512,
                "height": 512,
                "steps": 20,
                "batch_size": 1,
                "sample_index": 0,
                "memory_hard_ceiling_bytes": 1024 * 1024,
            }
            events = []
            with patch(
                "vllm_apple.mlx_gen_generation_worker.platform.system",
                return_value="Darwin",
            ), patch(
                "vllm_apple.mlx_gen_generation_worker.platform.machine",
                return_value="arm64",
            ):
                execute_mlx_gen_image_request(
                    request,
                    runtime,
                    telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                    emit=events.append,
                )
        index = seen["argv"].index("--mlx-cache-limit-gb")
        self.assertEqual(seen["argv"][index + 1], "0.25")
        self.assertEqual(events[-1].kind, "completed")

    def test_blockwise_profile_is_scoped_around_backend_execution(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            output = root / "output"
            model.mkdir()
            output.mkdir()
            profile_events = []
            seen = {}

            def backend_main():
                profile_events.append("backend")
                seen["argv"] = list(sys.argv)
                Path(sys.argv[sys.argv.index("--output") + 1]).write_bytes(b"png")
                print(json.dumps({"event": "step", "step": 1}))

            @contextmanager
            def profile(_module_loader):
                profile_events.append("enter")
                yield
                profile_events.append("exit")

            runtime = LocalMLXGenImageRuntime(
                "flux2-klein-9b-base-blockwise",
                module_loader=lambda name: SimpleNamespace(main=backend_main),
            )
            request = {
                "candidate_id": "flux2-klein-9b-base-blockwise",
                "modality": "image",
                "mode": "text-to-image",
                "model_root": str(model),
                "output_root": str(output),
                "prompt": "local test",
                "seed": 42,
                "width": 512,
                "height": 512,
                "steps": 20,
                "batch_size": 1,
                "sample_index": 0,
                "memory_hard_ceiling_bytes": 1024 * 1024,
            }
            events = []
            with patch(
                "vllm_apple.mlx_gen_generation_worker.platform.system", return_value="Darwin"
            ), patch(
                "vllm_apple.mlx_gen_generation_worker.platform.machine", return_value="arm64"
            ), patch(
                "vllm_apple.mlx_gen_generation_worker.flux2_blockwise_residency",
                side_effect=profile,
            ):
                execute_mlx_gen_image_request(
                    request,
                    runtime,
                    telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                    emit=events.append,
                )
        self.assertEqual(profile_events, ["enter", "backend", "exit"])
        self.assertNotIn("--mlx-cache-limit-gb", seen["argv"])
        self.assertEqual(events[-1].kind, "completed")

    def test_progress_sink_rejects_non_json_output(self) -> None:
        sink = _BoundedProgressSink(lambda: None)
        with self.assertRaisesRegex(RuntimeError, "non-JSON"):
            sink.write("unsafe diagnostic\n")

    def test_failure_code_distinguishes_the_worker_memory_ceiling(self) -> None:
        self.assertEqual(
            _failure_code(MemoryError("Generative worker exceeded its memory hard ceiling")),
            "memory_hard_ceiling_exceeded",
        )
        self.assertEqual(_failure_code(MemoryError("allocator failed")), "memory_error")
        self.assertEqual(
            _failure_code(MLXGenMemoryProfileError("blockwise_materialization_failed")),
            "blockwise_materialization_failed",
        )


if __name__ == "__main__":
    unittest.main()
