import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.diffusers_generation_worker import WorkerTelemetry
from vllm_apple.mlx_gen_generation_worker import (
    LocalMLXGenImageRuntime,
    _BoundedProgressSink,
    execute_mlx_gen_image_request,
)


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

    def test_progress_sink_rejects_non_json_output(self) -> None:
        sink = _BoundedProgressSink(lambda: None)
        with self.assertRaisesRegex(RuntimeError, "non-JSON"):
            sink.write("unsafe diagnostic\n")


if __name__ == "__main__":
    unittest.main()
