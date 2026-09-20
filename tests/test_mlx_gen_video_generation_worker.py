import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.diffusers_generation_worker import WorkerTelemetry
from vllm_apple.mlx_gen_video_generation_worker import (
    LocalMLXGenVideoRuntime,
    execute_mlx_gen_video_request,
)


class MLXGenVideoGenerationWorkerTests(unittest.TestCase):
    def test_local_worker_routes_bounded_wan_t2v_and_removes_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            output = root / "output"
            model.mkdir()
            output.mkdir()
            seen = {}

            def backend_main():
                seen["argv"] = list(sys.argv)
                Path(sys.argv[sys.argv.index("--output") + 1]).write_bytes(b"mp4")
                print(json.dumps({"event": "step", "step": 1}))

            request = {
                "candidate_id": "wan2.2-ti2v-5b",
                "modality": "video",
                "mode": "text-to-video",
                "model_root": str(model),
                "output_root": str(output),
                "prompt": "local test",
                "seed": 42,
                "width": 640,
                "height": 360,
                "frames": 33,
                "steps": 20,
                "batch_size": 1,
                "sample_index": 0,
                "memory_hard_ceiling_bytes": 1024 * 1024,
            }
            events = []
            runtime = LocalMLXGenVideoRuntime(
                module_loader=lambda name: SimpleNamespace(main=backend_main)
            )
            with patch(
                "vllm_apple.mlx_gen_video_generation_worker.platform.system",
                return_value="Darwin",
            ), patch(
                "vllm_apple.mlx_gen_video_generation_worker.platform.machine",
                return_value="arm64",
            ):
                execute_mlx_gen_video_request(
                    request,
                    runtime,
                    telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                    emit=events.append,
                )
            argv = seen["argv"]
            self.assertEqual(argv[:2], ["mlxgen", "generate"])
            for option in (
                "--family", "--task", "--frames", "--fps", "--low-ram",
                "--release-inactive-denoiser", "--json-events", "--no-progress",
            ):
                self.assertIn(option, argv)
            self.assertEqual(argv[argv.index("--task") + 1], "text-to-video")
            self.assertEqual(argv[argv.index("--frames") + 1], "33")
            self.assertFalse(list(output.glob("*.mp4")))
        self.assertEqual(
            [event.kind for event in events],
            ["started", "progress", "first_output", "completed"],
        )


if __name__ == "__main__":
    unittest.main()
