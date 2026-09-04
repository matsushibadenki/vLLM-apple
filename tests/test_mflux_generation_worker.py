import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.diffusers_generation_worker import WorkerTelemetry
from vllm_apple.mflux_generation_worker import (
    LocalMFluxImageRuntime,
    execute_mflux_image_request,
)


class MFluxGenerationWorkerTests(unittest.TestCase):
    def test_local_z_image_runtime_uses_only_supplied_model_path(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "model"
            output_root = root / "output"
            model_root.mkdir()
            output_root.mkdir()
            calls = {}

            class ModelConfig:
                @staticmethod
                def z_image_turbo():
                    return "z-config"

            class Image:
                def save(self, path):
                    Path(path).write_bytes(b"png")

            class ZImageTurbo:
                def __init__(self, **kwargs):
                    calls["load"] = kwargs

                def generate_image(self, **kwargs):
                    calls["generate"] = kwargs
                    return Image()

            modules = {
                "mflux.models.common.config": SimpleNamespace(ModelConfig=ModelConfig),
                "mflux.models.z_image": SimpleNamespace(ZImageTurbo=ZImageTurbo),
            }
            runtime = LocalMFluxImageRuntime(
                "z-image-turbo-mlx-4bit", module_loader=modules.__getitem__
            )
            request = {
                "candidate_id": "z-image-turbo-mlx-4bit",
                "modality": "image",
                "mode": "text-to-image",
                "model_root": str(model_root),
                "output_root": str(output_root),
                "prompt": "test",
                "seed": 7,
                "width": 512,
                "height": 512,
                "steps": 9,
                "batch_size": 1,
                "sample_index": 0,
                "memory_hard_ceiling_bytes": 1024 * 1024,
            }
            events = []
            with patch("vllm_apple.mflux_generation_worker.platform.system", return_value="Darwin"), patch(
                "vllm_apple.mflux_generation_worker.platform.machine", return_value="arm64"
            ):
                execute_mflux_image_request(
                    request,
                    runtime,
                    telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                    emit=events.append,
                )
        self.assertEqual(calls["load"]["model_path"], str(model_root.resolve()))
        self.assertEqual(calls["load"]["model_config"], "z-config")
        self.assertEqual(calls["generate"]["num_inference_steps"], 9)
        self.assertEqual(
            [event.kind for event in events],
            ["started", "progress", "first_output", "completed"],
        )
        self.assertFalse(list(output_root.glob("*.png")))


if __name__ == "__main__":
    unittest.main()
