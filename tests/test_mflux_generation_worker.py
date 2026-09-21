import hashlib
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

    def test_qwen_image_edit_uses_a_verified_private_source_copy(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_root, output_root = root / "model", root / "output"
            model_root.mkdir()
            output_root.mkdir()
            source_path = root / "source.png"
            source_path.write_bytes(b"\x89PNG\r\n\x1a\nprivate-source")
            source_path.chmod(0o600)
            source_data = source_path.read_bytes()
            calls = {}

            class ModelConfig:
                @staticmethod
                def qwen_image():
                    return "qwen-config"

            class Source:
                format = "PNG"
                size = (16, 16)

                def load(self):
                    pass

                def convert(self, _mode):
                    return self

                def save(self, path, *, format):
                    calls["source_format"] = format
                    Path(path).write_bytes(b"private-copy")

                def close(self):
                    pass

            class Output:
                def save(self, path):
                    Path(path).write_bytes(b"png")

            class QwenImage:
                def __init__(self, **_kwargs):
                    pass

                def generate_image(self, **kwargs):
                    calls["source_path"] = kwargs["image_path"]
                    calls["source_exists_during_generation"] = kwargs["image_path"].is_file()
                    return Output()

            modules = {
                "mflux.models.common.config": SimpleNamespace(ModelConfig=ModelConfig),
                "mflux.models.qwen.variants.txt2img.qwen_image": SimpleNamespace(
                    QwenImage=QwenImage
                ),
                "PIL.Image": SimpleNamespace(open=lambda _stream: Source()),
            }
            runtime = LocalMFluxImageRuntime(
                "qwen-image-2512", module_loader=modules.__getitem__
            )
            request = {
                "candidate_id": "qwen-image-2512",
                "modality": "image",
                "mode": "image-edit",
                "model_root": str(model_root),
                "output_root": str(output_root),
                "input_image_path": str(source_path),
                "input_image_sha256": hashlib.sha256(source_data).hexdigest(),
                "input_image_bytes": len(source_data),
                "prompt": "edit",
                "seed": 7,
                "width": 512,
                "height": 512,
                "steps": 20,
                "batch_size": 1,
                "sample_index": 0,
                "memory_hard_ceiling_bytes": 1024 * 1024,
            }
            with patch(
                "vllm_apple.mflux_generation_worker.platform.system", return_value="Darwin"
            ), patch(
                "vllm_apple.mflux_generation_worker.platform.machine", return_value="arm64"
            ):
                execute_mflux_image_request(
                    request,
                    runtime,
                    telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                    emit=lambda _event: None,
                )
            self.assertTrue(calls["source_exists_during_generation"])
            self.assertEqual(calls["source_format"], "PNG")
            self.assertFalse(calls["source_path"].exists())


if __name__ == "__main__":
    unittest.main()
