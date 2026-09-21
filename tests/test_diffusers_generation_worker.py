import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.diffusers_generation_worker import (
    GeneratedImageArtifact,
    LocalDiffusersImageRuntime,
    WorkerTelemetry,
    default_worker_telemetry,
    execute_diffusers_image_request,
)


class FakeRuntime:
    pipeline_class = "Flux2KleinPipeline"

    def __init__(self, output: Path) -> None:
        self.output = output

    def generate(self, request, progress):
        progress()
        self.output.write_bytes(b"private generated image bytes")
        self.output.chmod(0o600)
        return GeneratedImageArtifact(self.output, request["width"], request["height"])


class DiffusersGenerationWorkerTests(unittest.TestCase):
    def test_default_telemetry_includes_mlx_allocator_peak(self) -> None:
        memory = SimpleNamespace(pressure=SimpleNamespace(value="normal"))
        with patch.dict(
            sys.modules,
            {"mlx.core": SimpleNamespace(get_peak_memory=lambda: 9000)},
        ), patch(
            "vllm_apple.diffusers_generation_worker.resource.getrusage",
            return_value=SimpleNamespace(ru_maxrss=1000),
        ), patch(
            "vllm_apple.diffusers_generation_worker.detect_memory", return_value=memory
        ), patch(
            "vllm_apple.diffusers_generation_worker.platform.system", return_value="Darwin"
        ), patch(
            "vllm_apple.diffusers_generation_worker._darwin_thermal_state",
            return_value="nominal",
        ):
            snapshot = default_worker_telemetry()
        self.assertEqual(snapshot.process_rss_bytes, 9000)

    def test_default_telemetry_includes_mps_driver_allocation(self) -> None:
        memory = SimpleNamespace(pressure=SimpleNamespace(value="normal"))
        torch = SimpleNamespace(mps=SimpleNamespace(
            current_allocated_memory=lambda: 7000,
            driver_allocated_memory=lambda: 12000,
        ))
        with patch.dict(sys.modules, {"torch": torch}), patch(
            "vllm_apple.diffusers_generation_worker.resource.getrusage",
            return_value=SimpleNamespace(ru_maxrss=1000),
        ), patch(
            "vllm_apple.diffusers_generation_worker.detect_memory", return_value=memory
        ), patch(
            "vllm_apple.diffusers_generation_worker.platform.system", return_value="Darwin"
        ), patch(
            "vllm_apple.diffusers_generation_worker._darwin_thermal_state",
            return_value="nominal",
        ):
            snapshot = default_worker_telemetry()
        self.assertEqual(snapshot.process_rss_bytes, 12000)

    def test_image_worker_emits_telemetry_and_removes_private_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "sample.png"
            events = []
            ticks = iter((10.0, 10.0, 10.1, 10.2, 10.3, 10.4))
            execute_diffusers_image_request(
                {
                    "candidate_id": "flux2-klein-9b-base",
                    "modality": "image",
                    "mode": "text-to-image",
                    "output_root": str(root),
                    "width": 512,
                    "height": 512,
                },
                FakeRuntime(output),
                telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                emit=events.append,
                clock=lambda: next(ticks),
            )
            self.assertFalse(output.exists())
        self.assertEqual(
            [event.kind for event in events],
            ["started", "progress", "first_output", "completed"],
        )
        self.assertEqual(events[-1].output_width, 512)
        self.assertEqual(len(events[-1].output_sha256), 64)

    def test_candidate_pipeline_mismatch_is_rejected_before_generation(self) -> None:
        with TemporaryDirectory() as directory:
            runtime = FakeRuntime(Path(directory) / "sample.png")
            runtime.pipeline_class = "QwenImagePipeline"
            with self.assertRaisesRegex(ValueError, "does not match"):
                execute_diffusers_image_request(
                    {
                        "candidate_id": "flux2-dev",
                        "modality": "image",
                        "mode": "text-to-image",
                        "output_root": directory,
                    },
                    runtime,
                    telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                    emit=lambda event: None,
                )

    def test_output_outside_root_is_rejected_without_deleting_out_of_scope_file(self) -> None:
        with TemporaryDirectory() as directory, TemporaryDirectory() as outside:
            output = Path(outside) / "sample.png"
            with self.assertRaisesRegex(ValueError, "inside the output root"):
                execute_diffusers_image_request(
                    {
                        "candidate_id": "flux2-klein-9b-base",
                        "modality": "image",
                        "mode": "text-to-image",
                        "output_root": directory,
                        "width": 512,
                        "height": 512,
                    },
                    FakeRuntime(output),
                    telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                    emit=lambda event: None,
                )
            self.assertTrue(output.exists())

    def test_local_runtime_lazily_loads_local_only_mps_pipeline(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            output = root / "output"
            model.mkdir()
            output.mkdir()
            calls = {}

            class Image:
                def save(self, path, *, format):
                    calls["format"] = format
                    Path(path).write_bytes(b"png")

            class Pipeline:
                vae = SimpleNamespace(enable_tiling=lambda: calls.setdefault("tiling", True))

                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    calls["load"] = (path, kwargs)
                    return cls()

                def to(self, device):
                    calls["device"] = device

                def __call__(self, **kwargs):
                    calls["generate"] = kwargs
                    kwargs["callback_on_step_end"](self, 0, 1, {})
                    return SimpleNamespace(images=[Image()])

            class Generator:
                def __init__(self, *, device):
                    calls["generator_device"] = device

                def manual_seed(self, seed):
                    calls["seed"] = seed
                    return self

            torch = SimpleNamespace(
                bfloat16="bf16",
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                mps=SimpleNamespace(empty_cache=lambda: calls.setdefault("empty_cache", True)),
                Generator=Generator,
            )
            diffusers = SimpleNamespace(Flux2KleinPipeline=Pipeline)
            modules = {"torch": torch, "diffusers": diffusers}
            runtime = LocalDiffusersImageRuntime(
                "flux2-klein-9b-base", module_loader=modules.__getitem__
            )
            artifact = runtime.generate(
                {
                    "candidate_id": "flux2-klein-9b-base",
                    "model_root": str(model),
                    "output_root": str(output),
                    "prompt": "test",
                    "seed": 9,
                    "width": 512,
                    "height": 512,
                    "steps": 2,
                    "batch_size": 1,
                    "sample_index": 0,
                },
                lambda: calls.setdefault("progress", True),
            )
            self.addCleanup(lambda: artifact.path.unlink(missing_ok=True))
            self.assertTrue(artifact.path.is_file())
            self.assertTrue(calls["load"][1]["local_files_only"])
            self.assertEqual(calls["load"][1]["dtype"], "bf16")
            self.assertEqual(calls["device"], "mps")
            self.assertEqual(calls["generator_device"], "cpu")
            self.assertEqual(calls["seed"], 9)
            self.assertTrue(calls["tiling"])
            self.assertTrue(calls["empty_cache"])

    def test_local_runtime_rejects_unavailable_mps_before_model_load(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            output = root / "output"
            model.mkdir()
            output.mkdir()
            torch = SimpleNamespace(
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
            )
            modules = {"torch": torch, "diffusers": SimpleNamespace()}
            runtime = LocalDiffusersImageRuntime(
                "qwen-image-2512", module_loader=modules.__getitem__
            )
            with self.assertRaisesRegex(RuntimeError, "MPS"):
                runtime.generate(
                    {
                        "candidate_id": "qwen-image-2512",
                        "model_root": str(model),
                        "output_root": str(output),
                        "batch_size": 1,
                    },
                    lambda: None,
                )

    def test_local_runtime_rejects_batch_before_import_or_model_load(self) -> None:
        runtime = LocalDiffusersImageRuntime(
            "qwen-image-2.1",
            module_loader=lambda _name: self.fail("batch gate must precede imports"),
        )
        with self.assertRaisesRegex(ValueError, "batch size one"):
            runtime.generate(
                {"candidate_id": "qwen-image-2.1", "batch_size": 2}, lambda: None)

    def test_qwen_image_21_requires_sequential_mps_offload(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, output = root / "model", root / "output"
            model.mkdir()
            output.mkdir()
            calls = {}

            class Image:
                def save(self, path, format):
                    Path(path).write_bytes(b"png")

            class Pipeline:
                model_cpu_offload_seq = "text_encoder->transformer->vae"
                vae = SimpleNamespace(enable_tiling=lambda: calls.setdefault("tiling", True))
                text_encoder = object()
                tokenizer = object()
                _execution_device = "mps"

                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    calls["load"] = (path, kwargs)
                    return cls()

                def enable_model_cpu_offload(self, *, device):
                    calls.setdefault("offload", []).append(device)

                def encode_prompt(self, **kwargs):
                    calls["encode_prompt"] = kwargs
                    return "embeds", "mask", "image-mask"

                def remove_all_hooks(self):
                    calls["remove_hooks"] = True

                def to(self, device):
                    raise AssertionError("Qwen-Image-2.1 must not use full-pipeline MPS residency")

                def __call__(self, **kwargs):
                    calls["generate"] = kwargs
                    kwargs["callback_on_step_end"](self, 0, 0, {})
                    return SimpleNamespace(images=[Image()])

            torch = SimpleNamespace(
                bfloat16="bf16",
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: object()),
                mps=SimpleNamespace(empty_cache=lambda: None),
            )
            runtime = LocalDiffusersImageRuntime(
                "qwen-image-2.1",
                module_loader={
                    "torch": torch,
                    "diffusers": SimpleNamespace(QwenImage21Pipeline=Pipeline),
                }.__getitem__,
            )
            artifact = runtime.generate(
                {
                    "candidate_id": "qwen-image-2.1",
                    "model_root": str(model), "output_root": str(output),
                    "prompt": "test", "seed": 1, "width": 64, "height": 64,
                    "steps": 1, "batch_size": 1, "sample_index": 0,
                },
                lambda: None,
            )
        self.assertEqual(calls["offload"], ["mps", "mps"])
        self.assertEqual(calls["generate"]["prompt_embeds"], "embeds")
        self.assertEqual(calls["generate"]["prompt_embeds_mask"], "mask")
        self.assertIsNone(calls["generate"]["prompt"])
        self.assertTrue(calls["remove_hooks"])
        self.assertTrue(calls["load"][1]["local_files_only"])
        self.assertEqual(artifact.width, 64)


if __name__ == "__main__":
    unittest.main()
