import hashlib
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

    def test_image_edit_is_forwarded_as_a_bound_rgb_image(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, output = root / "model", root / "output"
            model.mkdir()
            output.mkdir()
            source = root / "source.png"
            source.write_bytes(b"\x89PNG\r\n\x1a\nprivate-test-payload")
            source.chmod(0o600)
            calls = {}

            class SourceImage:
                format = "PNG"
                size = (8, 8)

                def load(self):
                    calls["loaded"] = True

                def convert(self, mode):
                    calls["converted"] = mode
                    return "rgb-source"

                def close(self):
                    calls["closed"] = True

            class OutputImage:
                def save(self, path, *, format):
                    Path(path).write_bytes(b"png")

            class Pipeline:
                vae = None

                @classmethod
                def from_pretrained(cls, _path, **_kwargs):
                    return cls()

                def to(self, _device):
                    pass

                def __call__(self, **kwargs):
                    calls["image"] = kwargs.get("image")
                    return SimpleNamespace(images=[OutputImage()])

            class Generator:
                def __init__(self, *, device):
                    pass

                def manual_seed(self, _seed):
                    return self

            data = source.read_bytes()
            modules = {
                "torch": SimpleNamespace(
                    bfloat16="bf16",
                    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                    mps=SimpleNamespace(empty_cache=lambda: None),
                    Generator=Generator,
                ),
                "diffusers": SimpleNamespace(Flux2KleinPipeline=Pipeline),
                "PIL.Image": SimpleNamespace(open=lambda _stream: SourceImage()),
            }
            runtime = LocalDiffusersImageRuntime(
                "flux2-klein-9b-base", module_loader=modules.__getitem__
            )
            artifact = runtime.generate(
                {
                    "candidate_id": "flux2-klein-9b-base",
                    "mode": "image-edit",
                    "model_root": str(model),
                    "output_root": str(output),
                    "input_image_path": str(source),
                    "input_image_sha256": hashlib.sha256(data).hexdigest(),
                    "input_image_bytes": len(data),
                    "prompt": "edit",
                    "seed": 1,
                    "width": 64,
                    "height": 64,
                    "steps": 1,
                    "batch_size": 1,
                    "sample_index": 0,
                },
                lambda: None,
            )
            self.addCleanup(lambda: artifact.path.unlink(missing_ok=True))
            self.assertEqual(calls["image"], "rgb-source")
            self.assertEqual(calls["converted"], "RGB")
            self.assertTrue(calls["loaded"])
            self.assertTrue(calls["closed"])

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
                    calls.setdefault("loads", []).append((path, kwargs))
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
            self.assertTrue(calls["loads"][0][1]["local_files_only"])
            self.assertEqual(calls["loads"][0][1]["dtype"], "bf16")
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

    def test_qwen_image_21_uses_sequential_text_and_group_generation_offload(self) -> None:
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

                def __init__(self, **components):
                    self.__dict__.update(components)
                    self._execution_device = "mps"

                def enable_group_offload(self, **kwargs):
                    calls.setdefault("group_offload", []).append(kwargs)

                def enable_sequential_cpu_offload(self, *, device):
                    calls.setdefault("sequential_offload", []).append(device)

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

            def component(name, value=None):
                class Component:
                    @classmethod
                    def from_pretrained(cls, path, **kwargs):
                        calls.setdefault("loads", []).append((name, path, kwargs))
                        return value if value is not None else cls()
                return Component

            vae = SimpleNamespace(enable_tiling=lambda: calls.setdefault("tiling", True))

            torch = SimpleNamespace(
                bfloat16="bf16",
                device=lambda value: value,
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: object()),
                mps=SimpleNamespace(empty_cache=lambda: None),
            )
            runtime = LocalDiffusersImageRuntime(
                "qwen-image-2.1",
                module_loader={
                    "torch": torch,
                    "transformers": SimpleNamespace(
                        Qwen3VLProcessor=component("processor", object()),
                        Qwen3VLForConditionalGeneration=component("text_encoder"),
                    ),
                    "diffusers": SimpleNamespace(
                        QwenImage21Pipeline=Pipeline,
                        FlowMatchEulerDiscreteScheduler=component("scheduler"),
                        AutoencoderKLQwenImage21=component("vae", vae),
                        QwenImage21Transformer2DModel=component("transformer"),
                    ),
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
        self.assertEqual(calls["sequential_offload"], ["mps"])
        self.assertEqual(calls["group_offload"], [{
            "onload_device": "mps",
            "offload_device": "cpu",
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            "non_blocking": False,
            "use_stream": False,
            "low_cpu_mem_usage": False,
        }])
        self.assertEqual(calls["generate"]["prompt_embeds"], "embeds")
        self.assertEqual(calls["generate"]["prompt_embeds_mask"], "mask")
        self.assertIsNone(calls["generate"]["prompt"])
        self.assertTrue(calls["remove_hooks"])
        self.assertEqual(
            [load[0] for load in calls["loads"]],
            ["processor", "text_encoder", "scheduler", "vae", "transformer"],
        )
        self.assertTrue(all(load[2]["local_files_only"] for load in calls["loads"]))
        self.assertEqual(artifact.width, 64)

    def test_qwen_image_21_binds_precomputed_image_pad_mask(self) -> None:
        calls = {}

        class Pipeline:
            def encode_prompt(self, *args, **kwargs):
                calls["kwargs"] = kwargs
                return "embeds", "attention", kwargs["image_pad_mask"]

        pipeline = Pipeline()
        mask = object()
        LocalDiffusersImageRuntime._bind_image_pad_mask(pipeline, mask)
        result = pipeline.encode_prompt(
            prompt=None,
            image=[object()],
            prompt_embeds="embeds",
            prompt_embeds_mask="attention",
        )
        self.assertIs(calls["kwargs"]["image_pad_mask"], mask)
        self.assertIs(result[2], mask)

    def test_qwen_image_21_group_offload_can_bind_private_disk_path(self) -> None:
        calls = []
        pipeline = SimpleNamespace(
            model_cpu_offload_seq="text_encoder->transformer->vae",
            enable_group_offload=lambda **kwargs: calls.append(kwargs),
        )
        torch = SimpleNamespace(device=lambda value: value)
        with TemporaryDirectory() as directory:
            offload = Path(directory)
            LocalDiffusersImageRuntime._enable_generation_group_offload(
                pipeline, torch, disk_offload_root=offload
            )
        self.assertEqual(calls[0]["offload_to_disk_path"], str(offload))
        self.assertEqual(calls[0]["offload_type"], "block_level")

    def test_qwen_phase_handoff_rejects_edit_before_consuming_or_loading(self) -> None:
        runtime = LocalDiffusersImageRuntime(
            "qwen-image-2.1",
            phase_handoff_manifest="/private/handoff/prompt-embeddings.json",
            module_loader=lambda _name: self.fail("must reject before imports"),
        )
        with self.assertRaisesRegex(ValueError, "condition image transfer"):
            runtime.generate({"candidate_id": "qwen-image-2.1", "batch_size": 1,
                              "mode": "image-edit"}, lambda: None)

    def test_qwen_phase_handoff_rejects_identity_before_model_load(self) -> None:
        runtime = LocalDiffusersImageRuntime(
            "qwen-image-2.1",
            phase_handoff_manifest="/private/handoff/prompt-embeddings.json",
            module_loader=lambda _name: self.fail("must reject before imports"),
        )
        with patch(
            "vllm_apple.qwen_image_21_phase_handoff.consume_qwen_image_21_phase_handoff",
            side_effect=ValueError("handoff identity does not match"),
        ) as consume, self.assertRaisesRegex(ValueError, "identity"):
            runtime.generate({
                "candidate_id": "qwen-image-2.1", "batch_size": 1,
                "mode": "text-to-image", "plan_sha256": "a" * 64,
                "prompt_sha256": "b" * 64, "sample_index": 0,
            }, lambda: None)
        consume.assert_called_once()

    def test_qwen_image_21_condition_dimensions_are_bounded(self) -> None:
        dimensions = LocalDiffusersImageRuntime._condition_dimensions(640, 320)
        self.assertEqual(dimensions, (352, 192))
        self.assertLessEqual(dimensions[0] * dimensions[1], 2 * 256**2)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            LocalDiffusersImageRuntime._condition_dimensions(0, 320)


if __name__ == "__main__":
    unittest.main()
