import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from vllm_apple.diffusers_generation_worker import WorkerTelemetry
from vllm_apple.diffusers_video_generation_worker import (
    GeneratedVideoArtifact,
    LocalDiffusersVideoRuntime,
    execute_diffusers_video_request,
)


class FakeRuntime:
    pipeline_class = "WanPipeline"

    def __init__(self, output: Path) -> None:
        self.output = output

    def generate(self, request, progress):
        progress()
        self.output.write_bytes(b"private generated video bytes")
        self.output.chmod(0o600)
        return GeneratedVideoArtifact(
            self.output, request["width"], request["height"], request["frames"]
        )


def request(root: Path) -> dict[str, object]:
    return {
        "candidate_id": "wan2.2-ti2v-5b",
        "modality": "video",
        "mode": "text-to-video",
        "model_root": str(root / "model"),
        "output_root": str(root / "output"),
        "prompt": "test",
        "seed": 9,
        "width": 640,
        "height": 360,
        "frames": 33,
        "steps": 20,
        "batch_size": 1,
        "sample_index": 0,
        "memory_hard_ceiling_bytes": 1024**3,
    }


class DiffusersVideoGenerationWorkerTests(unittest.TestCase):
    def test_worker_emits_shape_digest_and_removes_private_video(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "output"
            output_root.mkdir()
            payload = request(root)
            events = []
            ticks = iter((10.0, 10.0, 10.1, 10.2, 10.3, 10.4))
            output = output_root / "sample.mp4"
            execute_diffusers_video_request(
                payload,
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
        self.assertEqual(events[-1].output_frames, 33)
        self.assertEqual(len(events[-1].output_sha256), 64)

    def test_worker_rejects_i2v_and_batch_before_generation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output").mkdir()
            payload = request(root)
            payload["mode"] = "image-to-video"
            with self.assertRaisesRegex(ValueError, "text-to-video only"):
                execute_diffusers_video_request(
                    payload,
                    FakeRuntime(root / "output" / "sample.mp4"),
                    telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                    emit=lambda event: None,
                )
            payload["mode"] = "text-to-video"
            payload["batch_size"] = 2
            with self.assertRaisesRegex(ValueError, "batch size one"):
                execute_diffusers_video_request(
                    payload,
                    FakeRuntime(root / "output" / "sample.mp4"),
                    telemetry=lambda: WorkerTelemetry(1024, "normal", "nominal"),
                    emit=lambda event: None,
                )

    def test_local_runtime_is_local_only_mps_tiled_and_bounded(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "output").mkdir()
            calls = {}

            class Pipeline:
                model_cpu_offload_seq = "text_encoder->transformer->vae"
                vae = SimpleNamespace(enable_tiling=lambda: calls.setdefault("tiling", True))

                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    calls["load"] = (path, kwargs)
                    return cls()

                def enable_model_cpu_offload(self, *, device):
                    calls["offload_device"] = device

                def __call__(self, **kwargs):
                    calls["generate"] = kwargs
                    kwargs["callback_on_step_end"](self, 0, 1, {})
                    return SimpleNamespace(frames=[[object()] * 33])

            class Generator:
                def __init__(self, *, device):
                    calls["generator_device"] = device

                def manual_seed(self, seed):
                    calls["seed"] = seed
                    return self

            def export_to_video(frames, path, *, fps):
                calls["export"] = (len(frames), fps)
                Path(path).write_bytes(b"mp4")

            modules = {
                "torch": SimpleNamespace(
                    bfloat16="bf16",
                    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                    mps=SimpleNamespace(empty_cache=lambda: calls.setdefault("empty_cache", True)),
                    Generator=Generator,
                ),
                "diffusers": SimpleNamespace(WanPipeline=Pipeline),
                "diffusers.utils": SimpleNamespace(export_to_video=export_to_video),
            }
            runtime = LocalDiffusersVideoRuntime(
                "wan2.2-ti2v-5b", module_loader=modules.__getitem__
            )
            artifact = runtime.generate(request(root), lambda: calls.setdefault("progress", True))
            self.addCleanup(lambda: artifact.path.unlink(missing_ok=True))
        self.assertTrue(calls["load"][1]["local_files_only"])
        self.assertEqual(calls["load"][1]["dtype"], "bf16")
        self.assertEqual(calls["offload_device"], "mps")
        self.assertTrue(calls["tiling"])
        self.assertEqual(calls["export"], (33, 16))
        self.assertTrue(calls["empty_cache"])

    def test_local_runtime_rejects_missing_module_residency_contract(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "output").mkdir()

            class Pipeline:
                model_cpu_offload_seq = "transformer->vae"

                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    return cls()

            modules = {
                "torch": SimpleNamespace(
                    bfloat16="bf16",
                    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                    mps=SimpleNamespace(empty_cache=lambda: None),
                ),
                "diffusers": SimpleNamespace(WanPipeline=Pipeline),
                "diffusers.utils": SimpleNamespace(export_to_video=lambda *args, **kwargs: None),
            }
            runtime = LocalDiffusersVideoRuntime(
                "wan2.2-ti2v-5b", module_loader=modules.__getitem__
            )
            with self.assertRaisesRegex(RuntimeError, "module residency sequence"):
                runtime.generate(request(root), lambda: None)


if __name__ == "__main__":
    unittest.main()
