import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vllm_apple.qwen_image_21_phase_handoff import (
    consume_qwen_image_21_phase_handoff,
    encode_qwen_image_21_prompt_handoff,
    main,
    save_qwen_image_21_phase_handoff,
)


class Tensor:
    def __init__(self, shape, dtype="torch.bfloat16"):
        self.shape = shape
        self.dtype = dtype

    def detach(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def contiguous(self):
        return self


class QwenImage21PhaseHandoffTests(unittest.TestCase):
    def test_private_handoff_round_trips_and_is_consumed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            tensors = {
                "prompt_embeds": Tensor((1, 86, 4096)),
                "prompt_embeds_mask": Tensor((1, 86), "torch.int64"),
            }
            manifest = save_qwen_image_21_phase_handoff(
                root, tensors, plan_sha256="a" * 64, prompt_sha256="b" * 64,
                sample_index=1, mode="text-to-image",
                save_file=lambda _tensors, path: Path(path).write_bytes(b"safe-tensors"),
            )
            loaded = consume_qwen_image_21_phase_handoff(
                manifest, plan_sha256="a" * 64, prompt_sha256="b" * 64,
                sample_index=1, mode="text-to-image", load_file=lambda _path: tensors,
            )
            self.assertEqual(set(loaded), set(tensors))
            self.assertEqual(list(root.iterdir()), [])

    def test_identity_mismatch_fails_closed_and_cleans_up(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            tensors = {
                "prompt_embeds": Tensor((1, 86, 4096)),
                "prompt_embeds_mask": Tensor((1, 86), "torch.int64"),
            }
            manifest = save_qwen_image_21_phase_handoff(
                root, tensors, plan_sha256="a" * 64, prompt_sha256="b" * 64,
                sample_index=0, mode="text-to-image",
                save_file=lambda _tensors, path: Path(path).write_bytes(b"safe-tensors"),
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                consume_qwen_image_21_phase_handoff(
                    manifest, plan_sha256="c" * 64, prompt_sha256="b" * 64,
                    sample_index=0, mode="text-to-image", load_file=lambda _path: tensors,
                )
            self.assertEqual(list(root.iterdir()), [])

    def test_manifest_promotion_failure_removes_payload(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            tensors = {
                "prompt_embeds": Tensor((1, 86, 4096)),
                "prompt_embeds_mask": Tensor((1, 86), "torch.int64"),
            }
            original = __import__("os").replace
            calls = 0

            def replace(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("manifest promotion failed")
                return original(source, target)

            with patch(
                "vllm_apple.qwen_image_21_phase_handoff.os.replace",
                side_effect=replace,
            ), self.assertRaisesRegex(OSError, "manifest promotion"):
                save_qwen_image_21_phase_handoff(
                    root, tensors, plan_sha256="a" * 64, prompt_sha256="b" * 64,
                    sample_index=0, mode="text-to-image",
                    save_file=lambda _tensors, path: Path(path).write_bytes(b"safe-tensors"),
                )
            self.assertEqual(list(root.iterdir()), [])

    def test_image_edit_requires_mask_and_private_root(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o755)
            tensors = {
                "prompt_embeds": Tensor((1, 86, 4096)),
                "prompt_embeds_mask": Tensor((1, 86), "torch.int64"),
            }
            with self.assertRaisesRegex(ValueError, "private"):
                save_qwen_image_21_phase_handoff(
                    root, tensors, plan_sha256="a" * 64, prompt_sha256="b" * 64,
                    sample_index=0, mode="image-edit",
                    save_file=lambda _tensors, path: Path(path).write_bytes(b"x"),
                )

    def test_encoder_loads_only_text_components_and_releases_hooks(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            handoff = root / "handoff"
            (model / "processor").mkdir(parents=True)
            (model / "text_encoder").mkdir()
            handoff.mkdir(mode=0o700)
            calls = []

            class Component:
                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    calls.append((Path(path).name, kwargs))
                    return cls()

            class Pipeline:
                model_cpu_offload_seq = "text_encoder->transformer->vae"
                _execution_device = "mps"

                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

                def enable_sequential_cpu_offload(self, *, device):
                    calls.append(("offload", device))

                def encode_prompt(self, **kwargs):
                    calls.append(("encode", kwargs["prompt"]))
                    return Tensor((1, 86, 4096)), Tensor((1, 86), "torch.int64"), None

                def remove_all_hooks(self):
                    calls.append(("remove_hooks", True))

            torch = type("Torch", (), {
                "bfloat16": "bf16",
                "backends": type("Backends", (), {
                    "mps": type("MPSBackend", (), {"is_available": staticmethod(lambda: True)})
                })(),
                "mps": type("MPS", (), {"empty_cache": staticmethod(lambda: None)})(),
            })()
            modules = {
                "torch": torch,
                "transformers": type("Transformers", (), {
                    "Qwen3VLProcessor": Component,
                    "Qwen3VLForConditionalGeneration": Component,
                })(),
                "diffusers": type("Diffusers", (), {"QwenImage21Pipeline": Pipeline})(),
            }
            request = {
                "candidate_id": "qwen-image-2.1",
                "model_root": str(model),
                "mode": "text-to-image",
                "prompt": "private prompt",
                "batch_size": 1,
                "plan_sha256": "a" * 64,
                "prompt_sha256": "b" * 64,
                "sample_index": 0,
            }
            # Exercise the model boundary while replacing only the serialization call.
            import vllm_apple.qwen_image_21_phase_handoff as module
            original = module.save_qwen_image_21_phase_handoff
            module.save_qwen_image_21_phase_handoff = lambda *_args, **_kwargs: handoff / "x"
            try:
                result = encode_qwen_image_21_prompt_handoff(
                    request, handoff, module_loader=modules.__getitem__
                )
            finally:
                module.save_qwen_image_21_phase_handoff = original
            self.assertEqual(result, handoff / "x")
            self.assertEqual([call[0] for call in calls[:2]], ["processor", "text_encoder"])
            self.assertIn(("remove_hooks", True), calls)

    def test_encoder_entry_point_rejects_outside_handoff_before_request(self) -> None:
        with TemporaryDirectory() as workspace, TemporaryDirectory() as outside:
            with patch(
                "vllm_apple.generative_worker_protocol.consume_private_generative_request"
            ) as consume:
                code = main([
                    "--request", str(Path(workspace) / "request.json"),
                    "--workspace-root", workspace,
                    "--handoff-root", outside,
                ])
            self.assertEqual(code, 1)
            consume.assert_not_called()


if __name__ == "__main__":
    unittest.main()
