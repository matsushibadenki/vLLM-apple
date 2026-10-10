import hashlib
import importlib.util
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

from vllm_apple.expert_execution import ResidentExpertExecutor
from vllm_apple.expert_residency import ExpertKey, ExpertResidencyManager
from vllm_apple.expert_timing import ExpertTimings
from vllm_apple.mlx_expert_backend import MLXFileExpertBackend
from vllm_apple.mlx_expert_export import export_switch_experts
from vllm_apple.qwen3_moe_residency import install_qwen3_moe_residency


class FileExpertAdmissionTests(unittest.TestCase):
    def test_unsupported_quantization_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for bits, group in ((2, 64), (True, 64), (4, 16), (4, True)):
                with self.assertRaises(ValueError):
                    MLXFileExpertBackend(Path(directory), maximum_file_bytes=4096,
                                         quantization_bits=bits, group_size=group)

    def test_missing_and_oversized_file_rejected_before_mlx_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = MLXFileExpertBackend(root, maximum_file_bytes=4)
            with self.assertRaises(ValueError):
                backend.load_expert(ExpertKey(0, 0))
            path = root / "layer-0-expert-0.safetensors"
            path.write_bytes(b"12345")
            with self.assertRaisesRegex(ValueError, "limit"):
                backend.load_expert(ExpertKey(0, 0))
            path.unlink()
            path.symlink_to(root / "other")
            with self.assertRaises(ValueError):
                backend.load_expert(ExpertKey(0, 0))


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires native MLX runtime")
class NativeExpertTests(unittest.TestCase):
    def test_grouped_eval_and_capacity_fallback(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

        mx.set_default_device(mx.cpu)
        for capacity, bits, budget, evaluations in (
            (1, None, 100000, 4), (2, None, 100000, 2),
            (2, None, 60000, 4), (2, 4, 100000, 2), (2, 8, 100000, 2),
        ):
            with self.subTest(capacity=capacity, bits=bits), tempfile.TemporaryDirectory() as directory:
                mx.random.seed(11)
                block = Qwen3MoeSparseMoeBlock(SimpleNamespace(
                    hidden_size=64, moe_intermediate_size=64, num_experts=3,
                    num_experts_per_tok=2, norm_topk_prob=True,
                ))
                if bits is not None:
                    nn.quantize(block, group_size=32, bits=bits)
                x = mx.cos(mx.arange(128).reshape(1, 2, 64))
                expected = block(x)
                mx.eval(expected)
                root = Path(directory) / "experts"
                export_switch_experts({0: block.switch_mlp}, root, model_sha256="a" * 64,
                                      maximum_expert_bytes=65536, maximum_total_bytes=262144)
                timings = ExpertTimings()
                backend = MLXFileExpertBackend.from_manifest(
                    root, expected_model_sha256="a" * 64,
                    maximum_file_bytes=65536, timings=timings,
                )
                manager = ExpertResidencyManager(backend, maximum_entries=capacity, maximum_bytes=budget)
                executor = ResidentExpertExecutor(manager, timings=timings)
                install_qwen3_moe_residency(block, backend, executor, layer=0,
                                           phase="prefill", grouped=True)
                try:
                    actual = block(x)
                    self.assertLess(float(mx.max(mx.abs(actual - expected)).item()), 1e-5)
                    self.assertEqual(timings.snapshot()["expert_eval"]["count"], evaluations)
                    self.assertEqual(manager.snapshot()["active_leases"], 0)
                    with self.assertRaisesRegex(ValueError, "dimension"):
                        backend.execute_unaggregated(
                            executor, mx.ones((1, 3)), phase="decode", layer=0,
                            selected_experts=(0,),
                        )
                    self.assertEqual(manager.snapshot()["active_leases"], 0)
                finally:
                    manager.close()

    def test_qwen3_router_hook_preserves_block_output_without_source_bank(self):
        import gc

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

        mx.set_default_device(mx.cpu)
        for bits in (None, 4, 8):
            for normalized in (False, True):
                with self.subTest(bits=bits, normalized=normalized), tempfile.TemporaryDirectory() as directory:
                    mx.random.seed(11)
                    block = Qwen3MoeSparseMoeBlock(SimpleNamespace(
                        hidden_size=64, moe_intermediate_size=64, num_experts=3,
                        num_experts_per_tok=2, norm_topk_prob=normalized,
                    ))
                    if bits is not None:
                        nn.quantize(block, group_size=32, bits=bits)
                    x = mx.cos(mx.arange(128).reshape(1, 2, 64))
                    expected = block(x)
                    mx.eval(expected)
                    reference = weakref.ref(block.switch_mlp)
                    root = Path(directory) / "experts"
                    export_switch_experts(
                        {0: block.switch_mlp}, root, model_sha256="a" * 64,
                        maximum_expert_bytes=65536, maximum_total_bytes=262144,
                    )
                    backend = MLXFileExpertBackend.from_manifest(
                        root, expected_model_sha256="a" * 64, maximum_file_bytes=65536,
                    )
                    manager = ExpertResidencyManager(backend, maximum_entries=1, maximum_bytes=50000)
                    executor = ResidentExpertExecutor(manager)
                    gate = block.gate
                    hook = install_qwen3_moe_residency(
                        block, backend, executor, layer=0, phase="prefill",
                    )
                    gc.collect()
                    self.assertIsNone(reference())
                    self.assertIs(block.gate, gate)
                    try:
                        actual = block(x)
                        self.assertLess(float(mx.max(mx.abs(actual - expected)).item()), 1e-5)
                        self.assertEqual(manager.snapshot()["active_leases"], 0)
                        self.assertEqual(manager.snapshot()["entries"], 1)
                        hook.set_phase("decode")
                        result = block(x[:, :1])
                        mx.eval(result)
                        self.assertGreater(executor.telemetry["decode"].snapshot()["samples"], 0)
                        with self.assertRaises(ValueError):
                            hook(x, mx.array([[[0, 0], [1, 2]]]))
                    finally:
                        manager.close()

    def test_exported_switchglu_matches_original_router_output(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.switch_layers import SwitchGLU

        mx.set_default_device(mx.cpu)
        for bits in (None, 4, 8):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as directory:
                mx.random.seed(7)
                switch = SwitchGLU(64, 64, 3)
                if bits is not None:
                    nn.quantize(switch, group_size=32, bits=bits)
                root = Path(directory) / "experts"
                export_switch_experts(
                    {0: switch}, root, model_sha256="a" * 64,
                    maximum_expert_bytes=65536, maximum_total_bytes=262144,
                )
                backend = MLXFileExpertBackend.from_manifest(
                    root, expected_model_sha256="a" * 64, maximum_file_bytes=65536,
                )
                manager = ExpertResidencyManager(backend, maximum_entries=2, maximum_bytes=100000)
                executor = ResidentExpertExecutor(manager)
                x = mx.cos(mx.arange(128).reshape(2, 64))
                selections = ((2, 0), (1, 2))
                scores = ((0.7, 0.3), (0.2, 0.8))
                expected = (switch(x, mx.array(selections)) * mx.array(scores)[..., None]).sum(-2)
                mx.eval(expected)
                try:
                    actual = backend.execute_routed(
                        executor, x, phase="prefill", layer=0,
                        selections=selections, weights=scores,
                    )
                    self.assertLess(float(mx.max(mx.abs(actual - expected)).item()), 1e-5)
                finally:
                    manager.close()
                with self.assertRaises(FileExistsError):
                    export_switch_experts(
                        {0: switch}, root, model_sha256="a" * 64,
                        maximum_expert_bytes=65536, maximum_total_bytes=262144,
                    )
                partial = Path(directory) / "partial"
                with self.assertRaisesRegex(ValueError, "limit"):
                    export_switch_experts(
                        {0: switch}, partial, model_sha256="a" * 64,
                        maximum_expert_bytes=65536, maximum_total_bytes=1,
                    )
                self.assertFalse((partial / "expert-manifest.json").exists())

    def test_affine_quantized_execution_matches_dequantized_reference(self):
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
        for bits in (4, 8):
            for group in (32, 64):
                with self.subTest(bits=bits, group=group), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    tensors = {}
                    reference = {}
                    for index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
                        dense = mx.sin(mx.arange(64 * 64).reshape(64, 64) + index) * 0.01
                        packed, scales, biases = mx.quantize(dense, group_size=group, bits=bits)
                        for part, array in zip(("weight", "scales", "biases"),
                                               (packed, scales, biases)):
                            tensors[f"{projection}.{part}"] = array
                        reference[projection] = mx.dequantize(
                            packed, scales, biases, group_size=group, bits=bits,
                        )
                    mx.save_safetensors(str(root / "layer-0-expert-0.safetensors"), tensors)
                    size = sum(t.nbytes for t in tensors.values())
                    artifact = root / "layer-0-expert-0.safetensors"
                    (root / "expert-manifest.json").write_text(json.dumps(dict(
                        schema_version=1, model_sha256="a" * 64,
                        quantization_bits=bits, group_size=group,
                        experts=[dict(layer=0, expert=0, size_bytes=artifact.stat().st_size,
                                      sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())],
                    )))
                    backend = MLXFileExpertBackend.from_manifest(
                        root, maximum_file_bytes=65536, expected_model_sha256="a" * 64,
                    )
                    manager = ExpertResidencyManager(backend, maximum_entries=1, maximum_bytes=size)
                    executor = ResidentExpertExecutor(manager)
                    x = mx.cos(mx.arange(128).reshape(2, 64))
                    try:
                        actual = backend.execute_selected(
                            executor, x, phase="prefill", layer=0,
                            selected_experts=(0,), routing_weights=(0.75,),
                        )
                        g = x @ reference["gate_proj"].T
                        expected = ((g * mx.sigmoid(g)) * (x @ reference["up_proj"].T))
                        expected = expected @ reference["down_proj"].T * 0.75
                        self.assertLess(float(mx.max(mx.abs(actual - expected)).item()), 1e-6)
                        self.assertEqual(manager.snapshot()["resident_bytes"], size)
                        self.assertEqual(manager.snapshot()["active_leases"], 0)
                    finally:
                        manager.close()
                    # Mismatched grouping is rejected before any expert is retained.
                    wrong = MLXFileExpertBackend(
                        root, maximum_file_bytes=65536, quantization_bits=bits,
                        group_size=64 if group == 32 else 32,
                    )
                    with self.assertRaisesRegex(ValueError, "incompatible"):
                        wrong.load_expert(ExpertKey(0, 0))

    def test_multiple_experts_and_per_row_routing(self):
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            banks = []
            for expert in range(3):
                tensors = {name: mx.array([[0.1 + expert, 0.2], [0.3, 0.4]])
                           for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")}
                mx.save_safetensors(str(root / f"layer-0-expert-{expert}.safetensors"), tensors)
                banks.append(tensors)
            backend = MLXFileExpertBackend(root, maximum_file_bytes=4096)
            manager = ExpertResidencyManager(backend, maximum_entries=2, maximum_bytes=96)
            executor = ResidentExpertExecutor(manager)
            x = mx.array([[0.5, 0.6], [0.7, 0.8]])
            selections = ((1, 0), (2, 1))
            weights = ((0.7, 0.2), (0.1, 0.4))
            try:
                actual = backend.execute_routed(
                    executor, x, phase="prefill", layer=0,
                    selections=selections, weights=weights,
                )
                expected = []
                for row in range(2):
                    contributions = []
                    for expert, weight in zip(selections[row], weights[row]):
                        w = banks[expert]
                        token = x[row:row + 1]
                        g = token @ w["gate_proj.weight"].T
                        contributions.append(
                            ((g * mx.sigmoid(g)) * (token @ w["up_proj.weight"].T))
                            @ w["down_proj.weight"].T * weight
                        )
                    expected.append(contributions[0] + contributions[1])
                error = float(mx.max(mx.abs(actual - mx.concatenate(expected))).item())
                self.assertLess(error, 1e-6)
                self.assertEqual(manager.snapshot()["active_leases"], 0)
                self.assertEqual(manager.snapshot()["resident_bytes"], 96)
                self.assertEqual(executor.telemetry["prefill"].snapshot()["selected_experts"], 4)
                misses = manager.snapshot()["misses"]
                with self.assertRaises(ValueError):
                    backend.execute_routed(
                        executor, x, phase="prefill", layer=0,
                        selections=((0,), (1, 1)), weights=((1.0,), (0.5, 0.5)),
                    )
                self.assertEqual(manager.snapshot()["misses"], misses)
                with self.assertRaisesRegex(ValueError, "dimension"):
                    backend.execute_selected(
                        executor, mx.ones((1, 3)), phase="decode", layer=0,
                        selected_experts=(1,), routing_weights=(1.0,),
                    )
                self.assertEqual(manager.snapshot()["active_leases"], 0)
            finally:
                manager.close()

    def test_wrong_tensor_contract_rejected(self):
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mx.save_safetensors(str(root / "layer-0-expert-0.safetensors"),
                                {"weight": mx.array([[1.0]])})
            backend = MLXFileExpertBackend(root, maximum_file_bytes=4096)
            with self.assertRaisesRegex(ValueError, "three dense"):
                backend.load_expert(ExpertKey(0, 0))

    def test_file_execution_matches_dense_reference_after_eviction(self):
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            banks = []
            for expert in range(2):
                tensors = {
                    name: mx.array([[0.1 + expert, 0.2], [0.3, 0.4]])
                    for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
                }
                mx.save_safetensors(str(root / f"layer-0-expert-{expert}.safetensors"), tensors)
                banks.append(tensors)
            backend = MLXFileExpertBackend(root, maximum_file_bytes=4096)
            manager = ExpertResidencyManager(
                backend, maximum_entries=1, maximum_bytes=48,
                eviction_policy="cost_frequency",
            )
            executor = ResidentExpertExecutor(manager)
            x = mx.array([[0.5, 0.6], [0.7, 0.8]])
            try:
                for expert in (0, 1, 0):
                    actual = backend.execute_selected(
                        executor, x, phase="prefill", layer=0,
                        selected_experts=(expert,), routing_weights=(1.0,),
                    )
                    w = banks[expert]
                    g = x @ w["gate_proj.weight"].T
                    expected = ((g * mx.sigmoid(g)) * (x @ w["up_proj.weight"].T)) @ w["down_proj.weight"].T
                    self.assertLess(float(mx.max(mx.abs(actual - expected)).item()), 1e-6)
                    self.assertEqual(manager.snapshot()["active_leases"], 0)
                    self.assertEqual(manager.snapshot()["resident_bytes"], 48)
                self.assertEqual(manager.snapshot()["evictions"], 2)
            finally:
                manager.close()
            self.assertEqual(manager.snapshot()["resident_bytes"], 0)
