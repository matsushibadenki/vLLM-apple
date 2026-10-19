"""Native multi-layer ownership, rollback and lifecycle integration."""
import gc
import importlib.util
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

from vllm_apple.strata_runtime import StrataRuntime


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires native MLX runtime")
class StrataRuntimeTests(unittest.TestCase):
    def test_multilayer_shared_budget_and_shutdown(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

        from vllm_apple.mlx_expert_backend import MLXFileExpertBackend
        from vllm_apple.mlx_expert_export import export_switch_experts
        mx.set_default_device(mx.cpu)
        for mode in ('sequential', 'grouped', 'row_batch', 'selection_batch'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                blocks = {i: Qwen3MoeSparseMoeBlock(SimpleNamespace(
                    hidden_size=64, moe_intermediate_size=64, num_experts=3,
                    num_experts_per_tok=2, norm_topk_prob=True)) for i in range(2)}
                for block in blocks.values():
                    nn.quantize(block, bits=4, group_size=32)
                x = mx.ones((1, 4, 64))
                expected = blocks[1](blocks[0](x))
                mx.eval(expected)
                refs = [weakref.ref(b.switch_mlp) for b in blocks.values()]
                root = Path(directory) / 'experts'
                export_switch_experts({i: b.switch_mlp for i, b in blocks.items()}, root,
                                      model_sha256='a'*64, maximum_expert_bytes=65536,
                                      maximum_total_bytes=524288)
                backend = MLXFileExpertBackend.from_manifest(
                    root, expected_model_sha256='a'*64, maximum_file_bytes=65536)
                runtime = StrataRuntime(blocks, backend, maximum_entries=2,
                                        maximum_bytes=18432, mode=mode)
                if mode == 'selection_batch':
                    from vllm_apple.expert_timing import ExpertTimings
                    backend.timings = ExpertTimings()
                    indices = mx.array([[[0, 1], [1, 0], [0, 1], [1, 0]]])
                    grouped_result = blocks[0].switch_mlp(x, indices)
                    mx.eval(grouped_result)
                    self.assertEqual(backend.timings.snapshot()['expert_eval']['count'], 1)
                    from vllm_apple.strata_memory import ProcessMemoryBudget
                    runtime.process_budget = ProcessMemoryBudget(100000, 10000)
                    runtime.memory_observer = lambda: dict(rss_bytes=20000, mlx_active_bytes=10000,
                                                          mlx_cache_bytes=10000, available_bytes=100000)
                    runtime.admit_forward()
                    self.assertEqual(runtime.snapshot()['memory_admission']['admitted_resident_bytes'], 18432)
                    runtime.memory_observer = lambda: dict(rss_bytes=95000, mlx_active_bytes=10000,
                                                          mlx_cache_bytes=10000, available_bytes=100000)
                    with self.assertRaises(MemoryError):
                        runtime.admit_forward()
                    runtime.process_budget = None
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))
                for phase in ('prefill', 'decode'):
                    runtime.set_phase(phase)
                    result = blocks[1](blocks[0](x))
                    mx.eval(result)
                    self.assertLess(float(mx.max(mx.abs(result-expected)).item()), 1e-5)
                runtime.resize(maximum_entries=1, maximum_bytes=9216)
                result = blocks[1](blocks[0](x))
                mx.eval(result)
                self.assertLess(float(mx.max(mx.abs(result-expected)).item()), 1e-5)
                self.assertLessEqual(runtime.snapshot()['residency']['resident_bytes'], 9216)
                runtime.close()
                runtime.close()
                self.assertEqual(runtime.snapshot()['residency']['resident_bytes'], 0)
                with self.assertRaises(RuntimeError):
                    blocks[0](x)
                with self.assertRaises(RuntimeError):
                    runtime.set_phase('decode')

    def test_install_failure_restores_all_layers(self):
        import mlx.core as mx
        from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

        from vllm_apple.mlx_expert_backend import MLXFileExpertBackend
        from vllm_apple.mlx_expert_export import export_switch_experts
        mx.set_default_device(mx.cpu)
        blocks = {i: Qwen3MoeSparseMoeBlock(SimpleNamespace(
            hidden_size=64, moe_intermediate_size=64, num_experts=3,
            num_experts_per_tok=2, norm_topk_prob=True)) for i in range(2)}
        originals = {i: b.switch_mlp for i, b in blocks.items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'experts'
            export_switch_experts({0: originals[0]}, root, model_sha256='a'*64,
                                  maximum_expert_bytes=65536, maximum_total_bytes=262144)
            backend = MLXFileExpertBackend.from_manifest(
                root, expected_model_sha256='a'*64, maximum_file_bytes=65536)
            for invalid in ({'process_budget': object()}, {'memory_observer': object()}):
                with self.assertRaisesRegex(ValueError, 'admission configuration'):
                    StrataRuntime(blocks, backend, maximum_entries=2, maximum_bytes=100000,
                                  **invalid)
                for i, block in blocks.items():
                    self.assertIs(block.switch_mlp, originals[i])
            with self.assertRaises(ValueError):
                StrataRuntime(blocks, backend, maximum_entries=2, maximum_bytes=100000)
            for i, block in blocks.items():
                self.assertIs(block.switch_mlp, originals[i])

    def test_full_model_install_logits_and_manifest_coverage(self):
        import mlx.core as mx
        from mlx_lm.models.qwen3_moe import Model, ModelArgs

        from vllm_apple.mlx_expert_export import export_switch_experts

        mx.set_default_device(mx.cpu)
        model = Model(ModelArgs(
            model_type='qwen3_moe', hidden_size=64, num_hidden_layers=2,
            intermediate_size=64, num_attention_heads=2, num_experts=3,
            num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
            moe_intermediate_size=64, rms_norm_eps=1e-6, vocab_size=32,
            num_key_value_heads=2, head_dim=32, rope_theta=10000,
            tie_word_embeddings=True, max_position_embeddings=128, norm_topk_prob=True))
        tokens = mx.array([[1, 2, 3]])
        expected = model(tokens)
        mx.eval(expected)
        originals = [layer.mlp.switch_mlp for layer in model.layers]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'experts'
            export_switch_experts({i: bank for i, bank in enumerate(originals)}, root,
                                  model_sha256='a'*64, maximum_expert_bytes=65536,
                                  maximum_total_bytes=524288)
            options = dict(expert_root=root, expected_model_sha256='b'*64,
                           maximum_file_bytes=65536, maximum_entries=2,
                           maximum_bytes=98304, mode='grouped')
            with self.assertRaises(ValueError):
                StrataRuntime.from_model(model, **options)
            self.assertTrue(all(layer.mlp.switch_mlp is originals[i]
                                for i, layer in enumerate(model.layers)))
            incomplete = Path(directory)/'incomplete'
            export_switch_experts({0: originals[0]}, incomplete, model_sha256='a'*64,
                                  maximum_expert_bytes=65536, maximum_total_bytes=262144)
            with self.assertRaisesRegex(ValueError, 'coverage'):
                StrataRuntime.from_model(model, **dict(options, expert_root=incomplete,
                                                       expected_model_sha256='a'*64))
            options['expected_model_sha256'] = 'a'*64
            with StrataRuntime.from_model(model, **options) as runtime:
                from vllm_apple.strata_serving import StrataServingModel
                serving_model = StrataServingModel(model, runtime)
                result = serving_model(tokens)
                mx.eval(result)
                self.assertLess(float(mx.max(mx.abs(result-expected)).item()), 1e-5)
                self.assertEqual(runtime.snapshot()['layers'], [0, 1])
                self.assertEqual(runtime.snapshot()['residency']['active_leases'], 0)
            with self.assertRaises(RuntimeError):
                model(tokens)

    def test_cpu_and_metal_cached_decode_all_modes(self):
        """Exercise real cache growth, shrinking and reuse on each local device."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.models.qwen3_moe import Model, ModelArgs

        from vllm_apple.mlx_expert_export import export_switch_experts
        from vllm_apple.strata_serving import StrataServingModel

        previous_device = mx.default_device()
        devices = [mx.cpu] + ([mx.gpu] if mx.metal.is_available() else [])
        try:
            for device in devices:
                mx.set_default_device(device)
                for mode in ('sequential', 'grouped', 'row_batch', 'selection_batch'):
                    with self.subTest(device=str(device), mode=mode):
                        mx.random.seed(11)
                        model = Model(ModelArgs(
                            model_type='qwen3_moe', hidden_size=64, num_hidden_layers=2,
                            intermediate_size=64, num_attention_heads=2, num_experts=3,
                            num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
                            moe_intermediate_size=64, rms_norm_eps=1e-6, vocab_size=32,
                            num_key_value_heads=2, head_dim=32, rope_theta=10000,
                            tie_word_embeddings=True, max_position_embeddings=128,
                            norm_topk_prob=True))
                        for layer in model.layers:
                            nn.quantize(layer.mlp, bits=4, group_size=32)
                        chunks = [mx.array([[1, 2, 3, 4]]), mx.array([[5]]),
                                  mx.array([[6]]), mx.array([[7]])]
                        cache = make_prompt_cache(model)
                        expected = []
                        for chunk in chunks:
                            output = model(chunk, cache=cache)
                            mx.eval(output)
                            expected.append(output)
                        with tempfile.TemporaryDirectory() as directory:
                            root = Path(directory) / 'experts'
                            export_switch_experts(
                                {i: layer.mlp.switch_mlp for i, layer in enumerate(model.layers)},
                                root, model_sha256='a'*64, maximum_expert_bytes=65536,
                                maximum_total_bytes=524288)
                            with StrataRuntime.from_model(
                                    model, expert_root=root, expected_model_sha256='a'*64,
                                    maximum_file_bytes=65536, maximum_entries=3,
                                    maximum_bytes=27648, mode=mode) as runtime:
                                from vllm_apple.strata_memory import (
                                    ProcessMemoryBudget,
                                    observe_process_memory,
                                )
                                observed = observe_process_memory()
                                runtime.process_budget = ProcessMemoryBudget(
                                    max(observed['rss_bytes'], observed['mlx_active_bytes']
                                        + observed['mlx_cache_bytes']) + (256 << 20), 16 << 20)
                                accepted_budget = runtime.process_budget
                                wrapped = StrataServingModel(model, runtime)
                                # A fresh cache represents a second independent request.
                                for _request in range(2):
                                    cache = make_prompt_cache(model)
                                    runtime.resize(maximum_entries=3, maximum_bytes=27648)
                                    for index, chunk in enumerate(chunks):
                                        if index == 1:
                                            runtime.resize(maximum_entries=1, maximum_bytes=9216)
                                        if index == 1:
                                            offsets = [entry.offset for entry in cache]
                                            runtime.process_budget = ProcessMemoryBudget(1, 0)
                                            with self.assertRaises(MemoryError):
                                                wrapped(chunk, cache=cache)
                                            self.assertEqual([entry.offset for entry in cache], offsets)
                                            self.assertEqual(runtime.snapshot()['residency']['active_leases'], 0)
                                            runtime.process_budget = accepted_budget
                                        actual = wrapped(chunk, cache=cache)
                                        mx.eval(actual)
                                        self.assertLess(float(mx.max(mx.abs(
                                            actual - expected[index])).item()), 1e-5)
                                        self.assertIsNotNone(runtime.snapshot()['memory_admission'])
                                        state = runtime.snapshot()['residency']
                                        self.assertEqual(state['active_leases'], 0)
                                        self.assertLessEqual(state['resident_bytes'],
                                                             27648 if index == 0 else 9216)
                                    self.assertTrue(all(entry.offset == 7 for entry in cache))
                                mx.synchronize()
                            self.assertEqual(runtime.snapshot()['residency']['resident_bytes'], 0)
        finally:
            mx.set_default_device(previous_device)
