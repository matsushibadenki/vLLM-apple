"""Native multi-layer ownership, rollback and lifecycle integration."""
import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

from vllm_apple.strata_runtime import StrataRuntime


class StrataRuntimeTests(unittest.TestCase):
    def test_multilayer_shared_budget_and_shutdown(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

        from vllm_apple.mlx_expert_backend import MLXFileExpertBackend
        from vllm_apple.mlx_expert_export import export_switch_experts
        mx.set_default_device(mx.cpu)
        for mode in ('sequential', 'grouped', 'row_batch'):
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
