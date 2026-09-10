import unittest
from types import SimpleNamespace

from vllm_apple.mlx_gen_memory_profile import (
    MLXGenMemoryProfileError,
    flux2_attention_query_chunking,
    flux2_blockwise_residency,
    flux2_mlp_sequence_chunking,
)


class _DoubleBlock:
    def __call__(self, value):
        return value, value + 1


class _SingleBlock:
    def __call__(self, value):
        return value + 2


class _Flux2Klein:
    mlx = None

    @staticmethod
    def _predict(transformer):
        return _Flux2Klein.mlx.compile(transformer)


class _MLX:
    compile = staticmethod(lambda function: ("compiled", function))


class _Query:
    def __init__(self, length):
        self.length = length
        self.shape = (1, 2, length, 4)

    def __getitem__(self, key):
        query_slice = key[2]
        return _Query(query_slice.stop - query_slice.start)


class _AttentionUtils:
    calls = []

    @staticmethod
    def compute_attention(query, key, value, batch_size, num_heads, head_dim, mask=None):
        _AttentionUtils.calls.append((query.length, mask))
        return ("attention", query.length)


class _Value:
    def __init__(self, length):
        self.length = length
        self.ndim = 3
        self.shape = (1, length, 8)

    def __getitem__(self, key):
        sequence_slice = key[1]
        return _Value(sequence_slice.stop - sequence_slice.start)


class _Linear:
    calls = []

    def __call__(self, value):
        _Linear.calls.append((id(self), value.length))
        return ("linear", value.length)


class _FeedForward:
    def __init__(self):
        self.linear_in = _Linear()


class _ParallelAttention:
    def __init__(self):
        self.to_qkv_mlp_proj = _Linear()


class MLXGenMemoryProfileTests(unittest.TestCase):
    def test_mlp_projection_chunking_keeps_outer_compile_and_restores_classes(self) -> None:
        mlx = SimpleNamespace(
            concatenate=lambda values, axis: ("concatenate", tuple(values), axis)
        )
        modules = {
            "mlx.core": mlx,
            "mlx.nn": SimpleNamespace(Linear=_Linear),
            "mflux.models.flux2.model.flux2_transformer.feed_forward": SimpleNamespace(
                Flux2FeedForward=_FeedForward
            ),
            "mflux.models.flux2.model.flux2_transformer.parallel_self_attention": SimpleNamespace(
                Flux2ParallelSelfAttention=_ParallelAttention
            ),
        }
        original_linear = _Linear.__call__
        original_feed_forward = _FeedForward.__init__
        original_parallel = _ParallelAttention.__init__
        untagged = _Linear()
        _Linear.calls = []

        with flux2_mlp_sequence_chunking(modules.__getitem__, chunk_size=512):
            feed_forward = _FeedForward()
            parallel = _ParallelAttention()
            feed_result = feed_forward.linear_in(_Value(1200))
            parallel_result = parallel.to_qkv_mlp_proj(_Value(1025))
            untagged_result = untagged(_Value(1200))

        lengths = [length for _, length in _Linear.calls]
        self.assertEqual(lengths, [512, 512, 176, 512, 512, 1, 1200])
        self.assertEqual(feed_result[0], "concatenate")
        self.assertEqual(parallel_result[0], "concatenate")
        self.assertEqual(untagged_result, ("linear", 1200))
        self.assertIs(_Linear.__call__, original_linear)
        self.assertIs(_FeedForward.__init__, original_feed_forward)
        self.assertIs(_ParallelAttention.__init__, original_parallel)

    def test_attention_query_chunks_are_materialized_and_restored(self) -> None:
        events = []
        mlx = SimpleNamespace(
            eval=lambda *values: events.append(("eval", values)),
            clear_cache=lambda: events.append(("clear",)),
            compile=_MLX.compile,
            concatenate=lambda values, axis: ("concatenate", tuple(values), axis),
        )
        _Flux2Klein.mlx = mlx
        _AttentionUtils.calls = []
        modules = {
            "mlx.core": mlx,
            "mflux.models.flux2.variants.txt2img.flux2_klein": SimpleNamespace(
                Flux2Klein=_Flux2Klein
            ),
            "mflux.models.flux.model.flux_transformer.common.attention_utils": SimpleNamespace(
                AttentionUtils=_AttentionUtils
            ),
        }
        original = _AttentionUtils.__dict__["compute_attention"]

        with flux2_attention_query_chunking(modules.__getitem__, chunk_size=512):
            result = _AttentionUtils.compute_attention(
                _Query(1200), object(), object(), 1, 2, 4
            )

        self.assertEqual(_AttentionUtils.calls, [(512, None), (512, None), (176, None)])
        self.assertEqual(result[0], "concatenate")
        self.assertEqual(result[2], 1)
        self.assertEqual([event[0] for event in events], ["eval", "clear"] * 3)
        self.assertIs(_AttentionUtils.__dict__["compute_attention"], original)

    def test_materializes_each_block_and_restores_classes(self) -> None:
        calls = []
        mlx = SimpleNamespace(
            eval=lambda *values: calls.append(("eval", values)),
            clear_cache=lambda: calls.append(("clear",)),
            compile=_MLX.compile,
        )
        _Flux2Klein.mlx = mlx
        modules = {
            "mlx.core": mlx,
            "mflux.models.flux2.model.flux2_transformer.transformer_block": SimpleNamespace(
                Flux2TransformerBlock=_DoubleBlock
            ),
            "mflux.models.flux2.model.flux2_transformer.single_transformer_block": SimpleNamespace(
                Flux2SingleTransformerBlock=_SingleBlock
            ),
            "mflux.models.flux2.variants.txt2img.flux2_klein": SimpleNamespace(
                Flux2Klein=_Flux2Klein
            ),
        }
        original_double = _DoubleBlock.__call__
        original_single = _SingleBlock.__call__
        original_predict = _Flux2Klein.__dict__["_predict"]

        with flux2_blockwise_residency(modules.__getitem__):
            def transformer():
                return None

            self.assertIs(_Flux2Klein._predict(transformer), transformer)
            self.assertEqual(_DoubleBlock()(3), (3, 4))
            self.assertEqual(_SingleBlock()(3), 5)

        self.assertIs(_DoubleBlock.__call__, original_double)
        self.assertIs(_SingleBlock.__call__, original_single)
        self.assertIs(_Flux2Klein.__dict__["_predict"], original_predict)
        self.assertEqual(
            calls,
            [("eval", (3, 4)), ("clear",), ("eval", (5,)), ("clear",)],
        )

    def test_restores_already_patched_classes_when_installation_fails(self) -> None:
        mlx = SimpleNamespace(
            eval=lambda *values: None,
            clear_cache=lambda: None,
            compile=_MLX.compile,
        )
        _Flux2Klein.mlx = mlx
        modules = {
            "mlx.core": mlx,
            "mflux.models.flux2.variants.txt2img.flux2_klein": SimpleNamespace(
                Flux2Klein=_Flux2Klein
            ),
            "mflux.models.flux2.model.flux2_transformer.transformer_block": SimpleNamespace(
                Flux2TransformerBlock=_DoubleBlock
            ),
        }
        original = _DoubleBlock.__call__
        original_predict = _Flux2Klein.__dict__["_predict"]

        with self.assertRaises(KeyError):
            with flux2_blockwise_residency(modules.__getitem__):
                pass

        self.assertIs(_DoubleBlock.__call__, original)
        self.assertIs(_Flux2Klein.__dict__["_predict"], original_predict)

    def test_exposes_only_a_fixed_stage_code_for_materialization_failure(self) -> None:
        mlx = SimpleNamespace(
            eval=lambda *values: (_ for _ in ()).throw(ValueError("private detail")),
            clear_cache=lambda: None,
            compile=_MLX.compile,
        )
        _Flux2Klein.mlx = mlx
        modules = {
            "mlx.core": mlx,
            "mflux.models.flux2.variants.txt2img.flux2_klein": SimpleNamespace(
                Flux2Klein=_Flux2Klein
            ),
            "mflux.models.flux2.model.flux2_transformer.transformer_block": SimpleNamespace(
                Flux2TransformerBlock=_DoubleBlock
            ),
            "mflux.models.flux2.model.flux2_transformer.single_transformer_block": SimpleNamespace(
                Flux2SingleTransformerBlock=_SingleBlock
            ),
        }

        with self.assertRaises(MLXGenMemoryProfileError) as raised:
            with flux2_blockwise_residency(modules.__getitem__):
                _SingleBlock()(3)

        self.assertEqual(raised.exception.diagnostic_code, "blockwise_materialization_failed")
        self.assertNotIn("private detail", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
