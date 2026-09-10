from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Callable, Iterator


_BLOCK_CLASSES = (
    (
        "mflux.models.flux2.model.flux2_transformer.transformer_block",
        "Flux2TransformerBlock",
    ),
    (
        "mflux.models.flux2.model.flux2_transformer.single_transformer_block",
        "Flux2SingleTransformerBlock",
    ),
)
_PREDICT_CLASS = (
    "mflux.models.flux2.variants.txt2img.flux2_klein",
    "Flux2Klein",
)
_ATTENTION_CLASS = (
    "mflux.models.flux.model.flux_transformer.common.attention_utils",
    "AttentionUtils",
)
_FEED_FORWARD_CLASS = (
    "mflux.models.flux2.model.flux2_transformer.feed_forward",
    "Flux2FeedForward",
)
_PARALLEL_ATTENTION_CLASS = (
    "mflux.models.flux2.model.flux2_transformer.parallel_self_attention",
    "Flux2ParallelSelfAttention",
)
_DIAGNOSTIC_CODES = frozenset(
    {
        "blockwise_block_call_failed",
        "blockwise_materialization_failed",
        "blockwise_cache_clear_failed",
        "attention_chunk_call_failed",
        "attention_chunk_materialization_failed",
        "attention_chunk_cache_clear_failed",
        "mlp_sequence_chunk_call_failed",
    }
)


class MLXGenMemoryProfileError(RuntimeError):
    def __init__(self, diagnostic_code: str) -> None:
        if diagnostic_code not in _DIAGNOSTIC_CODES:
            raise ValueError("unsupported MLX-Gen memory profile diagnostic code")
        super().__init__(diagnostic_code)
        self.diagnostic_code = diagnostic_code


def _install_uncompiled_predict(
    module_loader: Callable[[str], object],
    mlx: object,
    patches: list[tuple[type, str, object, object]],
) -> None:
    predict_module, predict_class_name = _PREDICT_CLASS
    predict_class = getattr(module_loader(predict_module), predict_class_name)
    original_descriptor = predict_class.__dict__.get("_predict")
    if not isinstance(original_descriptor, staticmethod):
        raise RuntimeError("MLX-Gen FLUX.2 predict contract is incompatible")
    original_predict = original_descriptor.__func__

    @wraps(original_predict)
    def uncompiled_predict(transformer):
        original_compile = mlx.compile
        try:
            mlx.compile = lambda function, *args, **kwargs: function
            return original_predict(transformer)
        finally:
            mlx.compile = original_compile

    installed = staticmethod(uncompiled_predict)
    predict_class._predict = installed
    patches.append((predict_class, "_predict", original_descriptor, installed))


def _restore_patches(patches: list[tuple[type, str, object, object]]) -> None:
    for target_class, attribute, original, installed in reversed(patches):
        if target_class.__dict__.get(attribute) is installed:
            setattr(target_class, attribute, original)


@contextmanager
def flux2_blockwise_residency(
    module_loader: Callable[[str], object],
) -> Iterator[None]:
    """Bound FLUX.2's lazy graph lifetime to one transformer block.

    MLX evaluates lazily. Materializing each block output prevents a complete
    denoising pass from remaining in one growing graph, while preserving the
    model's operations and weights. The patch is process-local and restored so
    tests and embedded callers cannot leak the experimental profile.
    """

    mlx = module_loader("mlx.core")
    patches: list[tuple[type, str, object, object]] = []
    try:
        _install_uncompiled_predict(module_loader, mlx, patches)

        for module_name, class_name in _BLOCK_CLASSES:
            block_class = getattr(module_loader(module_name), class_name)
            original = block_class.__call__

            @wraps(original)
            def materialized(self, *args, __original=original, **kwargs):
                try:
                    result = __original(self, *args, **kwargs)
                except ValueError as error:
                    raise MLXGenMemoryProfileError("blockwise_block_call_failed") from error
                outputs = result if isinstance(result, tuple) else (result,)
                try:
                    mlx.eval(*outputs)
                except ValueError as error:
                    raise MLXGenMemoryProfileError("blockwise_materialization_failed") from error
                try:
                    mlx.clear_cache()
                except ValueError as error:
                    raise MLXGenMemoryProfileError("blockwise_cache_clear_failed") from error
                return result

            block_class.__call__ = materialized
            patches.append((block_class, "__call__", original, materialized))
        yield
    finally:
        _restore_patches(patches)


@contextmanager
def flux2_attention_query_chunking(
    module_loader: Callable[[str], object],
    *,
    chunk_size: int,
) -> Iterator[None]:
    """Materialize fused-SDPA output in bounded query-axis chunks."""

    if chunk_size != 512:
        raise ValueError("formal FLUX.2 attention query chunk size must be 512")
    mlx = module_loader("mlx.core")
    patches: list[tuple[type, str, object, object]] = []
    try:
        _install_uncompiled_predict(module_loader, mlx, patches)
        attention_module, attention_class_name = _ATTENTION_CLASS
        attention_class = getattr(module_loader(attention_module), attention_class_name)
        original_descriptor = attention_class.__dict__.get("compute_attention")
        if not isinstance(original_descriptor, staticmethod):
            raise RuntimeError("MLX-Gen attention contract is incompatible")
        original_attention = original_descriptor.__func__

        @wraps(original_attention)
        def chunked_attention(
            query,
            key,
            value,
            batch_size,
            num_heads,
            head_dim,
            mask=None,
        ):
            query_length = query.shape[2]
            if query_length <= chunk_size:
                return original_attention(
                    query, key, value, batch_size, num_heads, head_dim, mask
                )
            chunks = []
            for start in range(0, query_length, chunk_size):
                stop = min(query_length, start + chunk_size)
                chunk_mask = mask
                if mask is not None and mask.shape[-2] == query_length:
                    chunk_mask = mask[..., start:stop, :]
                try:
                    output = original_attention(
                        query[:, :, start:stop, :],
                        key,
                        value,
                        batch_size,
                        num_heads,
                        head_dim,
                        chunk_mask,
                    )
                except ValueError as error:
                    raise MLXGenMemoryProfileError("attention_chunk_call_failed") from error
                try:
                    mlx.eval(output)
                except ValueError as error:
                    raise MLXGenMemoryProfileError(
                        "attention_chunk_materialization_failed"
                    ) from error
                try:
                    mlx.clear_cache()
                except ValueError as error:
                    raise MLXGenMemoryProfileError(
                        "attention_chunk_cache_clear_failed"
                    ) from error
                chunks.append(output)
            return mlx.concatenate(chunks, axis=1)

        installed = staticmethod(chunked_attention)
        attention_class.compute_attention = installed
        patches.append(
            (attention_class, "compute_attention", original_descriptor, installed)
        )
        yield
    finally:
        _restore_patches(patches)


@contextmanager
def flux2_mlp_sequence_chunking(
    module_loader: Callable[[str], object],
    *,
    chunk_size: int,
) -> Iterator[None]:
    """Chunk only FLUX.2 expansion projections while retaining outer compile."""

    if chunk_size != 512:
        raise ValueError("formal FLUX.2 MLP sequence chunk size must be 512")
    mlx = module_loader("mlx.core")
    linear_class = module_loader("mlx.nn").Linear
    target_linear_ids: set[int] = set()
    patches: list[tuple[type, str, object, object]] = []
    try:
        feed_forward_module, feed_forward_name = _FEED_FORWARD_CLASS
        feed_forward_class = getattr(module_loader(feed_forward_module), feed_forward_name)
        original_feed_forward_init = feed_forward_class.__init__

        @wraps(original_feed_forward_init)
        def feed_forward_init(self, *args, **kwargs):
            original_feed_forward_init(self, *args, **kwargs)
            target_linear_ids.add(id(self.linear_in))

        feed_forward_class.__init__ = feed_forward_init
        patches.append(
            (
                feed_forward_class,
                "__init__",
                original_feed_forward_init,
                feed_forward_init,
            )
        )

        parallel_module, parallel_name = _PARALLEL_ATTENTION_CLASS
        parallel_class = getattr(module_loader(parallel_module), parallel_name)
        original_parallel_init = parallel_class.__init__

        @wraps(original_parallel_init)
        def parallel_init(self, *args, **kwargs):
            original_parallel_init(self, *args, **kwargs)
            target_linear_ids.add(id(self.to_qkv_mlp_proj))

        parallel_class.__init__ = parallel_init
        patches.append(
            (parallel_class, "__init__", original_parallel_init, parallel_init)
        )

        original_linear_call = linear_class.__call__

        @wraps(original_linear_call)
        def chunked_linear(self, value):
            if id(self) not in target_linear_ids or value.ndim != 3:
                return original_linear_call(self, value)
            sequence_length = value.shape[1]
            if sequence_length <= chunk_size:
                return original_linear_call(self, value)
            outputs = []
            for start in range(0, sequence_length, chunk_size):
                stop = min(sequence_length, start + chunk_size)
                try:
                    outputs.append(original_linear_call(self, value[:, start:stop, :]))
                except ValueError as error:
                    raise MLXGenMemoryProfileError(
                        "mlp_sequence_chunk_call_failed"
                    ) from error
            return mlx.concatenate(outputs, axis=1)

        linear_class.__call__ = chunked_linear
        patches.append((linear_class, "__call__", original_linear_call, chunked_linear))
        yield
    finally:
        target_linear_ids.clear()
        _restore_patches(patches)
