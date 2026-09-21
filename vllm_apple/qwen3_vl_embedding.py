"""Qwen3-VL ANE output ABI and asynchronous GPU LLM handoff."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Generic, TypeVar

from .device_pipeline import (
    ANEAuxiliaryRoute,
    AsyncEncoderLLMPipeline,
    EncoderLLMPipelineResult,
)
from .execution import ExecutionBackend
from .inference_request import InferenceRequestContext
from .qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec
from .qwen3_vl_coreml import Qwen3VLCoreMLConversionManifest

_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class Qwen3VLVisionEmbeddingBundle:
    """Opaque tensors matching vLLM-Metal's Qwen3-VL encode result."""

    hidden_states: object
    deepstack_visual_embeds: tuple[object, ...]
    grid_thw: tuple[int, int, int]
    token_count: int
    hidden_size: int
    capability_id: str


@dataclass(frozen=True, slots=True)
class Qwen3VLCoreMLPipelineOutput:
    """Ordered numeric outputs produced by one provenance-bound Core ML run."""

    hidden_states: object
    deepstack_visual_embeds: tuple[object, ...]
    grid_thw: tuple[int, int, int]
    graph_id: str


def validate_qwen3_vl_vision_embeddings(
    source: Qwen3VLVisionANEAdapterSpec,
    route: ANEAuxiliaryRoute,
    *,
    grid_thw: tuple[int, int, int],
    hidden_states: object,
    deepstack_visual_embeds: Sequence[object],
) -> Qwen3VLVisionEmbeddingBundle:
    """Validate the exact hidden/deep-stack shape consumed by vLLM-Metal."""
    if (
        not isinstance(source, Qwen3VLVisionANEAdapterSpec)
        or not isinstance(route, ANEAuxiliaryRoute)
        or route.operator != source.operator
        or len(grid_thw) != 3
        or any(type(value) is not int or value <= 0 for value in grid_thw)
    ):
        raise ValueError("invalid Qwen3-VL vision embedding identity")
    frames, height, width = grid_thw
    merge = source.spatial_merge_size
    if height % merge or width % merge:
        raise ValueError("Qwen3-VL vision grid is not spatial-merge aligned")
    token_count = frames * (height // merge) * (width // merge)
    expected_shape = (token_count, source.output_hidden_size)
    if _tensor_shape(hidden_states) != expected_shape:
        raise ValueError("Qwen3-VL hidden-state shape does not match image grid")
    if (
        not isinstance(deepstack_visual_embeds, Sequence)
        or isinstance(deepstack_visual_embeds, (str, bytes, bytearray))
        or len(deepstack_visual_embeds) != len(source.deepstack_visual_indexes)
        or any(_tensor_shape(value) != expected_shape for value in deepstack_visual_embeds)
    ):
        raise ValueError("Qwen3-VL deep-stack embedding shape does not match")
    return Qwen3VLVisionEmbeddingBundle(
        hidden_states,
        tuple(deepstack_visual_embeds),
        grid_thw,
        token_count,
        source.output_hidden_size,
        route.capability_id,
    )


class Qwen3VLANEGPUPipeline(Generic[_Result]):
    """Source-bound Core ML encoder handoff into the existing GPU LLM callback."""

    def __init__(
        self,
        pipeline: AsyncEncoderLLMPipeline,
        *,
        source: Qwen3VLVisionANEAdapterSpec,
        conversion: Qwen3VLCoreMLConversionManifest,
        route: ANEAuxiliaryRoute,
        coreml_graph_id: str | None = None,
    ) -> None:
        if (
            not isinstance(pipeline, AsyncEncoderLLMPipeline)
            or not isinstance(source, Qwen3VLVisionANEAdapterSpec)
            or not isinstance(conversion, Qwen3VLCoreMLConversionManifest)
            or not isinstance(route, ANEAuxiliaryRoute)
            or conversion.source_artifact_fingerprint != source.artifact_fingerprint
            or conversion.model_revision != source.model_revision
            or conversion.output_shape[-1] != source.output_hidden_size
            or route.operator != source.operator
            or route.precision != conversion.compute_precision
            or (
                coreml_graph_id is not None
                and not _is_sha256(coreml_graph_id)
            )
        ):
            raise ValueError("Qwen3-VL ANE/GPU pipeline identity does not match")
        self._pipeline = pipeline
        self._source = source
        self._route = route
        self._coreml_graph_id = coreml_graph_id

    def submit(
        self,
        *,
        grid_thw: tuple[int, int, int],
        encoder_memory_bytes: int,
        llm_backend: ExecutionBackend,
        llm_memory_bytes: int,
        encode: Callable[
            [], tuple[object, Sequence[object]] | Qwen3VLCoreMLPipelineOutput
        ],
        consume: Callable[[Qwen3VLVisionEmbeddingBundle], _Result],
    ) -> Future[EncoderLLMPipelineResult[_Result]]:
        if not callable(encode) or not callable(consume):
            raise ValueError("Qwen3-VL encoder and consumer must be callable")

        def checked_encode() -> Qwen3VLVisionEmbeddingBundle:
            result = encode()
            if isinstance(result, Qwen3VLCoreMLPipelineOutput):
                if (
                    self._coreml_graph_id is None
                    or result.graph_id != self._coreml_graph_id
                    or result.grid_thw != grid_thw
                ):
                    raise ValueError(
                        "Qwen3-VL Core ML output provenance does not match"
                    )
                hidden_states = result.hidden_states
                deepstack = result.deepstack_visual_embeds
            elif self._coreml_graph_id is not None:
                raise ValueError("Qwen3-VL Core ML output provenance is missing")
            elif isinstance(result, tuple) and len(result) == 2:
                hidden_states, deepstack = result
            else:
                raise ValueError("Qwen3-VL Core ML encoder returned an invalid result")
            return validate_qwen3_vl_vision_embeddings(
                self._source,
                self._route,
                grid_thw=grid_thw,
                hidden_states=hidden_states,
                deepstack_visual_embeds=deepstack,
            )

        return self._pipeline.submit(
            self._route,
            encoder_memory_bytes=encoder_memory_bytes,
            llm_backend=llm_backend,
            llm_memory_bytes=llm_memory_bytes,
            encode=checked_encode,
            consume=consume,
        )

    def execute_inline(
        self,
        *,
        grid_thw: tuple[int, int, int],
        encoder_memory_bytes: int,
        llm_backend: ExecutionBackend,
        llm_memory_bytes: int,
        encode: Callable[
            [], tuple[object, Sequence[object]] | Qwen3VLCoreMLPipelineOutput
        ],
        consume: Callable[[Qwen3VLVisionEmbeddingBundle], _Result],
        request_context: InferenceRequestContext | None = None,
    ) -> EncoderLLMPipelineResult[_Result]:
        """Execute on the caller thread while retaining scheduler reservations."""
        if not callable(encode) or not callable(consume):
            raise ValueError("Qwen3-VL encoder and consumer must be callable")

        def checked_encode() -> Qwen3VLVisionEmbeddingBundle:
            result = encode()
            if not isinstance(result, Qwen3VLCoreMLPipelineOutput):
                raise ValueError("Qwen3-VL Core ML output provenance is missing")
            if (
                self._coreml_graph_id is None
                or result.graph_id != self._coreml_graph_id
                or result.grid_thw != grid_thw
            ):
                raise ValueError("Qwen3-VL Core ML output provenance does not match")
            return validate_qwen3_vl_vision_embeddings(
                self._source,
                self._route,
                grid_thw=grid_thw,
                hidden_states=result.hidden_states,
                deepstack_visual_embeds=result.deepstack_visual_embeds,
            )

        return self._pipeline.execute_inline(
            self._route,
            encoder_memory_bytes=encoder_memory_bytes,
            llm_backend=llm_backend,
            llm_memory_bytes=llm_memory_bytes,
            encode=checked_encode,
            consume=consume,
            request_context=request_context,
        )


def build_vllm_metal_qwen3_vl_encode_result(
    bundle: Qwen3VLVisionEmbeddingBundle,
    *,
    result_type: type | None = None,
) -> object:
    """Create Homebrew vLLM-Metal's exact Qwen3-VL vision result object."""
    if not isinstance(bundle, Qwen3VLVisionEmbeddingBundle):
        raise ValueError("Qwen3-VL embedding bundle is invalid")
    if result_type is None:
        try:
            from vllm_metal.multimodal.qwen3_vl import (
                Qwen3VLVisionEncodeResult as result_type,
            )
        except ImportError as error:
            raise RuntimeError("Homebrew vLLM-Metal Qwen3-VL ABI is unavailable") from error
    if not isinstance(result_type, type):
        raise ValueError("Qwen3-VL vLLM-Metal result type is invalid")
    try:
        result = result_type(
            hidden_states=bundle.hidden_states,
            deepstack_visual_embeds=bundle.deepstack_visual_embeds,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("Homebrew vLLM-Metal Qwen3-VL ABI changed") from error
    if (
        getattr(result, "hidden_states", None) is not bundle.hidden_states
        or tuple(getattr(result, "deepstack_visual_embeds", ()))
        != bundle.deepstack_visual_embeds
    ):
        raise RuntimeError("Homebrew vLLM-Metal Qwen3-VL result changed outputs")
    return result


def _tensor_shape(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if (
        shape is None
        or isinstance(shape, (str, bytes, bytearray))
        or not isinstance(shape, Sequence)
    ):
        raise ValueError("Qwen3-VL embedding tensor has no bounded shape")
    normalized = tuple(shape)
    if (
        not 1 <= len(normalized) <= 4
        or any(type(dimension) is not int or dimension <= 0 for dimension in normalized)
    ):
        raise ValueError("Qwen3-VL embedding tensor shape is invalid")
    return normalized


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
