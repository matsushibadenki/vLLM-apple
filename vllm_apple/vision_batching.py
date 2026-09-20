"""Deterministic, bounded batching for multimodal vision encoder requests."""
from __future__ import annotations

from dataclasses import dataclass

from .vision_frontend import VisionImageInput


@dataclass(frozen=True, slots=True)
class VisionBatchCompatibility:
    model_revision: str
    preprocessing_fingerprint: str
    encoder_fingerprint: str
    image_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if (
            not self.model_revision
            or not self.preprocessing_fingerprint
            or not self.encoder_fingerprint
            or len(self.image_shape) != 3
            or any(type(value) is not int or value <= 0 for value in self.image_shape)
        ):
            raise ValueError("invalid vision batch compatibility")


@dataclass(frozen=True, slots=True)
class VisionBatchRequest:
    request_id: str
    compatibility: VisionBatchCompatibility
    images: tuple[VisionImageInput, ...]
    patches_per_image: int

    def __post_init__(self) -> None:
        if (
            not self.request_id
            or len(self.request_id) > 128
            or not 1 <= len(self.images) <= 32
            or not 1 <= self.patches_per_image <= 1_048_576
        ):
            raise ValueError("invalid vision batch request")

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def patch_count(self) -> int:
        return len(self.images) * self.patches_per_image

    @property
    def encoded_bytes(self) -> int:
        return sum(len(image.data) for image in self.images)


@dataclass(frozen=True, slots=True)
class VisionBatch:
    compatibility: VisionBatchCompatibility
    requests: tuple[VisionBatchRequest, ...]
    image_count: int
    patch_count: int
    encoded_bytes: int

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)


@dataclass(frozen=True, slots=True)
class VisionBatchLimits:
    maximum_requests: int = 8
    maximum_images: int = 16
    maximum_patches: int = 16_384
    maximum_encoded_bytes: int = 24 * 1024 * 1024

    def __post_init__(self) -> None:
        values = (
            self.maximum_requests,
            self.maximum_images,
            self.maximum_patches,
            self.maximum_encoded_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("invalid vision batch limits")


def plan_multimodal_batches(
    requests: tuple[VisionBatchRequest, ...] | list[VisionBatchRequest],
    limits: VisionBatchLimits = VisionBatchLimits(),
) -> tuple[VisionBatch, ...]:
    """Create stable compatible batches while keeping each request atomic."""
    if len(requests) > 4096:
        raise ValueError("too many vision batch requests")
    identifiers = [request.request_id for request in requests]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("vision batch request identifiers must be unique")

    pending: dict[VisionBatchCompatibility, list[VisionBatchRequest]] = {}
    compatibility_order: list[VisionBatchCompatibility] = []
    for request in requests:
        _require_request_fits(request, limits)
        if request.compatibility not in pending:
            pending[request.compatibility] = []
            compatibility_order.append(request.compatibility)
        pending[request.compatibility].append(request)

    batches: list[VisionBatch] = []
    for compatibility in compatibility_order:
        current: list[VisionBatchRequest] = []
        images = patches = encoded_bytes = 0
        for request in pending[compatibility]:
            would_overflow = current and (
                len(current) + 1 > limits.maximum_requests
                or images + request.image_count > limits.maximum_images
                or patches + request.patch_count > limits.maximum_patches
                or encoded_bytes + request.encoded_bytes > limits.maximum_encoded_bytes
            )
            if would_overflow:
                batches.append(_make_batch(compatibility, current))
                current = []
                images = patches = encoded_bytes = 0
            current.append(request)
            images += request.image_count
            patches += request.patch_count
            encoded_bytes += request.encoded_bytes
        if current:
            batches.append(_make_batch(compatibility, current))
    return tuple(batches)


def _require_request_fits(request: VisionBatchRequest, limits: VisionBatchLimits) -> None:
    if (
        request.image_count > limits.maximum_images
        or request.patch_count > limits.maximum_patches
        or request.encoded_bytes > limits.maximum_encoded_bytes
    ):
        raise ValueError(f"vision request {request.request_id!r} exceeds batch limits")


def _make_batch(
    compatibility: VisionBatchCompatibility,
    requests: list[VisionBatchRequest],
) -> VisionBatch:
    return VisionBatch(
        compatibility=compatibility,
        requests=tuple(requests),
        image_count=sum(request.image_count for request in requests),
        patch_count=sum(request.patch_count for request in requests),
        encoded_bytes=sum(request.encoded_bytes for request in requests),
    )
