"""Backend-neutral sampled-frame video VLM integration."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from .video_cache import VideoArtifactCache, VideoCacheKey, VideoCacheKind
from .video_temporal_sampler import TemporalVideoFrame, sample_temporal_frames

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class VideoFrameEmbedding(Generic[E]):
    frame_id: str
    value: E
    size_bytes: int
    output_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.frame_id
            or type(self.size_bytes) is not int
            or not 1 <= self.size_bytes <= 1 << 34
            or not _is_digest(self.output_sha256)
        ):
            raise ValueError("invalid video frame embedding")


class VideoFrameEncoder(Protocol[T, E]):
    def encode(
        self, frames: tuple[TemporalVideoFrame[T], ...]
    ) -> tuple[VideoFrameEmbedding[E], ...]: ...


class VideoLanguageBackend(Protocol[E]):
    def generate(self, prompt: str, embeddings: tuple[VideoFrameEmbedding[E], ...]) -> str: ...


@dataclass(frozen=True, slots=True)
class VideoVLMRequest(Generic[T]):
    prompt: str
    video_sha256: str
    transform_fingerprint: str
    model_fingerprint: str
    frames: tuple[TemporalVideoFrame[T], ...]
    maximum_frames: int
    minimum_interval_seconds: float = 0.0
    scene_change_threshold: float = 0.5

    def __post_init__(self) -> None:
        if (
            not self.prompt.strip()
            or len(self.prompt) > 32_768
            or any(not _is_digest(value) for value in (
                self.video_sha256,
                self.transform_fingerprint,
                self.model_fingerprint,
            ))
        ):
            raise ValueError("invalid video VLM request")


@dataclass(frozen=True, slots=True)
class VideoVLMResult:
    text: str
    selected_frame_ids: tuple[str, ...]
    cache_hits: int
    cache_misses: int


class VideoVLMIntegrator(Generic[T, E]):
    """Sample, cache, encode, order, and consume video frame embeddings."""

    def __init__(
        self,
        encoder: VideoFrameEncoder[T, E],
        language_backend: VideoLanguageBackend[E],
        cache: VideoArtifactCache[VideoFrameEmbedding[E]],
    ) -> None:
        if (
            not callable(getattr(encoder, "encode", None))
            or not callable(getattr(language_backend, "generate", None))
            or not isinstance(cache, VideoArtifactCache)
        ):
            raise ValueError("invalid video VLM integration")
        self._encoder = encoder
        self._language_backend = language_backend
        self._cache = cache

    def run(self, request: VideoVLMRequest[T]) -> VideoVLMResult:
        sampled = sample_temporal_frames(
            request.frames,
            maximum_frames=request.maximum_frames,
            minimum_interval_seconds=request.minimum_interval_seconds,
            scene_change_threshold=request.scene_change_threshold,
        )
        if not sampled.selected_frames:
            raise ValueError("video VLM request contains no frames")
        resolved: dict[str, VideoFrameEmbedding[E]] = {}
        missing = []
        cache_hits = 0
        for frame in sampled.selected_frames:
            key = _cache_key(request, frame)
            cached = self._cache.get(key)
            if cached is None:
                missing.append(frame)
            else:
                if cached.frame_id != frame.frame_id:
                    raise RuntimeError("cached video embedding frame ID mismatch")
                resolved[frame.frame_id] = cached
                cache_hits += 1
        if missing:
            generated = self._encoder.encode(tuple(missing))
            if len(generated) != len(missing):
                raise RuntimeError("video encoder returned an invalid embedding count")
            expected_ids = [frame.frame_id for frame in missing]
            returned_ids = [embedding.frame_id for embedding in generated]
            if returned_ids != expected_ids or len(set(returned_ids)) != len(returned_ids):
                raise RuntimeError("video encoder returned misordered frame embeddings")
            for frame, embedding in zip(missing, generated):
                resolved[frame.frame_id] = embedding
                self._cache.put(
                    _cache_key(request, frame), embedding, size_bytes=embedding.size_bytes
                )
        ordered = tuple(resolved[frame.frame_id] for frame in sampled.selected_frames)
        text = self._language_backend.generate(request.prompt.strip(), ordered)
        if not isinstance(text, str) or not text.strip() or len(text) > 65_536:
            raise RuntimeError("video language backend returned an invalid response")
        return VideoVLMResult(
            text.strip(),
            tuple(frame.frame_id for frame in sampled.selected_frames),
            cache_hits,
            len(missing),
        )


def _cache_key(
    request: VideoVLMRequest[object], frame: TemporalVideoFrame[object]
) -> VideoCacheKey:
    start = round(frame.presentation_seconds * 1_000_000)
    return VideoCacheKey(
        VideoCacheKind.EMBEDDING,
        request.video_sha256,
        request.transform_fingerprint,
        request.model_fingerprint,
        start,
        start + 1,
    )


def embedding_digest(values: bytes) -> str:
    if not isinstance(values, bytes) or not values:
        raise ValueError("embedding bytes cannot be empty")
    return hashlib.sha256(values).hexdigest()


def _is_digest(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
