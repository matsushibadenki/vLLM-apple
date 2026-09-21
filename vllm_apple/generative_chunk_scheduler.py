"""Bounded temporal/spatial attention-state planning for video generation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

MAX_GENERATIVE_CHUNKS = 65_536


@dataclass(frozen=True, slots=True)
class AttentionStateRegion:
    frame_start: int
    frame_count: int
    y: int
    x: int
    height: int
    width: int
    state_bytes: int

    def __post_init__(self) -> None:
        if (min(self.frame_start, self.y, self.x) < 0
                or min(self.frame_count, self.height, self.width, self.state_bytes) <= 0):
            raise ValueError("invalid attention state region")

    @property
    def state_id(self) -> str:
        return hashlib.sha256(
            json.dumps(self.__dict__ if hasattr(self, "__dict__") else {
                "frame_start": self.frame_start, "frame_count": self.frame_count,
                "y": self.y, "x": self.x, "height": self.height,
                "width": self.width, "state_bytes": self.state_bytes,
            }, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class GenerativeChunkTask:
    task_id: str
    region: AttentionStateRegion
    temporal_context_start: int
    temporal_context_count: int
    dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenerativeChunkPlan:
    plan_id: str
    frames: int
    height: int
    width: int
    temporal_chunk: int
    temporal_overlap: int
    tile_height: int
    tile_width: int
    maximum_concurrency: int
    estimated_peak_bytes: int
    tasks: tuple[GenerativeChunkTask, ...]


def build_generative_chunk_plan(
    *,
    frames: int,
    height: int,
    width: int,
    temporal_chunk: int,
    temporal_overlap: int,
    tile_height: int,
    tile_width: int,
    bytes_per_attention_element: int,
    attention_channels: int,
    maximum_concurrency: int,
    memory_ceiling_bytes: int,
) -> GenerativeChunkPlan:
    integers = (
        frames, height, width, temporal_chunk, tile_height, tile_width,
        bytes_per_attention_element, attention_channels, maximum_concurrency,
        memory_ceiling_bytes,
    )
    if (any(type(value) is not int or value <= 0 for value in integers)
            or type(temporal_overlap) is not int or temporal_overlap < 0
            or temporal_overlap >= temporal_chunk
            or maximum_concurrency > 64):
        raise ValueError("invalid generative chunk configuration")
    temporal_starts = tuple(range(0, frames, temporal_chunk))
    y_starts = tuple(range(0, height, tile_height))
    x_starts = tuple(range(0, width, tile_width))
    task_count = len(temporal_starts) * len(y_starts) * len(x_starts)
    if task_count > MAX_GENERATIVE_CHUNKS:
        raise ValueError("generative chunk plan exceeds task limit")
    tasks = []
    prior_by_tile: dict[tuple[int, int], str] = {}
    maximum_state_bytes = 0
    for frame_start in temporal_starts:
        frame_count = min(temporal_chunk, frames - frame_start)
        context_start = max(0, frame_start - temporal_overlap)
        context_count = frame_start + frame_count - context_start
        for y in y_starts:
            tile_h = min(tile_height, height - y)
            for x in x_starts:
                tile_w = min(tile_width, width - x)
                state_bytes = (
                    context_count * tile_h * tile_w
                    * attention_channels * bytes_per_attention_element
                )
                if state_bytes > memory_ceiling_bytes:
                    raise ValueError("single attention state exceeds memory ceiling")
                region = AttentionStateRegion(
                    frame_start, frame_count, y, x, tile_h, tile_w, state_bytes
                )
                task_id = hashlib.sha256(
                    f"{region.state_id}:{context_start}:{context_count}".encode()
                ).hexdigest()
                previous = prior_by_tile.get((y, x))
                dependencies = (previous,) if previous is not None else ()
                tasks.append(GenerativeChunkTask(
                    task_id, region, context_start, context_count, dependencies
                ))
                prior_by_tile[(y, x)] = task_id
                maximum_state_bytes = max(maximum_state_bytes, state_bytes)
    maximum_concurrency = min(maximum_concurrency, len(tasks))
    estimated_peak = maximum_state_bytes * maximum_concurrency
    if estimated_peak > memory_ceiling_bytes:
        maximum_concurrency = max(1, memory_ceiling_bytes // maximum_state_bytes)
        estimated_peak = maximum_state_bytes * maximum_concurrency
    identity = {
        "frames": frames, "height": height, "width": width,
        "temporal_chunk": temporal_chunk, "temporal_overlap": temporal_overlap,
        "tile_height": tile_height, "tile_width": tile_width,
        "bytes_per_attention_element": bytes_per_attention_element,
        "attention_channels": attention_channels,
        "maximum_concurrency": maximum_concurrency,
        "estimated_peak_bytes": estimated_peak,
        "tasks": [{
            "task_id": task.task_id,
            "state_id": task.region.state_id,
            "context_start": task.temporal_context_start,
            "context_count": task.temporal_context_count,
            "dependencies": list(task.dependencies),
        } for task in tasks],
    }
    plan_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return GenerativeChunkPlan(
        plan_id, frames, height, width, temporal_chunk, temporal_overlap,
        tile_height, tile_width, maximum_concurrency, estimated_peak, tuple(tasks),
    )
