"""Deterministic scene-aware temporal sampling for bounded video inference."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TemporalVideoFrame(Generic[T]):
    frame_id: str
    presentation_seconds: float
    payload: T
    keyframe: bool = False
    scene_change_score: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.frame_id
            or len(self.frame_id) > 128
            or not math.isfinite(self.presentation_seconds)
            or self.presentation_seconds < 0
            or not math.isfinite(self.scene_change_score)
            or not 0 <= self.scene_change_score <= 1
        ):
            raise ValueError("invalid temporal video frame")


@dataclass(frozen=True, slots=True)
class TemporalSamplingReport(Generic[T]):
    input_frames: int
    selected_frames: tuple[TemporalVideoFrame[T], ...]
    dropped_frames: int
    keyframes_selected: int
    scene_changes_selected: int
    maximum_frames: int


def sample_temporal_frames(
    frames: Sequence[TemporalVideoFrame[T]],
    *,
    maximum_frames: int,
    minimum_interval_seconds: float = 0.0,
    scene_change_threshold: float = 0.5,
) -> TemporalSamplingReport[T]:
    """Preserve salient frames, then fill gaps by farthest temporal distance."""
    if (
        type(maximum_frames) is not int
        or not 1 <= maximum_frames <= 4096
        or len(frames) > 1_000_000
        or not math.isfinite(minimum_interval_seconds)
        or minimum_interval_seconds < 0
        or not math.isfinite(scene_change_threshold)
        or not 0 <= scene_change_threshold <= 1
    ):
        raise ValueError("invalid temporal sampling configuration")
    identifiers = [frame.frame_id for frame in frames]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("temporal frame identifiers must be unique")
    ordered = sorted(frames, key=lambda frame: (frame.presentation_seconds, frame.frame_id))
    if not ordered:
        return TemporalSamplingReport(0, (), 0, 0, 0, maximum_frames)

    by_id = {frame.frame_id: frame for frame in ordered}
    selected_ids = {ordered[0].frame_id}
    if maximum_frames > 1:
        selected_ids.add(ordered[-1].frame_id)
    salient = sorted(
        (
            frame for frame in ordered[1:-1]
            if frame.keyframe or frame.scene_change_score >= scene_change_threshold
        ),
        key=lambda frame: (
            not frame.keyframe,
            -frame.scene_change_score,
            frame.presentation_seconds,
            frame.frame_id,
        ),
    )
    for frame in salient:
        if len(selected_ids) >= maximum_frames:
            break
        if _far_enough(frame, selected_ids, by_id, minimum_interval_seconds):
            selected_ids.add(frame.frame_id)

    remaining = [frame for frame in ordered if frame.frame_id not in selected_ids]
    while len(selected_ids) < maximum_frames and remaining:
        eligible = [
            frame for frame in remaining
            if _far_enough(frame, selected_ids, by_id, minimum_interval_seconds)
        ]
        if not eligible:
            break
        candidate = max(
            eligible,
            key=lambda frame: (
                _nearest_distance(frame, selected_ids, by_id),
                frame.scene_change_score,
                -frame.presentation_seconds,
            ),
        )
        selected_ids.add(candidate.frame_id)
        remaining = [frame for frame in remaining if frame.frame_id != candidate.frame_id]

    selected = tuple(frame for frame in ordered if frame.frame_id in selected_ids)
    return TemporalSamplingReport(
        input_frames=len(ordered),
        selected_frames=selected,
        dropped_frames=len(ordered) - len(selected),
        keyframes_selected=sum(frame.keyframe for frame in selected),
        scene_changes_selected=sum(
            frame.scene_change_score >= scene_change_threshold for frame in selected
        ),
        maximum_frames=maximum_frames,
    )


def _far_enough(
    frame: TemporalVideoFrame[object],
    selected_ids: set[str],
    by_id: dict[str, TemporalVideoFrame[object]],
    minimum_interval: float,
) -> bool:
    return all(
        abs(frame.presentation_seconds - by_id[identifier].presentation_seconds)
        >= minimum_interval
        for identifier in selected_ids
    )


def _nearest_distance(
    frame: TemporalVideoFrame[object],
    selected_ids: set[str],
    by_id: dict[str, TemporalVideoFrame[object]],
) -> float:
    return min(
        abs(frame.presentation_seconds - by_id[identifier].presentation_seconds)
        for identifier in selected_ids
    )
