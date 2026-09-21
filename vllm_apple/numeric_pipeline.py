"""Bounded numeric tile pipeline with prefetch, barriers and bandwidth admission."""
from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar

MAX_NUMERIC_PIPELINE_TILES = 65_536
MAX_NUMERIC_PIPELINE_TILE_BYTES = 16 * 1024 * 1024
MAX_NUMERIC_PIPELINE_BANDWIDTH = 1 << 50
_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class NumericPipelineReport(Generic[_Result]):
    results: tuple[_Result, ...]
    tile_count: int
    source_bytes: int
    converted_bytes: int
    peak_buffered_bytes: int
    reserved_bandwidth_bytes_per_second: int
    elapsed_nanoseconds: int
    output_sha256: str
    completion_barriers: int


class NumericBandwidthLease:
    def __init__(self, ledger: "NumericBandwidthLedger", amount: int) -> None:
        self._ledger = ledger
        self._amount = amount
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        with self._ledger._lock:
            self._ledger._used -= self._amount
        self._released = True

    def __enter__(self) -> "NumericBandwidthLease":
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class NumericBandwidthLedger:
    def __init__(self, ceiling_bytes_per_second: int) -> None:
        if (type(ceiling_bytes_per_second) is not int
                or not 1 <= ceiling_bytes_per_second <= MAX_NUMERIC_PIPELINE_BANDWIDTH):
            raise ValueError("invalid numeric bandwidth ceiling")
        self.ceiling_bytes_per_second = ceiling_bytes_per_second
        self._used = 0
        self._lock = threading.Lock()

    def reserve(self, bytes_per_second: int) -> NumericBandwidthLease:
        if type(bytes_per_second) is not int or bytes_per_second <= 0:
            raise ValueError("invalid numeric bandwidth reservation")
        with self._lock:
            if self._used + bytes_per_second > self.ceiling_bytes_per_second:
                raise ValueError("numeric bandwidth ceiling exceeded")
            self._used += bytes_per_second
        return NumericBandwidthLease(self, bytes_per_second)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "ceiling_bytes_per_second": self.ceiling_bytes_per_second,
                "reserved_bytes_per_second": self._used,
                "available_bytes_per_second": self.ceiling_bytes_per_second - self._used,
            }


class NumericTilePipeline:
    def __init__(
        self,
        ledger: NumericBandwidthLedger,
        *,
        reserved_bandwidth_bytes_per_second: int,
        buffer_count: int = 2,
    ) -> None:
        if buffer_count not in {1, 2}:
            raise ValueError("numeric tile pipeline requires one or two buffers")
        self._ledger = ledger
        self._bandwidth = reserved_bandwidth_bytes_per_second
        self._buffer_count = buffer_count

    def execute(
        self,
        *,
        tile_count: int,
        read_tile: Callable[[int], bytes],
        convert_tile: Callable[[bytes], bytes],
        consume_tile: Callable[[int, bytes], _Result],
        cancellation: threading.Event | None = None,
    ) -> NumericPipelineReport[_Result]:
        if (type(tile_count) is not int or not 1 <= tile_count <= MAX_NUMERIC_PIPELINE_TILES
                or not callable(read_tile) or not callable(convert_tile)
                or not callable(consume_tile)
                or (cancellation is not None and not isinstance(cancellation, threading.Event))):
            raise ValueError("invalid numeric tile pipeline request")
        cancel = cancellation or threading.Event()
        source_bytes = converted_bytes = peak_buffered = barriers = 0
        prefetched_bytes = 0
        buffer_lock = threading.Lock()
        digest = hashlib.sha256(b"vllm-apple-numeric-pipeline-v1\0")
        results: list[_Result] = []
        started = time.perf_counter_ns()

        def checked_read(index: int) -> bytes:
            nonlocal peak_buffered, prefetched_bytes
            if cancel.is_set():
                raise RuntimeError("numeric tile pipeline cancelled")
            value = read_tile(index)
            if (not isinstance(value, bytes)
                    or not 1 <= len(value) <= MAX_NUMERIC_PIPELINE_TILE_BYTES):
                raise ValueError("numeric source tile is invalid")
            with buffer_lock:
                prefetched_bytes += len(value)
                peak_buffered = max(peak_buffered, prefetched_bytes)
            return value

        with self._ledger.reserve(self._bandwidth):
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="numeric-prefetch")
            future: Future[bytes] | None = None
            try:
                if self._buffer_count == 2:
                    future = executor.submit(checked_read, 0)
                for index in range(tile_count):
                    if cancel.is_set():
                        raise RuntimeError("numeric tile pipeline cancelled")
                    if future is None:
                        source = checked_read(index)
                    else:
                        source = future.result()
                        barriers += 1
                    next_future = None
                    if self._buffer_count == 2 and index + 1 < tile_count:
                        next_future = executor.submit(checked_read, index + 1)
                    converted = convert_tile(source)
                    if (not isinstance(converted, bytes)
                            or not 1 <= len(converted) <= MAX_NUMERIC_PIPELINE_TILE_BYTES):
                        raise ValueError("numeric converted tile is invalid")
                    with buffer_lock:
                        peak_buffered = max(peak_buffered, prefetched_bytes + len(converted))
                    source_bytes += len(source)
                    converted_bytes += len(converted)
                    digest.update(index.to_bytes(8, "little"))
                    digest.update(len(converted).to_bytes(8, "little"))
                    digest.update(converted)
                    results.append(consume_tile(index, converted))
                    with buffer_lock:
                        prefetched_bytes -= len(source)
                    future = next_future
            finally:
                if future is not None:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
        return NumericPipelineReport(
            tuple(results), tile_count, source_bytes, converted_bytes,
            peak_buffered, self._bandwidth,
            max(1, time.perf_counter_ns() - started), digest.hexdigest(), barriers,
        )
