"""Admission estimates for unified-memory processes, not an OS hard limit."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessMemoryBudget:
    maximum_bytes: int
    reserve_bytes: int

    def __post_init__(self):
        if (type(self.maximum_bytes) is not int or type(self.reserve_bytes) is not int
                or not 0 <= self.reserve_bytes < self.maximum_bytes):
            raise ValueError("invalid process memory budget")

    def resident_limit(self, *, rss_bytes, mlx_active_bytes, mlx_cache_bytes,
                       resident_bytes, available_bytes, requested_bytes):
        values = (rss_bytes, mlx_active_bytes, mlx_cache_bytes, resident_bytes,
                  available_bytes, requested_bytes)
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("invalid memory observation")
        # RSS and MLX allocations overlap in unified memory. Taking their maximum
        # avoids summing the same pages while retaining allocator cache in the estimate.
        observed = max(rss_bytes, mlx_active_bytes + mlx_cache_bytes)
        remaining = min(self.maximum_bytes - observed - self.reserve_bytes,
                        available_bytes - self.reserve_bytes)
        if remaining < 0:
            raise MemoryError("Strata process memory admission exhausted")
        return min(requested_bytes, resident_bytes + remaining)


def observe_process_memory():
    import mlx.core as mx
    import psutil
    return dict(rss_bytes=int(psutil.Process().memory_info().rss),
                mlx_active_bytes=int(mx.get_active_memory()),
                mlx_cache_bytes=int(mx.get_cache_memory()),
                available_bytes=int(psutil.virtual_memory().available))
