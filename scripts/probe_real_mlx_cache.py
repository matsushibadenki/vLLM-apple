"""Observe logical KV metadata on real MLX arrays; emit JSON without qualification."""
import json
import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from vllm_apple.mlx_server import bounded_cache_nbytes

cache = KVCache()
k = mx.zeros((1, 2, 16, 64), dtype=mx.float16)
v = mx.zeros((1, 2, 16, 64), dtype=mx.float16)
cache.update_and_fetch(k, v)
size, complete = bounded_cache_nbytes(cache)
active_state_bytes = sum(array.nbytes for array in cache.state)
expected = cache.keys.nbytes + cache.values.nbytes
alias = bounded_cache_nbytes([k, k])
view = k[:, :, :8, :]
view_count = bounded_cache_nbytes([k, view])
started = time.perf_counter_ns()
for _ in range(1000):
    bounded_cache_nbytes(cache)
duration = (time.perf_counter_ns() - started) / 1e6
report = dict(report_kind='real_mlx_cache_metadata_audit', cache_type='KVCache',
              dtype='float16', shape=[1, 2, 16, 64], measured_bytes=size,
              expected_allocated_logical_bytes=expected, active_state_logical_bytes=active_state_bytes,
              traversal_complete=complete,
              exact_object_alias_bytes=alias[0], parent_and_view_logical_bytes=view_count[0],
              iterations=1000, total_observation_ms=duration,
              shares_storage_deduplicated=False, physical_storage_bytes=None,
              qualification=False, inference_performance_measured=False)
assert size == expected and complete
assert alias[0] == k.nbytes
print(json.dumps(report, indent=2))
