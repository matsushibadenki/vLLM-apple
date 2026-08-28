# Unified Memory telemetry

`GET /v1/runtime` exposes `memory_telemetry` as a source-aware, two-layer
snapshot. The ledger stores only the latest values and monotonic peaks, so its
memory use is constant.

## Layers

- OS: Unified Memory total/available bytes and separate control/backend RSS.
- Framework: allocator current/peak bytes and its peak delta from backend RSS.
- KV cache: used/capacity bytes and a derived usage ratio.

Framework and KV fields are `null` until a backend adapter submits a measured
sample through `RuntimeService.record_framework_memory` or
`RuntimeService.record_kv_cache_memory`. Unknown measurements are deliberately
not represented as zero. Every measured layer carries a `source` identifier so
estimated scheduler reservations cannot be confused with allocator telemetry.

The peak delta is signed because an allocator may account for shared or reserved
pages differently from RSS. Consumers must treat it as an accounting difference,
not as leaked memory.

## Unified budget ledger

`memory_budget` reconciles six fixed, constant-space categories: model weights,
KV state, reusable prefix state, scheduler scratch reservations, the Metal heap,
and Core ML buffers. Every category retains only its latest value, monotonic peak,
and source. Missing backend measurements remain `null` and are listed in
`unknown_components`; they are never silently converted to zero.

Weights, KV, prefix, scratch, and Core ML are additive components.
`known_component_bytes` and `known_remaining_bytes` include only those categories,
and `overcommitted_bytes` makes a known over-budget state explicit. Metal heap is
an `overlap_envelope`: MLX allocations, KV pages, and weights may already be inside
the IOGPU/Metal observation, so it is published separately and never added again.
IOGPU is preferred for that envelope, with the MLX allocator as a fallback.

The runtime automatically reconciles measured KV bytes, semantic-prefix residency,
active scheduler reservations, and the best Metal envelope on every snapshot.
Model loaders and Core ML adapters can submit their exact values through
`record_memory_budget_component`; until they do, those values remain unknown.
Core ML adapters must report exclusive runtime buffers rather than model weights,
which belong in the weights category. The Swift SDK exposes the same snapshot as
typed `MemoryBudget` values through both HTTP and Unix-domain-socket clients.

Managed model startup records the deduplicated size of local/Hugging Face cached
weight shards even when `--max-model-len` is explicitly supplied. Before each
inference reservation, admission refreshes the ledger and rejects a known existing
overcommit as `memory_budget_overcommitted`, or a request that would cross the
known component ceiling as `memory_budget_exceeded`. Unknown components remain
observable but do not cause false rejection of backends that cannot report them.

## vLLM adapter

The managed daemon polls the local backend `/metrics` endpoint once per second on
a dedicated daemon thread. Responses are limited to 1 MiB, 20,000 lines, and
4 KiB per line. Sampling failure preserves the last good value and never blocks
`GET /v1/runtime`.

The adapter accepts the current `vllm:kv_cache_usage_perc` gauge and the legacy
`vllm:gpu_cache_usage_perc` alias, preferring the current name. It also reads the
standard Prometheus `process_resident_memory_bytes` gauge. For verified vLLM
0.24.x–0.28.x backends and standard inspected models, one unambiguous
`vllm:cache_config_info` series may supply `kv_cache_size_tokens`. The adapter
multiplies that group-aware logical capacity by the inspected KV bytes/token and
converts the ratio to used bytes. Unknown versions, missing/`None` labels, multiple
config series, hybrid models that cannot be inspected, and values over 2 TiB retain
ratio-only telemetry; planned context budgets are never substituted.

When exact KV capacity becomes available after backend load, `context_reevaluation`
divides it by the inspected model's KV bytes/token and compares the result with the
configured context. A smaller capacity changes only the control-plane admission
ceiling; the running backend is never mutated in place. Requests above the effective
ceiling fail as `backend_context_capacity_exceeded`. Pending, sufficient, and reduced
states expose the configured/effective token counts, inspected weights footprint,
measurement source, and a constant-space reevaluation counter.
Each changed capacity publishes one `runtime.context_reevaluation` SSE event;
identical polling samples are coalesced. The Swift SDK exposes a typed event view,
and the Mac sample displays a localized warning when the effective limit is reduced.

The same in-backend middleware exposes a loopback-only JSON memory snapshot
using MLX `get_active_memory`, `get_cache_memory`, and `get_peak_memory`.
Allocator current bytes are active plus cache bytes. These framework counters
are intentionally kept separate from RSS and IOGPU because they may omit
command-buffer-resident or other Metal allocations.

On macOS the monitor also performs a best-effort bounded `ioreg` query for
`AGXAccelerator` and legacy `IOAccelerator` services. Known public registry
memory properties are parsed from the binary plist with a 1 MiB limit. Missing
services, sandbox restrictions, and OS versions that omit these properties leave
IOGPU fields `null` rather than failing inference.

## Memory pressure notifications

On macOS the daemon registers a native libdispatch
`DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` source for normal, warning, and critical
transitions. Duplicate levels are coalesced. Every transition updates the
runtime memory snapshot and publishes `memory.pressure`; when the semantic-state
elastic controller is enabled, the same callback requests its budget change
through the scheduler safe point. Active reservations therefore defer eviction
until completion.

The source owns one callback and no polling history. Registration failure leaves
the startup `vm_stat` pressure estimate in place and publishes an unavailable
monitor event instead of preventing daemon startup.
Control HTTP and UDS servers are created first; native source registration runs
on an independent daemon thread. A virtualized runner that stalls inside
libdispatch therefore cannot block runtime readiness.

## Admission gate

Before scheduler reservation, new work is evaluated against the native pressure
level, available-memory ratio, backend RSS, and IOGPU usage. RSS and IOGPU are
overlapping Unified Memory views, so the gate uses their maximum rather than
adding them.

- Warning: reject background work; retain interactive and normal inference.
- Critical: reject new normal/background work; retain a small interactive and
  realtime escape hatch for cancellation and recovery actions.
- Any level: reject estimates larger than currently available Unified Memory.

The derived warning thresholds are below 18% available or at least 82% observed;
critical thresholds are below 8% available or at least 92% observed. Native
libdispatch pressure can only raise the effective level. Existing reservations
are never cancelled. A return to normal clears the last rejection reason and
automatically reopens admission.

After a warning or critical state returns to normal, recovery uses three
monotonic-clock stages instead of reopening at full capacity:

| Elapsed | Max batch | Max measured/requested context | Max transient share |
| --- | ---: | ---: | ---: |
| 0–5 seconds | 1 | 4,096 tokens | 12.5% available |
| 5–15 seconds | 2 | 8,192 tokens | 25% available |
| 15–30 seconds | 4 | 32,768 tokens | 50% available |
| 30+ seconds | unrestricted by recovery policy | unrestricted | 100% |

For OpenAI requests, `n` supplies batch size and `max_completion_tokens` or
legacy `max_tokens` supplies the known lower bound of requested context. Backend
tokenizer adapters may provide a fuller prompt-plus-output estimate. A renewed
warning or critical notification cancels recovery immediately; a later normal
transition starts again from stage zero.

### Tokenizer-backed context estimate

For managed vLLM backends, the control plane sends a bounded chat-form request
to the official `/tokenize` endpoint before admission. It scans at most 64 KiB
of the response for the integer `count` and closes the response immediately;
the potentially large token-ID array is neither retained nor logged. Prompt
count plus requested completion tokens becomes `estimated_context_tokens`.

Only tokenizer-related fields are forwarded. Sampling parameters and generated
content are excluded. A timeout, unsupported chat tokenization, malformed
response, or missing backend falls back to the completion-only lower bound and
increments bounded counters in `token_estimation` rather than failing inference.

Successful counts are reused through a process-local cache with a hard limit of
256 entries, a five-minute TTL, and LRU eviction. The cache retains only a SHA-256
fingerprint of the canonical tokenizer request and its integer count; prompt text,
tool definitions, and token IDs are not retained. Runtime telemetry exposes cache
capacity, current entries, hits, misses, evictions, and expirations without exposing
request-derived keys.

Concurrent cache misses for the same fingerprint are coalesced with a bounded
single-flight coordinator. At most 64 distinct keys may be coordinated at once;
additional distinct keys bypass coordination instead of growing memory. Followers
wait at most 5.5 seconds and share either the leader's count or its failure. The
coordinator removes completed entries immediately and publishes only aggregate
active, leader, follower, bypass, and timeout counters.
