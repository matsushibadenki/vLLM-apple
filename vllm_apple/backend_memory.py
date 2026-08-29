from __future__ import annotations

import math
import json
import plistlib
import re
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass

MAX_METRICS_BYTES = 1024 * 1024
MAX_METRIC_LINES = 20_000
MAX_METRIC_LINE_BYTES = 4096
_KV_METRICS = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
_CACHE_CONFIG_METRIC = "vllm:cache_config_info"
MAX_KV_CAPACITY_BYTES = 2 * 1024**4
_STANDARD_KV_MODEL_TYPES = frozenset(
    {"gemma", "gemma2", "gpt2", "gpt_neox", "llama", "mistral", "mixtral", "phi3", "qwen2", "qwen3"}
)


@dataclass(frozen=True, slots=True)
class BackendMemorySample:
    resident_bytes: int | None
    kv_usage_ratio: float | None
    allocator_current_bytes: int | None = None
    allocator_peak_bytes: int | None = None
    kv_used_bytes: int | None = None
    kv_capacity_bytes: int | None = None
    source: str = "vllm-prometheus"


@dataclass(frozen=True, slots=True)
class KVCacheCapacityResolver:
    """Version-gated conversion of vLLM logical KV token capacity to bytes."""

    vllm_version: str
    kv_bytes_per_token: int
    model_type: str

    def __post_init__(self) -> None:
        match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", self.vllm_version)
        if match is None:
            raise ValueError("vLLM version is not parseable")
        version = tuple(int(value or 0) for value in match.groups())
        if not (0, 24, 0) <= version < (0, 29, 0):
            raise ValueError("vLLM version does not have a verified KV capacity contract")
        if not 0 < self.kv_bytes_per_token <= MAX_KV_CAPACITY_BYTES:
            raise ValueError("invalid KV bytes-per-token")
        if self.model_type not in _STANDARD_KV_MODEL_TYPES:
            raise ValueError("model architecture has no verified KV byte conversion")

    def resolve(self, identifiers: list[bytes]) -> int | None:
        if len(identifiers) != 1:
            return None
        match = re.search(rb'(?:\{|,)kv_cache_size_tokens="([1-9]\d*)"(?:,|})', identifiers[0])
        if match is None:
            return None
        capacity = int(match.group(1)) * self.kv_bytes_per_token
        return capacity if capacity <= MAX_KV_CAPACITY_BYTES else None


def parse_prometheus_memory_metrics(
    payload: bytes,
    kv_capacity_resolver: KVCacheCapacityResolver | None = None,
) -> BackendMemorySample:
    if len(payload) > MAX_METRICS_BYTES:
        raise ValueError("backend metrics payload is too large")
    resident_values: list[float] = []
    kv_values: dict[str, list[float]] = {name: [] for name in _KV_METRICS}
    cache_config_identifiers: list[bytes] = []
    for index, raw_line in enumerate(payload.splitlines()):
        if index >= MAX_METRIC_LINES:
            raise ValueError("backend metrics has too many lines")
        if len(raw_line) > MAX_METRIC_LINE_BYTES:
            raise ValueError("backend metric line is too large")
        line = raw_line.strip()
        if not line or line.startswith(b"#"):
            continue
        try:
            identifier, raw_value = line.rsplit(None, 1)
            name = identifier.split(b"{", 1)[0].decode("ascii")
            value = float(raw_value)
        except (UnicodeDecodeError, ValueError):
            continue
        if not math.isfinite(value) or value < 0:
            continue
        if name == "process_resident_memory_bytes":
            resident_values.append(value)
        elif name in kv_values and value <= 1:
            kv_values[name].append(value)
        elif name == _CACHE_CONFIG_METRIC and value == 1:
            cache_config_identifiers.append(identifier)
    # Prefer the current v1 name; use the deprecated v0 alias only when needed.
    ratios = next((kv_values[name] for name in _KV_METRICS if kv_values[name]), [])
    ratio = max(ratios) if ratios else None
    capacity = (
        kv_capacity_resolver.resolve(cache_config_identifiers)
        if kv_capacity_resolver is not None
        else None
    )
    used = min(capacity, round(capacity * ratio)) if capacity is not None and ratio is not None else None
    return BackendMemorySample(
        resident_bytes=int(max(resident_values)) if resident_values else None,
        kv_usage_ratio=ratio,
        kv_used_bytes=used,
        kv_capacity_bytes=capacity if used is not None else None,
    )


class VLLMMemoryMetricsAdapter:
    def __init__(
        self,
        base_url: str,
        timeout: float = 1.0,
        kv_capacity_resolver: KVCacheCapacityResolver | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("metrics timeout must be positive")
        self._url = base_url.rstrip("/") + "/metrics"
        self._timeout = timeout
        self._kv_capacity_resolver = kv_capacity_resolver

    def sample(self) -> BackendMemorySample:
        request = urllib.request.Request(self._url, headers={"Accept": "text/plain"})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = response.read(MAX_METRICS_BYTES + 1)
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError("backend memory metrics are unavailable") from error
        prometheus = parse_prometheus_memory_metrics(payload, self._kv_capacity_resolver)
        allocator_current = None
        allocator_peak = None
        try:
            with urllib.request.urlopen(
                self._url.removesuffix("/metrics") + "/v1/vllm-apple/memory",
                timeout=self._timeout,
            ) as response:
                mlx_payload = response.read(4097)
            if len(mlx_payload) <= 4096:
                decoded = json.loads(mlx_payload)
                active = decoded.get("active_bytes")
                cache = decoded.get("cache_bytes")
                peak = decoded.get("peak_bytes")
                if all(isinstance(value, int) and value >= 0 for value in (active, cache, peak)):
                    allocator_current = active + cache
                    allocator_peak = max(peak, allocator_current)
        except (OSError, urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        return BackendMemorySample(
            resident_bytes=prometheus.resident_bytes,
            kv_usage_ratio=prometheus.kv_usage_ratio,
            allocator_current_bytes=allocator_current,
            allocator_peak_bytes=allocator_peak,
            kv_used_bytes=prometheus.kv_used_bytes,
            kv_capacity_bytes=prometheus.kv_capacity_bytes,
        )


class MLXMemoryMetricsAdapter:
    """Bounded reader for the backend-local MLX telemetry wrapper."""

    def __init__(self, base_url: str, timeout: float = 1.0) -> None:
        if timeout <= 0:
            raise ValueError("metrics timeout must be positive")
        self._url = base_url.rstrip("/") + "/v1/vllm-apple/memory"
        self._timeout = timeout

    def sample(self) -> BackendMemorySample:
        try:
            with urllib.request.urlopen(self._url, timeout=self._timeout) as response:
                raw = response.read(4097)
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError("MLX memory metrics are unavailable") from error
        if len(raw) > 4096:
            raise RuntimeError("MLX memory metrics are oversized")
        try:
            payload = json.loads(raw)
            active = payload["active_bytes"]
            cache = payload["cache_bytes"]
            peak = payload["peak_bytes"]
            kv_used = payload["kv_cache_bytes"]
            complete = payload["traversal_complete"]
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("MLX memory metrics are invalid") from error
        values = (active, cache, peak, kv_used)
        if (
            not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values)
            or complete is not True
        ):
            raise RuntimeError("MLX memory metrics are incomplete")
        current = active + cache
        return BackendMemorySample(
            resident_bytes=None,
            kv_usage_ratio=None,
            allocator_current_bytes=current,
            allocator_peak_bytes=max(peak, current),
            kv_used_bytes=kv_used,
            source="mlx-wrapper-v1",
        )


class IOGPUMemoryAdapter:
    """Best-effort parser for public ioreg registry properties on macOS."""

    _KEYS = ("In use system memory", "Alloc system memory", "gartUsedBytes", "vramUsedBytes")

    @classmethod
    def parse(cls, payload: bytes) -> int | None:
        if len(payload) > MAX_METRICS_BYTES:
            raise ValueError("IOGPU registry payload is too large")
        decoded = plistlib.loads(payload)
        values: list[int] = []

        def visit(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in cls._KEYS and isinstance(child, int) and not isinstance(child, bool):
                        values.append(child)
                    else:
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(decoded)
        return max(values) if values else None

    def sample(self) -> int | None:
        for service_class in ("AGXAccelerator", "IOAccelerator"):
            try:
                completed = subprocess.run(
                    ["/usr/sbin/ioreg", "-r", "-c", service_class, "-a"],
                    capture_output=True,
                    check=False,
                    timeout=1.0,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if completed.returncode == 0 and completed.stdout:
                return self.parse(completed.stdout[: MAX_METRICS_BYTES + 1])
        return None


class MemoryMetricsMonitor:
    """One bounded polling thread; failures never replace the last good sample."""

    def __init__(
        self,
        adapter: VLLMMemoryMetricsAdapter,
        sink: object,
        interval: float = 1.0,
        iogpu_adapter: IOGPUMemoryAdapter | None = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("metrics interval must be positive")
        self._adapter = adapter
        self._sink = sink
        self._interval = interval
        self._iogpu_adapter = iogpu_adapter
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> BackendMemorySample:
        sample = self._adapter.sample()
        if sample.resident_bytes is not None:
            self._sink.record_backend_resident_memory(
                sample.resident_bytes, source=sample.source
            )
        if sample.kv_used_bytes is not None and sample.kv_capacity_bytes is not None:
            self._sink.record_kv_cache_memory(
                sample.kv_used_bytes,
                sample.kv_capacity_bytes,
                source="vllm-prometheus-cache-config-v1",
            )
        elif sample.kv_usage_ratio is not None:
            self._sink.record_kv_cache_ratio(sample.kv_usage_ratio, source=sample.source)
        if sample.allocator_current_bytes is not None:
            self._sink.record_framework_memory(
                sample.allocator_current_bytes,
                peak_bytes=sample.allocator_peak_bytes,
                source="mlx",
            )
        if self._iogpu_adapter is not None:
            iogpu_bytes = self._iogpu_adapter.sample()
            if iogpu_bytes is not None:
                self._sink.record_iogpu_memory(iogpu_bytes, source="ioreg")
        return sample

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("memory metrics monitor was already started")
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vllm-apple-memory-metrics"
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except (OSError, RuntimeError, ValueError):
                pass
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval + 0.25))
