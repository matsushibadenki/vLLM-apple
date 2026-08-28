from __future__ import annotations

import json
import math
import os
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .vllm_metal_v2_tuning import (
    Measurement,
    V2DispatchConfiguration,
    V2PagedAttentionFamily,
    V2PagedAttentionShape,
)

VLLM_METAL_V2_MEASUREMENT_ABI_VERSION = 1
MAX_V2_MEASUREMENT_OUTPUT_BYTES = 16 * 1024


class V2MeasurementAdapterError(RuntimeError):
    pass


def build_v2_measurement_request(
    shape: V2PagedAttentionShape,
    configuration: V2DispatchConfiguration,
) -> dict[str, object]:
    """Build the strict JSON boundary consumed by a native vLLM-Metal helper."""
    return {
        "abi_version": VLLM_METAL_V2_MEASUREMENT_ABI_VERSION,
        "operation": "measure_paged_attention_v2",
        "shape": asdict(shape),
        "configuration": {
            **asdict(configuration),
            "family": configuration.family.value,
        },
    }


def parse_v2_measurement_request(
    payload: object,
) -> tuple[V2PagedAttentionShape, V2DispatchConfiguration]:
    expected = {"abi_version", "operation", "shape", "configuration"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("native v2 measurement request fields are invalid")
    if (
        payload["abi_version"] != VLLM_METAL_V2_MEASUREMENT_ABI_VERSION
        or payload["operation"] != "measure_paged_attention_v2"
    ):
        raise ValueError("native v2 measurement request ABI is unsupported")
    shape_value = payload["shape"]
    configuration_value = payload["configuration"]
    shape_fields = {
        "context_tokens",
        "query_tokens",
        "sequences",
        "query_heads",
        "kv_heads",
        "head_size",
        "block_size",
        "gpu_cores",
        "query_dtype",
        "cache_dtype",
        "turboquant",
        "window_seqlen_q",
    }
    configuration_fields = {
        "family",
        "threads",
        "tile_query",
        "tile_kv",
        "partition_size",
    }
    if not isinstance(shape_value, dict) or set(shape_value) != shape_fields:
        raise ValueError("native v2 measurement shape fields are invalid")
    if (
        not isinstance(configuration_value, dict)
        or set(configuration_value) != configuration_fields
    ):
        raise ValueError("native v2 measurement configuration fields are invalid")
    try:
        shape = V2PagedAttentionShape(**shape_value)
        configuration = V2DispatchConfiguration(
            family=V2PagedAttentionFamily(configuration_value["family"]),
            threads=configuration_value["threads"],
            tile_query=configuration_value["tile_query"],
            tile_kv=configuration_value["tile_kv"],
            partition_size=configuration_value["partition_size"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("native v2 measurement request values are invalid") from error
    return shape, configuration


def parse_v2_measurement_response(payload: object) -> Measurement:
    expected = {
        "abi_version",
        "passed",
        "latency_nanoseconds",
        "output_digest",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("native v2 measurement response fields are invalid")
    version = payload["abi_version"]
    passed = payload["passed"]
    latency = payload["latency_nanoseconds"]
    digest = payload["output_digest"]
    if version != VLLM_METAL_V2_MEASUREMENT_ABI_VERSION:
        raise ValueError("native v2 measurement ABI version mismatch")
    if (
        not isinstance(passed, bool)
        or isinstance(latency, bool)
        or not isinstance(latency, int)
        or latency <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("native v2 measurement values are invalid")
    return passed, latency, digest


@dataclass(frozen=True, slots=True)
class VLLMMetalV2MeasurementAdapter:
    """Isolated adapter for the vLLM-Metal native-v2 benchmark executable.

    The helper receives one JSON request on stdin and must emit one bounded JSON
    object on stdout. Kernel code remains outside the stable control process.
    """

    executable: Path
    timeout_seconds: float = 30.0
    maximum_output_bytes: int = MAX_V2_MEASUREMENT_OUTPUT_BYTES

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("native v2 measurement timeout must be positive and finite")
        if not 1024 <= self.maximum_output_bytes <= 64 * 1024:
            raise ValueError("native v2 measurement output limit must be 1 to 64 KiB")
        path = self.executable.expanduser().resolve()
        try:
            attributes = path.stat()
        except OSError as error:
            raise ValueError("native v2 measurement executable is unavailable") from error
        if (
            not stat.S_ISREG(attributes.st_mode)
            or attributes.st_uid != os.getuid()
            or not attributes.st_mode & stat.S_IXUSR
        ):
            raise ValueError(
                "native v2 measurement executable must be an executable current-user file"
            )
        object.__setattr__(self, "executable", path)

    def measure(
        self,
        shape: V2PagedAttentionShape,
        configuration: V2DispatchConfiguration,
    ) -> Measurement:
        request = json.dumps(
            build_v2_measurement_request(shape, configuration),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        with tempfile.TemporaryFile() as output:
            try:
                completed = subprocess.run(
                    [str(self.executable)],
                    input=request,
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise V2MeasurementAdapterError(
                    "native v2 measurement helper failed"
                ) from error
            size = output.tell()
            if completed.returncode != 0:
                raise V2MeasurementAdapterError(
                    f"native v2 measurement helper exited with {completed.returncode}"
                )
            if not 1 <= size <= self.maximum_output_bytes:
                raise V2MeasurementAdapterError("native v2 measurement response is oversized")
            output.seek(0)
            raw = output.read()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise V2MeasurementAdapterError(
                "native v2 measurement response is not valid JSON"
            ) from error
        try:
            return parse_v2_measurement_response(payload)
        except ValueError as error:
            raise V2MeasurementAdapterError(str(error)) from error
