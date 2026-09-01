from __future__ import annotations

import json
import math
import os
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


QWEN4_CONVERSION_ABI_VERSION = 1
MAX_CONVERSION_REQUEST_BYTES = 16 * 1024
MAX_CONVERSION_RESPONSE_BYTES = 16 * 1024
_DTYPES = {"BF16", "F16", "F32"}
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def _digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_qwen4_conversion_request(
    *,
    stage_root: str | Path,
    tensor_name: str,
    contract_id: str,
    load_plan_id: str,
    target_dtype: str,
    maximum_artifact_bytes: int,
    memory_capacity_bytes: int,
    scratch_bytes: int = 0,
    axis0_slice: tuple[int, int] | None = None,
    requested_modes: tuple[str, ...] = ("text",),
) -> dict[str, object]:
    root = Path(stage_root).expanduser().resolve(strict=True)
    slice_payload = None
    if axis0_slice is not None:
        if (
            not isinstance(axis0_slice, tuple)
            or len(axis0_slice) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in axis0_slice)
            or axis0_slice[0] < 0
            or axis0_slice[1] <= 0
        ):
            raise ValueError("Qwen4 conversion axis-0 slice is invalid")
        slice_payload = {"start": axis0_slice[0], "count": axis0_slice[1]}
    payload = {
        "abi_version": QWEN4_CONVERSION_ABI_VERSION,
        "operation": "convert_qwen4_tensor",
        "stage_root": str(root),
        "tensor_name": tensor_name,
        "contract_id": contract_id,
        "load_plan_id": load_plan_id,
        "target_dtype": target_dtype,
        "maximum_artifact_bytes": maximum_artifact_bytes,
        "memory_capacity_bytes": memory_capacity_bytes,
        "scratch_bytes": scratch_bytes,
        "axis0_slice": slice_payload,
        "requested_modes": list(requested_modes),
    }
    parse_qwen4_conversion_request(payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > MAX_CONVERSION_REQUEST_BYTES:
        raise ValueError("Qwen4 conversion request exceeds the bounded limit")
    return payload


def parse_qwen4_conversion_request(payload: object) -> dict[str, object]:
    expected = {
        "abi_version",
        "operation",
        "stage_root",
        "tensor_name",
        "contract_id",
        "load_plan_id",
        "target_dtype",
        "maximum_artifact_bytes",
        "memory_capacity_bytes",
        "scratch_bytes",
        "axis0_slice",
        "requested_modes",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Qwen4 conversion request schema is invalid")
    stage_root = payload["stage_root"]
    tensor_name = payload["tensor_name"]
    axis0_slice = payload["axis0_slice"]
    requested_modes = payload["requested_modes"]
    if (
        payload["abi_version"] != QWEN4_CONVERSION_ABI_VERSION
        or payload["operation"] != "convert_qwen4_tensor"
        or not isinstance(stage_root, str)
        or not Path(stage_root).is_absolute()
        or not isinstance(tensor_name, str)
        or not 1 <= len(tensor_name.encode()) <= 1024
        or any(ord(character) < 0x20 for character in tensor_name)
        or not _digest(payload["contract_id"])
        or not _digest(payload["load_plan_id"])
        or payload["target_dtype"] not in _DTYPES
        or not isinstance(requested_modes, list)
        or not requested_modes
        or any(not isinstance(mode, str) for mode in requested_modes)
        or len(set(requested_modes)) != len(requested_modes)
        or any(mode not in {"text", "mtp", "vision"} for mode in requested_modes)
    ):
        raise ValueError("Qwen4 conversion request identity is invalid")
    for name in ("maximum_artifact_bytes", "memory_capacity_bytes"):
        value = payload[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("Qwen4 conversion request memory limit is invalid")
    scratch = payload["scratch_bytes"]
    if not isinstance(scratch, int) or isinstance(scratch, bool) or scratch < 0:
        raise ValueError("Qwen4 conversion request scratch limit is invalid")
    if axis0_slice is not None and (
        not isinstance(axis0_slice, dict)
        or set(axis0_slice) != {"start", "count"}
        or not isinstance(axis0_slice["start"], int)
        or isinstance(axis0_slice["start"], bool)
        or axis0_slice["start"] < 0
        or not isinstance(axis0_slice["count"], int)
        or isinstance(axis0_slice["count"], bool)
        or axis0_slice["count"] <= 0
    ):
        raise ValueError("Qwen4 conversion request axis-0 slice is invalid")
    return payload


def parse_qwen4_conversion_response(payload: object) -> dict[str, object]:
    expected = {
        "abi_version",
        "passed",
        "backend",
        "backend_version",
        "contract_id",
        "load_plan_id",
        "target_dtype",
        "output_shape",
        "output_bytes",
        "output_digest",
        "peak_reserved_bytes",
        "stores_tensor_values",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Qwen4 conversion response schema is invalid")
    shape = payload["output_shape"]
    if (
        payload["abi_version"] != QWEN4_CONVERSION_ABI_VERSION
        or payload["passed"] is not True
        or payload["backend"] not in {"mlx", "test"}
        or not isinstance(payload["backend_version"], str)
        or not 1 <= len(payload["backend_version"]) <= 128
        or not _digest(payload["contract_id"])
        or not _digest(payload["load_plan_id"])
        or payload["target_dtype"] not in _DTYPES
        or not isinstance(shape, list)
        or len(shape) > 16
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape)
        or not _digest(payload["output_digest"])
        or payload["stores_tensor_values"] is not False
    ):
        raise ValueError("Qwen4 conversion response identity is invalid")
    for name in ("output_bytes", "peak_reserved_bytes"):
        value = payload[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Qwen4 conversion response memory value is invalid")
    elements = 1
    for dimension in shape:
        elements *= dimension
        if elements * _DTYPE_BYTES[payload["target_dtype"]] > payload["peak_reserved_bytes"]:
            raise ValueError("Qwen4 conversion response shape exceeded its reservation")
    if payload["output_bytes"] != elements * _DTYPE_BYTES[payload["target_dtype"]]:
        raise ValueError("Qwen4 conversion response byte count does not match its shape")
    if payload["output_bytes"] > payload["peak_reserved_bytes"]:
        raise ValueError("Qwen4 conversion response exceeded its reservation")
    return payload


@dataclass(frozen=True, slots=True)
class Qwen4IsolatedConversionAdapter:
    executable: Path
    timeout_seconds: float = 120.0
    maximum_response_bytes: int = MAX_CONVERSION_RESPONSE_BYTES

    def __post_init__(self) -> None:
        unresolved = self.executable.expanduser()
        if unresolved.is_symlink():
            raise ValueError("Qwen4 conversion helper must not be a symlink")
        path = unresolved.resolve()
        info = path.stat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or not info.st_mode & stat.S_IXUSR
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise ValueError("Qwen4 conversion helper must be a current-user executable file")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Qwen4 conversion timeout is invalid")
        if not 1024 <= self.maximum_response_bytes <= 64 * 1024:
            raise ValueError("Qwen4 conversion response limit is invalid")
        object.__setattr__(self, "executable", path)

    def convert(self, request: object) -> dict[str, object]:
        parsed = parse_qwen4_conversion_request(request)
        encoded = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > MAX_CONVERSION_REQUEST_BYTES:
            raise ValueError("Qwen4 conversion request exceeds the bounded limit")
        with tempfile.TemporaryFile() as stdout:
            try:
                completed = subprocess.run(
                    [str(self.executable)],
                    input=encoded,
                    stdout=stdout,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=self.timeout_seconds,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError("Qwen4 conversion helper failed") from error
            size = stdout.tell()
            if completed.returncode != 0 or not 1 <= size <= self.maximum_response_bytes:
                raise ValueError("Qwen4 conversion helper returned an invalid bounded response")
            stdout.seek(0)
            raw = stdout.read(self.maximum_response_bytes + 1)
        try:
            response = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Qwen4 conversion helper response is not JSON") from error
        result = parse_qwen4_conversion_response(response)
        if (
            result["contract_id"] != parsed["contract_id"]
            or result["load_plan_id"] != parsed["load_plan_id"]
            or result["target_dtype"] != parsed["target_dtype"]
            or result["peak_reserved_bytes"] > parsed["memory_capacity_bytes"]
        ):
            raise ValueError("Qwen4 conversion response is not bound to its request")
        return result
