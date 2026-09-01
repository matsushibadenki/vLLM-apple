from __future__ import annotations

import json
import secrets
import threading
from collections import OrderedDict
from typing import Protocol


QWEN4_RUNTIME_ABI_VERSION = 1
MAX_RUNTIME_MESSAGE_BYTES = 16 * 1024
MAX_CACHED_RESPONSES = 256
_OPERATIONS = {"load", "unload", "status", "retry_quarantine", "shutdown"}
_DTYPES = {"BF16", "F16", "F32"}


class Qwen4RuntimeStore(Protocol):
    def load(
        self,
        tensor_name: str,
        *,
        target_dtype: str,
        scratch_bytes: int = 0,
        axis0_slice: tuple[int, int] | None = None,
    ) -> str: ...

    def unload(self, handle: str) -> None: ...

    def snapshot(self) -> dict[str, object]: ...

    def retry_quarantined_releases(self) -> int: ...

    def shutdown(self) -> dict[str, object]: ...


def create_qwen4_runtime_session_id() -> str:
    return secrets.token_hex(16)


def _identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def parse_qwen4_runtime_request(payload: object) -> dict[str, object]:
    common = {"abi_version", "session_id", "sequence", "request_id", "operation"}
    if not isinstance(payload, dict) or not common.issubset(payload):
        raise ValueError("Qwen4 runtime request schema is invalid")
    operation = payload["operation"]
    fields = {
        "load": common | {"tensor_name", "target_dtype", "scratch_bytes", "axis0_slice"},
        "unload": common | {"handle"},
        "status": common,
        "retry_quarantine": common,
        "shutdown": common,
    }
    if not isinstance(operation, str) or operation not in _OPERATIONS or set(payload) != fields[operation]:
        raise ValueError("Qwen4 runtime request operation schema is invalid")
    sequence = payload["sequence"]
    if (
        payload["abi_version"] != QWEN4_RUNTIME_ABI_VERSION
        or not _identifier(payload["session_id"])
        or not _identifier(payload["request_id"])
        or not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence <= 0
    ):
        raise ValueError("Qwen4 runtime request identity is invalid")
    if operation == "load":
        tensor_name = payload["tensor_name"]
        scratch = payload["scratch_bytes"]
        axis0_slice = payload["axis0_slice"]
        if (
            not isinstance(tensor_name, str)
            or not 1 <= len(tensor_name.encode()) <= 1024
            or any(ord(character) < 0x20 for character in tensor_name)
            or payload["target_dtype"] not in _DTYPES
            or not isinstance(scratch, int)
            or isinstance(scratch, bool)
            or scratch < 0
        ):
            raise ValueError("Qwen4 runtime load request is invalid")
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
            raise ValueError("Qwen4 runtime load slice is invalid")
    elif operation == "unload" and not _identifier(payload["handle"]):
        raise ValueError("Qwen4 runtime unload handle is invalid")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > MAX_RUNTIME_MESSAGE_BYTES:
        raise ValueError("Qwen4 runtime request exceeds the bounded limit")
    return payload


def parse_qwen4_runtime_response(payload: object) -> dict[str, object]:
    expected = {
        "abi_version",
        "session_id",
        "sequence",
        "request_id",
        "operation",
        "passed",
        "result",
        "error_code",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Qwen4 runtime response schema is invalid")
    if (
        payload["abi_version"] != QWEN4_RUNTIME_ABI_VERSION
        or not _identifier(payload["session_id"])
        or not _identifier(payload["request_id"])
        or not isinstance(payload["sequence"], int)
        or payload["sequence"] <= 0
        or payload["operation"] not in _OPERATIONS
        or not isinstance(payload["passed"], bool)
        or not isinstance(payload["result"], dict)
        or (payload["passed"] and payload["error_code"] is not None)
        or (not payload["passed"] and not isinstance(payload["error_code"], str))
    ):
        raise ValueError("Qwen4 runtime response identity is invalid")
    if len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()) > MAX_RUNTIME_MESSAGE_BYTES:
        raise ValueError("Qwen4 runtime response exceeds the bounded limit")
    result = payload["result"]
    if not payload["passed"]:
        if result:
            raise ValueError("Qwen4 failed runtime response must not contain a result")
        return payload
    operation = payload["operation"]
    if operation == "load" and (set(result) != {"handle"} or not _identifier(result["handle"])):
        raise ValueError("Qwen4 runtime load response is invalid")
    if operation == "unload" and result != {"unloaded": True}:
        raise ValueError("Qwen4 runtime unload response is invalid")
    if operation == "retry_quarantine" and (
        set(result) != {"released"}
        or not isinstance(result["released"], int)
        or isinstance(result["released"], bool)
        or result["released"] < 0
    ):
        raise ValueError("Qwen4 runtime quarantine response is invalid")
    if operation == "shutdown" and result != {"shutdown": True}:
        raise ValueError("Qwen4 runtime shutdown response is invalid")
    if operation == "status":
        status_fields = {
            "schema_version",
            "resident_tensors",
            "quarantined_tensors",
            "resident_components",
            "memory",
            "stores_tensor_names",
        }
        if (
            set(result) != status_fields
            or result["schema_version"] != 1
            or result["stores_tensor_names"] is not False
            or any(
                not isinstance(result[name], int)
                or isinstance(result[name], bool)
                or result[name] < 0
                for name in ("resident_tensors", "quarantined_tensors")
            )
            or not isinstance(result["resident_components"], dict)
            or not isinstance(result["memory"], dict)
        ):
            raise ValueError("Qwen4 runtime status response is invalid")
    return payload


class Qwen4RuntimeCommandService:
    def __init__(self, session_id: str, store: Qwen4RuntimeStore) -> None:
        if not _identifier(session_id):
            raise ValueError("Qwen4 runtime session ID is invalid")
        self.session_id = session_id
        self.store = store
        self._lock = threading.Lock()
        self._last_sequence = 0
        self._closed = False
        self._cache: OrderedDict[int, tuple[bytes, dict[str, object]]] = OrderedDict()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def handle(self, payload: object) -> dict[str, object]:
        request = parse_qwen4_runtime_request(payload)
        canonical = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        with self._lock:
            if request["session_id"] != self.session_id:
                raise ValueError("Qwen4 runtime request belongs to another session")
            sequence = request["sequence"]
            cached = self._cache.get(sequence)
            if cached is not None:
                if cached[0] != canonical:
                    raise ValueError("Qwen4 runtime sequence was reused with different content")
                return dict(cached[1])
            if sequence != self._last_sequence + 1:
                raise ValueError("Qwen4 runtime request sequence is not contiguous")
            if self._closed:
                raise ValueError("Qwen4 runtime session is closed")
            try:
                result = self._execute(request)
                passed = True
                error_code = None
            except MemoryError:
                result, passed, error_code = {}, False, "memory_admission_rejected"
            except (KeyError, OSError, RuntimeError, ValueError):
                result, passed, error_code = {}, False, "operation_failed"
            response = parse_qwen4_runtime_response(
                {
                    "abi_version": QWEN4_RUNTIME_ABI_VERSION,
                    "session_id": self.session_id,
                    "sequence": sequence,
                    "request_id": request["request_id"],
                    "operation": request["operation"],
                    "passed": passed,
                    "result": result,
                    "error_code": error_code,
                }
            )
            self._last_sequence = sequence
            self._cache[sequence] = (canonical, response)
            while len(self._cache) > MAX_CACHED_RESPONSES:
                self._cache.popitem(last=False)
            return dict(response)

    def _execute(self, request: dict[str, object]) -> dict[str, object]:
        operation = request["operation"]
        if operation == "load":
            slice_value = request["axis0_slice"]
            axis0_slice = (
                None
                if slice_value is None
                else (slice_value["start"], slice_value["count"])
            )
            handle = self.store.load(
                request["tensor_name"],
                target_dtype=request["target_dtype"],
                scratch_bytes=request["scratch_bytes"],
                axis0_slice=axis0_slice,
            )
            if not _identifier(handle):
                raise ValueError("Qwen4 runtime store returned an invalid handle")
            return {"handle": handle}
        if operation == "unload":
            self.store.unload(request["handle"])
            return {"unloaded": True}
        if operation == "status":
            return self.store.snapshot()
        if operation == "retry_quarantine":
            return {"released": self.store.retry_quarantined_releases()}
        snapshot = self.store.shutdown()
        if snapshot.get("resident_tensors") or snapshot.get("quarantined_tensors"):
            raise RuntimeError("Qwen4 runtime shutdown retained resources")
        self._closed = True
        return {"shutdown": True}
