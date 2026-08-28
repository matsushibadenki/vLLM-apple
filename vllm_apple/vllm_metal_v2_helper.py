from __future__ import annotations

import json
import sys
from collections.abc import Callable

from .vllm_metal_v2_adapter import (
    MAX_V2_MEASUREMENT_OUTPUT_BYTES,
    build_v2_measurement_request,
    parse_v2_measurement_request,
    parse_v2_measurement_response,
)

MAX_V2_MEASUREMENT_REQUEST_BYTES = 16 * 1024
NATIVE_MEASUREMENT_SYMBOL = "vllm_apple_measure_paged_attention_v2"


def invoke_native_measurement(
    payload: object,
    *,
    get_ops: Callable[[], object] | None = None,
) -> dict[str, object]:
    shape, configuration = parse_v2_measurement_request(payload)
    if get_ops is None:
        try:
            from vllm_metal.metal import get_ops as native_get_ops
        except ImportError as error:
            raise RuntimeError("vLLM-Metal native extension is unavailable") from error

        get_ops = native_get_ops
    ops = get_ops()
    native_measure = getattr(ops, NATIVE_MEASUREMENT_SYMBOL, None)
    if not callable(native_measure):
        raise RuntimeError(
            "vLLM-Metal does not expose the native v2 measurement ABI v1"
        )
    canonical_request = json.dumps(
        build_v2_measurement_request(shape, configuration),
        sort_keys=True,
        separators=(",", ":"),
    )
    raw_response = native_measure(canonical_request)
    if not isinstance(raw_response, str):
        raise RuntimeError("native v2 measurement ABI returned a non-string response")
    if not 1 <= len(raw_response.encode("utf-8")) <= MAX_V2_MEASUREMENT_OUTPUT_BYTES:
        raise RuntimeError("native v2 measurement ABI returned an oversized response")
    try:
        response = json.loads(raw_response)
        passed, latency, digest = parse_v2_measurement_response(response)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError("native v2 measurement ABI returned invalid JSON") from error
    return {
        "abi_version": 1,
        "passed": passed,
        "latency_nanoseconds": latency,
        "output_digest": digest,
    }


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_V2_MEASUREMENT_REQUEST_BYTES + 1)
    if not 1 <= len(raw) <= MAX_V2_MEASUREMENT_REQUEST_BYTES:
        print("native v2 measurement request is empty or oversized", file=sys.stderr)
        return 2
    try:
        payload = json.loads(raw)
        response = invoke_native_measurement(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    encoded = json.dumps(response, sort_keys=True, separators=(",", ":"))
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
