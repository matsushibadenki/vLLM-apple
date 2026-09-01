from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

from .qwen4_reference import (
    qwen4_gated_residual_reference,
    qwen4_qsa_select_tokens_reference,
)


MAX_FIXTURE_OUTPUT_BYTES = 16 * 1024
FIXTURE_TOLERANCE = 2e-5


def expected_qwen4_fixture() -> dict[str, object]:
    gated = qwen4_gated_residual_reference(
        [3.0, 4.0, 0.0, 5.0],
        hc_count=2,
        hidden_size=2,
        norm_weight=[0.0] * 4,
        mix_down_weight=[[0.0] * 4],
        mix_up_weight=[[0.0], [0.0], [0.0], [0.0]],
        inject_weight=[[0.0] * 4, [0.0] * 4],
        eps=1e-6,
    )
    selected = qwen4_qsa_select_tokens_reference(
        [[0.0, 1.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 4.0], [8.0, 0.0]],
        [0, 1, 2, 3, 4],
        compress_ratio=2,
        token_budget=2,
        eps=1e-6,
    )
    return {
        "gated_mixed_input": gated["mixed_input"],
        "gated_injection_weights": gated["injection_weights"],
        "qsa_selected_tokens": selected,
    }


def assess_qwen4_mlx_fixture(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "mlx_version",
        "mlx_lm_version",
        "gated_mixed_input",
        "gated_injection_weights",
        "qsa_selected_tokens",
        "fixture_tensor_bytes",
    }:
        raise ValueError("Qwen4 MLX fixture response has an invalid schema")
    if payload["schema_version"] != 1:
        raise ValueError("Qwen4 MLX fixture schema version is unsupported")
    if not isinstance(payload["mlx_version"], str) or not isinstance(
        payload["mlx_lm_version"], str
    ):
        raise ValueError("Qwen4 MLX fixture versions are invalid")
    fixture_bytes = payload["fixture_tensor_bytes"]
    if (
        not isinstance(fixture_bytes, int)
        or isinstance(fixture_bytes, bool)
        or not 1 <= fixture_bytes <= 4096
    ):
        raise ValueError("Qwen4 MLX fixture tensor size is outside the bounded limit")
    expected = expected_qwen4_fixture()
    numeric_errors: dict[str, float] = {}
    for name in ("gated_mixed_input", "gated_injection_weights"):
        actual = payload[name]
        wanted = expected[name]
        if not isinstance(actual, list) or not isinstance(wanted, list) or len(actual) != len(wanted):
            raise ValueError(f"Qwen4 MLX fixture {name} shape is invalid")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in actual):
            raise ValueError(f"Qwen4 MLX fixture {name} values are invalid")
        error = max(abs(float(left) - float(right)) for left, right in zip(actual, wanted))
        if not math.isfinite(error):
            raise ValueError(f"Qwen4 MLX fixture {name} error is invalid")
        numeric_errors[name] = error
    selected = payload["qsa_selected_tokens"]
    if not isinstance(selected, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in selected
    ):
        raise ValueError("Qwen4 MLX fixture selected tokens are invalid")
    qsa_matches = selected == expected["qsa_selected_tokens"]
    passed = qsa_matches and all(error <= FIXTURE_TOLERANCE for error in numeric_errors.values())
    return {
        "schema_version": 1,
        "passed": passed,
        "mlx_version": payload["mlx_version"],
        "mlx_lm_version": payload["mlx_lm_version"],
        "fixture_tensor_bytes": fixture_bytes,
        "tolerance": FIXTURE_TOLERANCE,
        "numeric_errors": numeric_errors,
        "qsa_matches": qsa_matches,
        "stores_tensor_values": False,
        "measures_peak_memory": False,
    }


def run_qwen4_mlx_fixture(python_executable: str | Path) -> dict[str, object]:
    executable = Path(python_executable).expanduser().resolve()
    if (
        not executable.is_file()
        or not os.access(executable, os.X_OK)
        or executable.stat().st_uid != os.getuid()
    ):
        raise ValueError("Qwen4 MLX fixture Python is not a current-user executable regular file")
    try:
        result = subprocess.run(
            [str(executable), "-m", "vllm_apple.qwen4_mlx_fixture_worker"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Qwen4 MLX fixture worker failed") from error
    raw = result.stdout.strip()
    if not 1 <= len(raw.encode("utf-8")) <= MAX_FIXTURE_OUTPUT_BYTES:
        raise ValueError("Qwen4 MLX fixture output is outside the bounded limit")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Qwen4 MLX fixture output is not JSON") from error
    return assess_qwen4_mlx_fixture(payload)
