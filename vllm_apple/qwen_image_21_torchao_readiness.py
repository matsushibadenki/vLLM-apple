from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


MAX_PROBE_OUTPUT_BYTES = 16 * 1024


def inspect_qwen_image_21_torchao_readiness(executable: str | Path) -> dict[str, object]:
    path = Path(executable).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("Qwen-Image-2.1 Python executable is not executable")
    script = """
import importlib.metadata as metadata
import json
import torch
from diffusers import PipelineQuantizationConfig, TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig, quantize_

probe = torch.nn.Linear(32, 16, bias=False).eval()
quantize_(probe, Int8WeightOnlyConfig())
cpu_shape = list(probe(torch.randn(2, 32)).shape)
mps_built = torch.backends.mps.is_built()
mps_available = torch.backends.mps.is_available()
mps_int8_weight_only = False
mps_error = None
if mps_available:
    try:
        device_probe = probe.to("mps")
        device_shape = list(device_probe(torch.randn(2, 32, device="mps")).shape)
        mps_int8_weight_only = device_shape == [2, 16]
    except Exception as error:
        mps_error = f"{type(error).__name__}: {error}"[:1000]
print(json.dumps({
    "torch_version": metadata.version("torch"),
    "torchao_version": metadata.version("torchao"),
    "diffusers_version": metadata.version("diffusers"),
    "diffusers_pipeline_quantization_api": all((PipelineQuantizationConfig, TorchAoConfig)),
    "int8_weight_only_api": True,
    "cpu_int8_probe": cpu_shape == [2, 16],
    "mps_built": mps_built,
    "mps_available": mps_available,
    "mps_int8_weight_only_probe": mps_int8_weight_only,
    "mps_error": mps_error,
}))
"""
    try:
        completed = subprocess.run(
            [str(path), "-c", script],
            capture_output=True,
            check=True,
            text=True,
            timeout=30.0,
        )
        raw = completed.stdout.strip()
        if not 1 <= len(raw.encode("utf-8")) <= MAX_PROBE_OUTPUT_BYTES:
            raise ValueError("TorchAO readiness output is outside the bounded limit")
        payload = json.loads(raw)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ValueError("TorchAO readiness probe failed") from error
    required = {
        "torch_version",
        "torchao_version",
        "diffusers_version",
        "diffusers_pipeline_quantization_api",
        "int8_weight_only_api",
        "cpu_int8_probe",
        "mps_built",
        "mps_available",
        "mps_int8_weight_only_probe",
        "mps_error",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("TorchAO readiness output has an invalid schema")
    conversion_ready = all(
        payload[key]
        for key in (
            "diffusers_pipeline_quantization_api",
            "int8_weight_only_api",
            "cpu_int8_probe",
        )
    )
    mps_runtime_ready = conversion_ready and all(
        payload[key]
        for key in ("mps_built", "mps_available", "mps_int8_weight_only_probe")
    )
    return {
        "schema_version": 1,
        "backend": "torchao-int8-weight-only",
        "executable": str(path.resolve()),
        **payload,
        "conversion_ready": conversion_ready,
        "mps_runtime_ready": mps_runtime_ready,
        "loads_model_weights": False,
    }
