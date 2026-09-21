from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

from .generative_artifact_inspection import inspect_generative_artifact
from .model_integrity import build_model_integrity_manifest


def _convert_with_torchao(source: Path, staging: Path) -> None:
    import torch
    from diffusers import PipelineQuantizationConfig, QwenImage21Pipeline, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    quantization = PipelineQuantizationConfig(
        quant_mapping={
            "transformer": TorchAoConfig(Int8WeightOnlyConfig()),
            "text_encoder": TorchAoConfig(Int8WeightOnlyConfig()),
        }
    )
    pipeline = QwenImage21Pipeline.from_pretrained(
        str(source),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=quantization,
    )
    try:
        pipeline.save_pretrained(
            staging,
            safe_serialization=True,
            max_shard_size="5GB",
        )
    finally:
        del pipeline
        gc.collect()


def convert_qwen_image_21_int8_atomic(
    source: str | Path,
    output: str | Path,
    *,
    converter: Callable[[Path, Path], None] = _convert_with_torchao,
) -> dict[str, object]:
    source_root = Path(source).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().absolute()
    output_parent = output_path.parent.resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("conversion source must be a directory")
    if output_path.exists():
        raise ValueError("conversion output must not already exist")
    if output_parent == source_root or output_parent.is_relative_to(source_root):
        raise ValueError("conversion output must be outside the source artifact")

    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_parent))
    promoted = False
    try:
        converter(source_root, staging)
        artifact = inspect_generative_artifact(staging)
        if artifact.get("pipeline_class") != "QwenImage21Pipeline":
            raise RuntimeError("converted artifact pipeline identity is invalid")
        quantization = artifact.get("quantization")
        if not isinstance(quantization, dict) or not (
            quantization.get("method") == "torchao"
            and quantization.get("bits") == 8
            and quantization.get("weight_only") is True
        ):
            raise RuntimeError("converted artifact is not TorchAO INT8 weight-only")
        if set(artifact.get("quantized_components", [])) != {"transformer", "text_encoder"}:
            raise RuntimeError("converted artifact does not quantize both required components")
        integrity = build_model_integrity_manifest(staging)
        os.replace(staging, output_path)
        promoted = True
        return {
            "schema_version": 1,
            "candidate_id": "qwen-image-2.1",
            "output": str(output_path),
            "artifact_bytes": artifact["artifact_bytes"],
            "file_count": artifact["file_count"],
            "quantization": quantization,
            "quantized_components": artifact["quantized_components"],
            "artifact_root_sha256": integrity["root_sha256"],
            "atomic_promotion": True,
            "passed": True,
        }
    finally:
        if not promoted:
            shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = convert_qwen_image_21_int8_atomic(arguments.source, arguments.output)
    except (OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "error_code": "qwen_image_21_int8_conversion_failed",
                    "detail": str(error)[:512],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
