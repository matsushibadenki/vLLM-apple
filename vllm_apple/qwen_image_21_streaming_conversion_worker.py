from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from .generative_artifact_inspection import inspect_generative_artifact
from .model_integrity import build_model_integrity_manifest


COMPONENTS = ("transformer", "text_encoder")


def _convert_component(source: Path, staging: Path, component: str) -> None:
    import torch
    from torchao.quantization import Int8WeightOnlyConfig

    destination = staging / component
    if component == "transformer":
        from diffusers import QwenImage21Transformer2DModel, TorchAoConfig

        model = QwenImage21Transformer2DModel.from_pretrained(
            source / component,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=TorchAoConfig(Int8WeightOnlyConfig()),
        )
    elif component == "text_encoder":
        from transformers import Qwen3VLForConditionalGeneration, TorchAoConfig

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            source / component,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=TorchAoConfig(Int8WeightOnlyConfig()),
        )
    else:
        raise ValueError("unsupported Qwen-Image-2.1 conversion component")
    model.save_pretrained(destination, safe_serialization=True, max_shard_size="5GB")


def _run_component_process(source: Path, staging: Path, component: str) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "vllm_apple.qwen_image_21_streaming_conversion_worker",
            "--component",
            component,
            str(source),
            str(staging),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"INT8 conversion component failed: {component}")


def _copy_unquantized_files(source: Path, staging: Path) -> None:
    for entry in source.iterdir():
        if entry.name in {".git", *COMPONENTS}:
            continue
        if entry.is_symlink():
            raise ValueError("conversion source must not contain symlinks")
        destination = staging / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination)
        elif entry.is_file():
            shutil.copy2(entry, destination)


def convert_qwen_image_21_int8_streaming_atomic(
    source: str | Path,
    output: str | Path,
    *,
    component_runner: Callable[[Path, Path, str], None] = _run_component_process,
) -> dict[str, object]:
    source_root = Path(source).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().absolute()
    output_parent = output_path.parent.resolve(strict=True)
    source_report = inspect_generative_artifact(source_root)
    if source_report.get("pipeline_class") != "QwenImage21Pipeline":
        raise ValueError("conversion source must be QwenImage21Pipeline")
    if output_path.exists():
        raise ValueError("conversion output must not already exist")
    if output_parent == source_root or output_parent.is_relative_to(source_root):
        raise ValueError("conversion output must be outside the source artifact")

    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_parent))
    promoted = False
    completed_components: list[str] = []
    try:
        _copy_unquantized_files(source_root, staging)
        for component in COMPONENTS:
            component_runner(source_root, staging, component)
            completed_components.append(component)
        artifact = inspect_generative_artifact(staging)
        quantization = artifact.get("quantization")
        if artifact.get("pipeline_class") != "QwenImage21Pipeline":
            raise RuntimeError("converted artifact pipeline identity is invalid")
        if not isinstance(quantization, dict) or not (
            quantization.get("method") == "torchao"
            and quantization.get("bits") == 8
            and quantization.get("weight_only") is True
        ):
            raise RuntimeError("converted artifact is not TorchAO INT8 weight-only")
        if set(artifact.get("quantized_components", [])) != set(COMPONENTS):
            raise RuntimeError("converted artifact does not quantize both required components")
        integrity = build_model_integrity_manifest(staging)
        os.replace(staging, output_path)
        promoted = True
        return {
            "schema_version": 1,
            "candidate_id": "qwen-image-2.1",
            "strategy": "component-streaming-int8-weight-only",
            "completed_components": completed_components,
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
    parser.add_argument("--component", choices=COMPONENTS)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.component:
            _convert_component(arguments.source, arguments.output, arguments.component)
            return 0
        report = convert_qwen_image_21_int8_streaming_atomic(
            arguments.source, arguments.output
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "error_code": "qwen_image_21_streaming_conversion_failed",
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
