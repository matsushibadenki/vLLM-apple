from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping

from .diffusers_generation_worker import (
    GeneratedImageArtifact,
    default_worker_telemetry,
    execute_local_image_request,
)
from .generative_collector import GenerationTelemetryEvent
from .generative_worker_protocol import consume_private_generative_request


_MFLUX_RUNTIME_CLASSES = {
    "z-image-turbo-mlx-4bit": "ZImageTurbo",
    "qwen-image-2512": "QwenImage",
}


class LocalMFluxImageRuntime:
    """Lazily imports MFLUX only inside an admitted, isolated worker process."""

    def __init__(self, candidate_id: str, *, module_loader=importlib.import_module) -> None:
        try:
            self.pipeline_class = _MFLUX_RUNTIME_CLASSES[candidate_id]
        except KeyError as error:
            raise ValueError("unsupported local MFLUX image candidate") from error
        self._candidate_id = candidate_id
        self._module_loader = module_loader

    def generate(
        self,
        request: Mapping[str, object],
        progress: Callable[[], None],
    ) -> GeneratedImageArtifact:
        if request.get("candidate_id") != self._candidate_id:
            raise ValueError("local MFLUX runtime candidate does not match its request")
        if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
            raise RuntimeError("MFLUX image worker requires Apple Silicon")
        model_root = Path(str(request["model_root"])).resolve(strict=True)
        output_root = Path(str(request["output_root"])).resolve(strict=True)
        if not model_root.is_dir() or not output_root.is_dir():
            raise ValueError("local MFLUX model and output roots must be directories")

        config_module = self._module_loader("mflux.models.common.config")
        model_config = config_module.ModelConfig
        if self._candidate_id == "z-image-turbo-mlx-4bit":
            model_module = self._module_loader("mflux.models.z_image")
            model_type = model_module.ZImageTurbo
            config = model_config.z_image_turbo()
        else:
            model_module = self._module_loader(
                "mflux.models.qwen.variants.txt2img.qwen_image"
            )
            model_type = model_module.QwenImage
            config = model_config.qwen_image()

        model = model_type(model_config=config, model_path=str(model_root))
        try:
            progress()
            image = model.generate_image(
                prompt=request["prompt"],
                seed=request["seed"],
                num_inference_steps=request["steps"],
                width=request["width"],
                height=request["height"],
            )
            descriptor, temporary = tempfile.mkstemp(
                prefix=f"qualification-{request['sample_index']}-",
                suffix=".png",
                dir=output_root,
            )
            os.fchmod(descriptor, 0o600)
            os.close(descriptor)
            output = Path(temporary)
            try:
                image.save(output)
            except BaseException:
                output.unlink(missing_ok=True)
                raise
            return GeneratedImageArtifact(output, int(request["width"]), int(request["height"]))
        finally:
            del model


def execute_mflux_image_request(
    request: Mapping[str, object],
    runtime: LocalMFluxImageRuntime,
    *,
    telemetry,
    emit,
    clock=None,
) -> None:
    arguments = {
        "expected_runtimes": _MFLUX_RUNTIME_CLASSES,
        "backend_name": "MFLUX",
        "telemetry": telemetry,
        "emit": emit,
    }
    if clock is not None:
        arguments["clock"] = clock
    execute_local_image_request(request, runtime, **arguments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm-apple-mflux-worker")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        request = consume_private_generative_request(
            arguments.request,
            workspace_root=arguments.workspace_root,
        )
        runtime = LocalMFluxImageRuntime(str(request["candidate_id"]))

        def emit(event: GenerationTelemetryEvent) -> None:
            print(json.dumps(asdict(event), sort_keys=True, separators=(",", ":")), flush=True)

        execute_mflux_image_request(
            request,
            runtime,
            telemetry=default_worker_telemetry,
            emit=emit,
        )
    except (ImportError, MemoryError, OSError, RuntimeError, ValueError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
