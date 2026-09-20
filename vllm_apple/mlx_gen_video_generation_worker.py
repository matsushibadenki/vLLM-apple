from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping

from .diffusers_generation_worker import default_worker_telemetry
from .diffusers_video_generation_worker import (
    DEFAULT_VIDEO_FPS,
    GeneratedVideoArtifact,
    execute_local_video_request,
)
from .generative_collector import GenerationTelemetryEvent
from .generative_worker_protocol import consume_private_generative_request
from .mlx_gen_generation_worker import _BoundedProgressSink, _failure_code


_MLX_GEN_VIDEO_RUNTIMES = {"wan2.2-ti2v-5b": "MLXGenWanTI2V5B"}


class LocalMLXGenVideoRuntime:
    """Run MLX-Gen Wan T2V in-process for complete memory telemetry."""

    pipeline_class = "MLXGenWanTI2V5B"

    def __init__(self, *, module_loader=importlib.import_module) -> None:
        self._module_loader = module_loader

    def generate(
        self,
        request: Mapping[str, object],
        progress: Callable[[], None],
    ) -> GeneratedVideoArtifact:
        if request.get("candidate_id") != "wan2.2-ti2v-5b":
            raise ValueError("MLX-Gen video runtime candidate does not match its request")
        if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
            raise RuntimeError("MLX-Gen video worker requires Apple Silicon")
        model_root = Path(str(request["model_root"])).resolve(strict=True)
        output_root = Path(str(request["output_root"])).resolve(strict=True)
        if not model_root.is_dir() or not output_root.is_dir():
            raise ValueError("local MLX-Gen model and output roots must be directories")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f"qualification-{request['sample_index']}-",
            suffix=".mp4",
            dir=output_root,
        )
        os.fchmod(descriptor, 0o600)
        os.close(descriptor)
        output = Path(temporary)
        output.unlink()
        argv = [
            "mlxgen",
            "generate",
            "--model",
            str(model_root),
            "--family",
            "wan",
            "--task",
            "text-to-video",
            "--prompt",
            str(request["prompt"]),
            "--width",
            str(request["width"]),
            "--height",
            str(request["height"]),
            "--frames",
            str(request["frames"]),
            "--steps",
            str(request["steps"]),
            "--fps",
            str(DEFAULT_VIDEO_FPS),
            "--seed",
            str(request["seed"]),
            "--output",
            str(output),
            "--json-events",
            "--no-progress",
            "--low-ram",
            "--release-inactive-denoiser",
        ]
        backend = self._module_loader("mflux.cli.mlx_gen")
        previous_argv = sys.argv
        sink = _BoundedProgressSink(
            progress,
            ignored_prefixes=(
                "Saving video to:",
                "Saved video to:",
                "⚠️  Normalizing Wan q8 runtime-sensitive paths to BF16 at load:",
            ),
        )
        try:
            sys.argv = argv
            with redirect_stdout(sink):
                try:
                    backend.main()
                except SystemExit as error:
                    if error.code not in (None, 0):
                        raise
            sink.finish()
            if not output.is_file() or output.is_symlink():
                raise RuntimeError("MLX-Gen did not produce the requested local video")
            output.chmod(0o600)
            return GeneratedVideoArtifact(
                output,
                int(request["width"]),
                int(request["height"]),
                int(request["frames"]),
            )
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        finally:
            sys.argv = previous_argv


def execute_mlx_gen_video_request(request, runtime, *, telemetry, emit, clock=None) -> None:
    arguments = {
        "expected_runtimes": _MLX_GEN_VIDEO_RUNTIMES,
        "backend_name": "MLX-Gen",
        "telemetry": telemetry,
        "emit": emit,
        "enforce_memory_ceiling_during_generation": False,
    }
    if clock is not None:
        arguments["clock"] = clock
    execute_local_video_request(request, runtime, **arguments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm-apple-mlx-gen-video-worker")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        request = consume_private_generative_request(
            arguments.request, workspace_root=arguments.workspace_root
        )
        telemetry_stream = sys.stdout

        def emit(event: GenerationTelemetryEvent) -> None:
            print(
                json.dumps(asdict(event), sort_keys=True, separators=(",", ":")),
                file=telemetry_stream,
                flush=True,
            )

        execute_mlx_gen_video_request(
            request,
            LocalMLXGenVideoRuntime(),
            telemetry=default_worker_telemetry,
            emit=emit,
        )
    except (ImportError, MemoryError, OSError, RuntimeError, SystemExit, ValueError) as error:
        detail = str(error).replace("\n", " ").replace("\r", " ")[:512]
        print(
            "\n"
            + json.dumps(
                {
                    "vllm_apple_error_code": _failure_code(error),
                    "vllm_apple_error_detail": detail or type(error).__name__,
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
