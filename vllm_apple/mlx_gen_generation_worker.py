from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import platform
import sys
import tempfile
from contextlib import redirect_stdout
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


MAX_BACKEND_EVENT_BYTES = 16 * 1024
MAX_BACKEND_EVENTS = 4096
_MLX_GEN_RUNTIME_CLASSES = {"flux2-klein-9b-base": "MLXGenFlux2KleinBase9B"}


class _BoundedProgressSink(io.TextIOBase):
    def __init__(self, progress: Callable[[], None]) -> None:
        self._progress = progress
        self._buffer = ""
        self._events = 0

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("MLX-Gen output must be text")
        self._buffer += value
        if len(self._buffer.encode("utf-8")) > MAX_BACKEND_EVENT_BYTES:
            raise RuntimeError("MLX-Gen emitted an oversized event")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume(line)
        return len(value)

    def flush(self) -> None:
        return None

    def finish(self) -> None:
        if self._buffer:
            self._consume(self._buffer)
            self._buffer = ""

    def _consume(self, line: str) -> None:
        if not line.strip():
            return
        self._events += 1
        if self._events > MAX_BACKEND_EVENTS:
            raise RuntimeError("MLX-Gen event count exceeded the bounded limit")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError("MLX-Gen emitted non-JSON output in JSON event mode") from error
        if not isinstance(event, dict):
            raise RuntimeError("MLX-Gen emitted an invalid JSON event")
        self._progress()


class LocalMLXGenImageRuntime:
    """Runs MLX-Gen in-process so RSS and memory-pressure evidence cover model execution."""

    pipeline_class = "MLXGenFlux2KleinBase9B"

    def __init__(self, *, module_loader=importlib.import_module) -> None:
        self._module_loader = module_loader

    def generate(
        self,
        request: Mapping[str, object],
        progress: Callable[[], None],
    ) -> GeneratedImageArtifact:
        if request.get("candidate_id") != "flux2-klein-9b-base":
            raise ValueError("MLX-Gen runtime only supports FLUX.2 Klein Base 9B")
        if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
            raise RuntimeError("MLX-Gen image worker requires Apple Silicon")
        model_root = Path(str(request["model_root"])).resolve(strict=True)
        output_root = Path(str(request["output_root"])).resolve(strict=True)
        if not model_root.is_dir() or not output_root.is_dir():
            raise ValueError("local MLX-Gen model and output roots must be directories")

        descriptor, temporary = tempfile.mkstemp(
            prefix=f"qualification-{request['sample_index']}-",
            suffix=".png",
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
            "--prompt",
            str(request["prompt"]),
            "--width",
            str(request["width"]),
            "--height",
            str(request["height"]),
            "--steps",
            str(request["steps"]),
            "--seed",
            str(request["seed"]),
            "--output",
            str(output),
            "--json-events",
            "--no-progress",
            "--low-ram",
        ]
        backend = self._module_loader("mflux.cli.mlx_gen")
        previous_argv = sys.argv
        sink = _BoundedProgressSink(progress)
        try:
            sys.argv = argv
            with redirect_stdout(sink):
                backend.main()
            sink.finish()
            if not output.is_file() or output.is_symlink():
                raise RuntimeError("MLX-Gen did not produce the requested local output")
            output.chmod(0o600)
            return GeneratedImageArtifact(
                output,
                int(request["width"]),
                int(request["height"]),
            )
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        finally:
            sys.argv = previous_argv


def execute_mlx_gen_image_request(
    request: Mapping[str, object],
    runtime: LocalMLXGenImageRuntime,
    *,
    telemetry,
    emit,
    clock=None,
) -> None:
    arguments = {
        "expected_runtimes": _MLX_GEN_RUNTIME_CLASSES,
        "backend_name": "MLX-Gen",
        "telemetry": telemetry,
        "emit": emit,
    }
    if clock is not None:
        arguments["clock"] = clock
    execute_local_image_request(request, runtime, **arguments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm-apple-mlx-gen-worker")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        request = consume_private_generative_request(
            arguments.request,
            workspace_root=arguments.workspace_root,
        )
        telemetry_stream = sys.stdout

        def emit(event: GenerationTelemetryEvent) -> None:
            print(
                json.dumps(asdict(event), sort_keys=True, separators=(",", ":")),
                file=telemetry_stream,
                flush=True,
            )

        execute_mlx_gen_image_request(
            request,
            LocalMLXGenImageRuntime(),
            telemetry=default_worker_telemetry,
            emit=emit,
        )
    except (ImportError, MemoryError, OSError, RuntimeError, SystemExit, ValueError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
