from __future__ import annotations

import hashlib
import importlib
import argparse
import ctypes
import ctypes.util
import json
import os
import platform
import resource
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .generative_collector import GenerationTelemetryEvent
from .generative_worker_protocol import consume_private_generative_request
from .hardware import detect_memory


MAX_GENERATED_ARTIFACT_BYTES = 16 * 1024**3
_IMAGE_PIPELINES = {
    "flux2-klein-9b-base": "Flux2KleinPipeline",
    "qwen-image-2512": "QwenImagePipeline",
    "flux2-dev": "Flux2Pipeline",
}


@dataclass(frozen=True, slots=True)
class WorkerTelemetry:
    process_rss_bytes: int
    memory_pressure: str
    thermal_state: str


@dataclass(frozen=True, slots=True)
class GeneratedImageArtifact:
    path: Path
    width: int
    height: int


class DiffusersImageRuntime(Protocol):
    pipeline_class: str

    def generate(
        self,
        request: Mapping[str, object],
        progress: Callable[[], None],
    ) -> GeneratedImageArtifact: ...


class LocalDiffusersImageRuntime:
    """Lazily imports a local-only Diffusers pipeline inside the isolated worker."""

    def __init__(self, candidate_id: str, *, module_loader=importlib.import_module) -> None:
        try:
            self.pipeline_class = _IMAGE_PIPELINES[candidate_id]
        except KeyError as error:
            raise ValueError("unsupported local Diffusers image candidate") from error
        self._candidate_id = candidate_id
        self._module_loader = module_loader

    def generate(
        self,
        request: Mapping[str, object],
        progress: Callable[[], None],
    ) -> GeneratedImageArtifact:
        if request.get("candidate_id") != self._candidate_id:
            raise ValueError("local Diffusers runtime candidate does not match its request")
        model_root = Path(str(request["model_root"])).resolve(strict=True)
        output_root = Path(str(request["output_root"])).resolve(strict=True)
        if not model_root.is_dir() or not output_root.is_dir():
            raise ValueError("local Diffusers model and output roots must be directories")
        torch = self._module_loader("torch")
        diffusers = self._module_loader("diffusers")
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("Diffusers image worker requires an available MPS device")
        pipeline_type = getattr(diffusers, self.pipeline_class, None)
        if pipeline_type is None:
            raise RuntimeError(f"Diffusers does not expose {self.pipeline_class}")
        pipeline = pipeline_type.from_pretrained(
            str(model_root),
            local_files_only=True,
            dtype=torch.bfloat16,
        )
        try:
            vae = getattr(pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            pipeline.to("mps")
            progress()
            generator = torch.Generator(device="cpu").manual_seed(request["seed"])

            def callback(_pipeline, _step, _timestep, callback_kwargs):
                progress()
                return callback_kwargs

            result = pipeline(
                prompt=request["prompt"],
                width=request["width"],
                height=request["height"],
                num_inference_steps=request["steps"],
                num_images_per_prompt=request["batch_size"],
                generator=generator,
                callback_on_step_end=callback,
            )
            images = getattr(result, "images", None)
            if not isinstance(images, (list, tuple)) or len(images) != request["batch_size"]:
                raise RuntimeError("Diffusers image pipeline returned an invalid image batch")
            if request["batch_size"] != 1:
                raise RuntimeError("qualification image worker requires batch size one")
            descriptor, temporary = tempfile.mkstemp(
                prefix=f"qualification-{request['sample_index']}-",
                suffix=".png",
                dir=output_root,
            )
            os.fchmod(descriptor, 0o600)
            os.close(descriptor)
            output = Path(temporary)
            try:
                images[0].save(output, format="PNG")
            except BaseException:
                try:
                    output.unlink()
                except FileNotFoundError:
                    pass
                raise
            return GeneratedImageArtifact(output, request["width"], request["height"])
        finally:
            del pipeline
            mps_empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
            if callable(mps_empty_cache):
                mps_empty_cache()


def _remove_output_if_owned(path: Path, output_root: Path) -> None:
    try:
        if path.is_symlink():
            return
        resolved = path.resolve(strict=True)
        info = resolved.stat()
        if (
            resolved != output_root
            and resolved.is_relative_to(output_root)
            and stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
        ):
            resolved.unlink()
    except FileNotFoundError:
        pass


def _hash_private_output(path: Path, output_root: Path) -> str:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ValueError("Diffusers worker output must not be a symlink")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_relative_to(output_root) or resolved == output_root:
        raise ValueError("Diffusers worker output must be inside the output root")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    info = None
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or not 1 <= info.st_size <= MAX_GENERATED_ARTIFACT_BYTES
        ):
            raise ValueError("Diffusers worker output is not a bounded current-user file")
        digest = hashlib.sha256()
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
        if remaining:
            raise ValueError("Diffusers worker output changed while being hashed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)
        try:
            current = resolved.lstat()
            if (
                info is not None
                and current.st_dev == info.st_dev
                and current.st_ino == info.st_ino
            ):
                resolved.unlink()
        except FileNotFoundError:
            pass


def execute_diffusers_image_request(
    request: Mapping[str, object],
    runtime: DiffusersImageRuntime,
    *,
    telemetry: Callable[[], WorkerTelemetry],
    emit: Callable[[GenerationTelemetryEvent], None],
    clock: Callable[[], float] = time.monotonic,
) -> None:
    _execute_image_request(
        request,
        runtime,
        expected_runtimes=_IMAGE_PIPELINES,
        backend_name="Diffusers",
        telemetry=telemetry,
        emit=emit,
        clock=clock,
    )


def execute_local_image_request(
    request: Mapping[str, object],
    runtime: DiffusersImageRuntime,
    *,
    expected_runtimes: Mapping[str, str],
    backend_name: str,
    telemetry: Callable[[], WorkerTelemetry],
    emit: Callable[[GenerationTelemetryEvent], None],
    clock: Callable[[], float] = time.monotonic,
) -> None:
    _execute_image_request(
        request,
        runtime,
        expected_runtimes=expected_runtimes,
        backend_name=backend_name,
        telemetry=telemetry,
        emit=emit,
        clock=clock,
    )


def _execute_image_request(
    request: Mapping[str, object],
    runtime: DiffusersImageRuntime,
    *,
    expected_runtimes: Mapping[str, str],
    backend_name: str,
    telemetry: Callable[[], WorkerTelemetry],
    emit: Callable[[GenerationTelemetryEvent], None],
    clock: Callable[[], float],
) -> None:
    candidate_id = request.get("candidate_id")
    expected_pipeline = expected_runtimes.get(candidate_id)
    if expected_pipeline is None or request.get("modality") != "image":
        raise ValueError(f"{backend_name} image worker does not support this candidate")
    if request.get("mode") != "text-to-image":
        raise ValueError(f"{backend_name} image worker currently supports text-to-image only")
    if runtime.pipeline_class != expected_pipeline:
        raise ValueError(f"{backend_name} runtime class does not match the candidate")
    output_root_value = request.get("output_root")
    if not isinstance(output_root_value, str):
        raise ValueError(f"{backend_name} worker output root is invalid")
    output_root = Path(output_root_value).resolve(strict=True)
    started = clock()

    def make_event(kind: str, *, output: GeneratedImageArtifact | None = None, digest=None):
        snapshot = telemetry()
        ceiling = request.get("memory_hard_ceiling_bytes")
        if isinstance(ceiling, int) and snapshot.process_rss_bytes > ceiling:
            raise MemoryError("Diffusers worker exceeded its memory hard ceiling")
        elapsed_ms = max(0.0, (clock() - started) * 1000.0)
        return GenerationTelemetryEvent(
            kind=kind,
            elapsed_ms=elapsed_ms,
            process_rss_bytes=snapshot.process_rss_bytes,
            memory_pressure=snapshot.memory_pressure,
            thermal_state=snapshot.thermal_state,
            output_width=output.width if output is not None else None,
            output_height=output.height if output is not None else None,
            output_frames=1 if output is not None else None,
            output_sha256=digest,
        )

    emit(make_event("started"))

    def progress() -> None:
        emit(make_event("progress"))

    output = runtime.generate(request, progress)
    if output.width != request.get("width") or output.height != request.get("height"):
        _remove_output_if_owned(output.path, output_root)
        raise ValueError("Diffusers worker output shape does not match the request")
    digest = _hash_private_output(output.path, output_root)
    emit(make_event("first_output"))
    emit(make_event("completed", output=output, digest=digest))


def _darwin_thermal_state() -> str:
    if platform.system() != "Darwin":
        return "unknown"
    objc_path = ctypes.util.find_library("objc")
    foundation_path = ctypes.util.find_library("Foundation")
    if objc_path is None or foundation_path is None:
        return "unknown"
    try:
        ctypes.CDLL(foundation_path)
        objc = ctypes.CDLL(objc_path)
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        message = objc.objc_msgSend
        message.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        message.restype = ctypes.c_void_p
        process_info = message(
            objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
        )
        message.restype = ctypes.c_long
        value = int(message(process_info, objc.sel_registerName(b"thermalState")))
    except (AttributeError, OSError, TypeError, ValueError):
        return "unknown"
    return {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}.get(value, "unknown")


def default_worker_telemetry() -> WorkerTelemetry:
    memory = detect_memory()
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(usage if platform.system() == "Darwin" else usage * 1024)
    return WorkerTelemetry(max(1, rss), memory.pressure.value, _darwin_thermal_state())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vllm-apple-diffusers-worker")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        request = consume_private_generative_request(
            arguments.request,
            workspace_root=arguments.workspace_root,
        )
        runtime = LocalDiffusersImageRuntime(request["candidate_id"])

        def emit(event: GenerationTelemetryEvent) -> None:
            print(json.dumps(asdict(event), sort_keys=True, separators=(",", ":")), flush=True)

        execute_diffusers_image_request(
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
