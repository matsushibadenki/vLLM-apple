from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compat import inspect_backend, inspect_mlx_lm_backend
from .context import recommend_context
from .daemon import serve
from .execution_profile import detect_apple_chip_profile, save_chip_profile
from .hardware import detect_hardware
from .kernel_probe import build_environment_fingerprint
from .kernel_profile import build_model_kernel_shape_profile
from .kv_calibration import (
    default_calibration_report_path,
    discover_latest_kv_calibration,
    load_kv_calibration,
)
from .long_context import (
    LongContextEvaluationError,
    LongContextEvaluator,
    save_long_context_report,
)
from .long_context_backend import MLXLongContextAdapter, VLLMLongContextAdapter
from .metal_probe import NativeMetalProbeAdapter
from .metal_tuning import save_metal_tuning_report, tune_metal_shape_profile
from .model import ModelInspectionError, inspect_model
from .model_integrity import (
    ModelIntegrityError,
    build_model_integrity_manifest,
    save_model_integrity_manifest,
    verify_model_integrity,
)
from .phase_probe import PhaseProbeConfig, PhaseProbeError, run_phase_probe
from .profile import build_profile, save_profile
from .qualification import (
    QualificationConfig,
    default_qualification_report_path,
    qualify_model,
    save_qualification_report,
)
from .qualification_preflight import run_qualification_preflight
from .runtime_probe import discover_runtime_versions
from .shape_benchmark import (
    default_metal_shape_benchmark_path,
    run_metal_shape_benchmark,
    save_metal_shape_benchmark,
)
from .soak import _read_private_token
from .types import GIB, ModelMemorySpec
from .vllm_metal_integration import inspect_vllm_metal_integration
from .vllm_metal_v2_adapter import V2MeasurementAdapterError, VLLMMetalV2MeasurementAdapter
from .vllm_metal_v2_tuning import save_v2_tuning_profile, tune_v2_model_profile


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-apple")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("hardware", help="detect Apple hardware and memory")

    doctor = commands.add_parser("doctor", help="inspect the vLLM-Metal environment")
    doctor.add_argument("--backend-executable")

    integrity_create = commands.add_parser(
        "model-integrity-create", help="create a streaming SHA-256 model manifest"
    )
    integrity_create.add_argument("model", type=Path)
    integrity_create.add_argument("--output", required=True, type=Path)
    integrity_verify = commands.add_parser(
        "model-integrity-verify", help="verify a model against a trusted manifest"
    )
    integrity_verify.add_argument("model", type=Path)
    integrity_verify.add_argument("--manifest", required=True, type=Path)

    profile = commands.add_parser("profile", help="build a runtime profile")
    profile.add_argument("--save", action="store_true")
    profile.add_argument("--output")

    execution_profile = commands.add_parser(
        "execution-profile", help="detect execution backend capabilities"
    )
    execution_profile.add_argument("--backend-executable")
    execution_profile.add_argument("--save", action="store_true")
    execution_profile.add_argument("--output")

    kernel_shape = commands.add_parser(
        "kernel-shape-profile",
        help="derive bounded kernel shapes from local model metadata",
    )
    kernel_shape.add_argument("model")
    kernel_shape.add_argument("--contexts", default="128,1024,4096,16384")
    kernel_shape.add_argument("--block-tokens", type=int, default=16)

    metal_benchmark = commands.add_parser(
        "metal-shape-benchmark",
        help="measure model-backed Paged Attention shapes on the local Metal device",
    )
    metal_benchmark.add_argument("model")
    metal_benchmark.add_argument("--contexts", default="128,1024")
    metal_benchmark.add_argument("--block-tokens", type=int, default=16)
    metal_benchmark.add_argument("--samples", type=int, default=1)
    metal_benchmark.add_argument("--maximum-shapes", type=int, default=4)
    metal_benchmark.add_argument("--timeout", type=float, default=60)
    metal_benchmark.add_argument("--backend-version")
    benchmark_output = metal_benchmark.add_mutually_exclusive_group()
    benchmark_output.add_argument("--output", type=Path)
    benchmark_output.add_argument("--stdout", action="store_true")

    metal_tune = commands.add_parser(
        "metal-shape-tune",
        help="autotune model-backed Metal thread widths and persist the winners",
    )
    metal_tune.add_argument("model")
    metal_tune.add_argument("--contexts", default="128,1024")
    metal_tune.add_argument("--block-tokens", type=int, default=16)
    metal_tune.add_argument("--samples", type=int, default=3)
    metal_tune.add_argument("--maximum-shapes", type=int, default=4)
    metal_tune.add_argument("--timeout", type=float, default=60)
    metal_tune.add_argument("--backend-version")
    tuning_output = metal_tune.add_mutually_exclusive_group()
    tuning_output.add_argument("--output", type=Path)
    tuning_output.add_argument("--stdout", action="store_true")

    integration = commands.add_parser(
        "vllm-metal-integration-inspect",
        help="inspect a vLLM-Metal source tree for the thread tuning ABI",
    )
    integration.add_argument("source_root", type=Path)

    v2_tune = commands.add_parser(
        "vllm-metal-v2-tune",
        help="measure native-v2 kernel families and persist a device-bound profile",
    )
    v2_tune.add_argument("model")
    v2_tune.add_argument("--source-root", required=True, type=Path)
    v2_tune.add_argument("--helper", required=True, type=Path)
    v2_tune.add_argument("--contexts", default="1024,4096")
    v2_tune.add_argument("--block-tokens", type=int, default=16)
    v2_tune.add_argument("--prefill-query-tokens", type=int, default=128)
    v2_tune.add_argument("--samples", type=int, default=3)
    v2_tune.add_argument("--maximum-shapes", type=int, default=4)
    v2_tune.add_argument("--timeout", type=float, default=30)
    v2_tune.add_argument("--disable-nax", action="store_true")
    v2_output = v2_tune.add_mutually_exclusive_group()
    v2_output.add_argument("--output", type=Path)
    v2_output.add_argument("--stdout", action="store_true")

    v2_capability = commands.add_parser(
        "vllm-metal-v2-capability",
        help="probe the native-v2 measurement symbol without running a benchmark",
    )
    v2_capability.add_argument("--helper", required=True, type=Path)
    v2_capability.add_argument("--timeout", type=float, default=10)

    phase_profile = commands.add_parser(
        "phase-profile", help="measure prefill and decode phases through a streaming backend"
    )
    phase_profile.add_argument("--url", default="http://127.0.0.1:8000")
    phase_profile.add_argument("--model", required=True)
    phase_profile.add_argument("--hardware-fingerprint")
    phase_profile.add_argument("--backend", default="vllm_metal")
    phase_profile.add_argument("--samples", type=int, default=3)
    phase_profile.add_argument("--max-tokens", type=int, default=32)
    phase_profile.add_argument("--timeout", type=float, default=300)
    phase_profile.add_argument("--session-token-file", type=Path)
    phase_profile.add_argument("--pid", type=int)
    phase_profile.add_argument("--allow-remote", action="store_true")

    long_context = commands.add_parser(
        "long-context-evaluate",
        help="run tokenizer-aligned retrieval stages against a streaming vLLM backend",
    )
    long_context.add_argument("--url", default="http://127.0.0.1:8001")
    long_context.add_argument("--model", required=True)
    long_context.add_argument("--stages", default="1024,4096,16384")
    long_context.add_argument("--memory-ceiling-gb", type=float, required=True)
    long_context.add_argument("--state-bytes-per-token", type=int, required=True)
    long_context.add_argument("--load-peak-rss-bytes", type=int, default=0)
    long_context.add_argument("--model-max-context", type=int)
    long_context.add_argument("--hardware-fingerprint")
    long_context.add_argument(
        "--backend", choices=("vllm_metal", "mlx_lm"), default="vllm_metal"
    )
    long_context.add_argument("--max-tokens", type=int, default=32)
    long_context.add_argument("--timeout", type=float, default=300)
    long_context.add_argument("--session-token-file", type=Path)
    long_context.add_argument("--pid", type=int)
    long_context.add_argument("--allow-remote", action="store_true")
    long_context_output = long_context.add_mutually_exclusive_group()
    long_context_output.add_argument("--output", type=Path)
    long_context_output.add_argument("--save", action="store_true")
    long_context.add_argument("--application-support", type=Path)

    qualify = commands.add_parser(
        "qualify-model", help="launch and stability-test a local model backend"
    )
    qualify.add_argument("model")
    qualify.add_argument("--backend-executable", required=True, type=Path)
    qualify.add_argument(
        "--backend-kind", choices=("vllm_metal", "mlx_lm"), default="vllm_metal"
    )
    qualify.add_argument("--backend-port", type=int, default=8001)
    qualify.add_argument("--max-model-len", type=int)
    qualify.add_argument("--startup-timeout", type=float, default=600)
    qualify.add_argument("--duration", type=float, default=1800)
    qualify.add_argument("--warmup", type=float, default=30)
    qualify.add_argument("--concurrency", type=int, default=4)
    qualify.add_argument("--request-timeout", type=float, default=300)
    qualify.add_argument("--max-rss-growth-mib", type=float, default=256)
    qualify.add_argument("--allow-short-run", action="store_true")
    qualify.add_argument("--allow-context-reduction", action="store_true")
    qualify.add_argument("--output", type=Path)

    qualification_preflight = commands.add_parser(
        "qualification-preflight",
        help="verify an Apple Silicon runner and its selected vLLM-Metal platform",
    )
    qualification_preflight.add_argument("--backend-executable", required=True, type=Path)

    context = commands.add_parser("context", help="calculate safe context tiers")
    context.add_argument("--model-id", default="unknown")
    context.add_argument("--model-memory-gb", type=float, required=True)
    context.add_argument("--kv-bytes-per-token", type=int, required=True)
    context.add_argument("--workspace-gb", type=float, default=0.0)
    context.add_argument("--model-max-context", type=int)
    context_calibration = context.add_mutually_exclusive_group()
    context_calibration.add_argument("--long-context-report", type=Path)
    context_calibration.add_argument("--auto-calibration", action="store_true")
    context.add_argument("--calibration-margin", type=float, default=0.25)
    context.add_argument("--calibration-hardware-fingerprint")
    context.add_argument(
        "--calibration-backend", choices=("vllm_metal", "mlx_lm"), default="mlx_lm"
    )
    context.add_argument("--application-support", type=Path)

    server = commands.add_parser("serve", help="run the local control daemon")
    server.add_argument("model", nargs="?", help="model ID or local model path")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--max-concurrent-requests", type=int, default=32)
    server.add_argument("--backend-executable")
    server.add_argument("--backend-port", type=int, default=8001)
    server.add_argument("--backend-startup-timeout", type=float, default=600.0)
    server.add_argument("--max-model-len", type=int)
    server.add_argument("--model-integrity-manifest", type=Path)
    server.add_argument("--skip-backend-check", action="store_true")
    server.add_argument("--socket-path")
    server.add_argument("--session-token")
    server.add_argument("--session-token-file")
    server.add_argument("--shutdown-grace-period", type=float, default=30.0)
    server.add_argument("--skip-runtime-probes", action="store_true")
    server_tuning = server.add_mutually_exclusive_group()
    server_tuning.add_argument("--metal-tuning-report", type=Path)
    server_tuning.add_argument("--disable-metal-tuning", action="store_true")
    server.add_argument("--disable-kv-calibration", action="store_true")
    server.add_argument("--kv-calibration-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "hardware":
        _json(detect_hardware().to_dict())
        return 0
    if arguments.command == "doctor":
        report = inspect_backend(arguments.backend_executable)
        _json(report.to_dict())
        return 0 if report.compatible else 1
    if arguments.command == "model-integrity-create":
        try:
            model_root = arguments.model.expanduser().resolve(strict=True)
            output = arguments.output.expanduser().resolve(strict=False)
            if output.is_relative_to(model_root):
                raise ModelIntegrityError("manifest must be stored outside the model tree")
            manifest = build_model_integrity_manifest(model_root)
            save_model_integrity_manifest(manifest, output)
        except (OSError, ModelIntegrityError) as error:
            _json({"passed": False, "error_code": "model_integrity_failed", "detail": str(error)})
            return 2
        _json({"passed": True, "root_sha256": manifest["root_sha256"]})
        return 0
    if arguments.command == "model-integrity-verify":
        try:
            manifest = verify_model_integrity(arguments.model, arguments.manifest)
        except (OSError, ModelIntegrityError) as error:
            _json({"passed": False, "error_code": "model_integrity_failed", "detail": str(error)})
            return 1
        _json({"passed": True, "root_sha256": manifest["root_sha256"]})
        return 0
    if arguments.command == "profile":
        profile = build_profile()
        if arguments.save:
            path = save_profile(profile, Path(arguments.output) if arguments.output else None)
            print(path)
        else:
            _json(profile.to_dict())
        return 0
    if arguments.command == "context":
        if arguments.model_memory_gb < 0 or arguments.workspace_gb < 0:
            raise SystemExit("memory values cannot be negative")
        hardware = detect_hardware()
        spec = ModelMemorySpec(
            model_id=arguments.model_id,
            weights_bytes=int(arguments.model_memory_gb * GIB),
            kv_bytes_per_token=arguments.kv_bytes_per_token,
            workspace_bytes=int(arguments.workspace_gb * GIB),
            model_max_context=arguments.model_max_context,
        )
        calibration = None
        if arguments.long_context_report is not None or arguments.auto_calibration:
            expected_hardware = (
                arguments.calibration_hardware_fingerprint
                or detect_apple_chip_profile().hardware_fingerprint
            )
            if arguments.auto_calibration:
                calibration, _ = discover_latest_kv_calibration(
                    expected_model_id=arguments.model_id,
                    expected_hardware_fingerprint=expected_hardware,
                    expected_backend=arguments.calibration_backend,
                    application_support=arguments.application_support,
                    safety_margin_ratio=arguments.calibration_margin,
                )
            else:
                calibration = load_kv_calibration(
                    arguments.long_context_report,
                    expected_model_id=arguments.model_id,
                    expected_hardware_fingerprint=expected_hardware,
                    expected_backend=arguments.calibration_backend,
                    safety_margin_ratio=arguments.calibration_margin,
                )
            spec = calibration.apply(spec)
        result = recommend_context(hardware.memory, spec).to_dict()
        if calibration is not None:
            result["kv_calibration"] = calibration.to_dict()
        _json(result)
        return 0
    if arguments.command == "execution-profile":
        profile = detect_apple_chip_profile(backend_executable=arguments.backend_executable)
        if arguments.save:
            path = save_chip_profile(profile, Path(arguments.output) if arguments.output else None)
            print(path)
        else:
            _json(profile.to_dict())
        return 0
    if arguments.command == "kernel-shape-profile":
        try:
            contexts = tuple(int(value) for value in arguments.contexts.split(","))
        except ValueError as error:
            raise SystemExit("contexts must be comma-separated integers") from error
        profile = build_model_kernel_shape_profile(
            inspect_model(arguments.model),
            context_tiers=contexts,
            block_tokens=arguments.block_tokens,
        )
        _json(profile.to_dict())
        return 0
    if arguments.command == "metal-shape-benchmark":
        try:
            contexts = tuple(int(value) for value in arguments.contexts.split(","))
        except ValueError as error:
            raise SystemExit("contexts must be comma-separated integers") from error
        chip = detect_apple_chip_profile()
        versions = discover_runtime_versions(arguments.backend_version)
        environment_fingerprint = build_environment_fingerprint(
            platform=chip.platform,
            os_version=chip.os_version,
            toolchain_version=versions.toolchain_version,
            mlx_version=versions.mlx_version,
            backend_version=versions.backend_version,
        )
        profile = build_model_kernel_shape_profile(
            inspect_model(arguments.model),
            context_tiers=contexts,
            block_tokens=arguments.block_tokens,
        )
        benchmark = run_metal_shape_benchmark(
            profile,
            NativeMetalProbeAdapter(timeout_seconds=arguments.timeout),
            hardware_fingerprint=chip.hardware_fingerprint,
            environment_fingerprint=environment_fingerprint,
            samples=arguments.samples,
            maximum_shapes=arguments.maximum_shapes,
        )
        if arguments.stdout:
            _json(benchmark.to_dict())
        else:
            destination = arguments.output or default_metal_shape_benchmark_path(benchmark)
            print(save_metal_shape_benchmark(benchmark, destination))
        return 0 if all(result.passed for result in benchmark.results) else 1
    if arguments.command == "metal-shape-tune":
        try:
            contexts = tuple(int(value) for value in arguments.contexts.split(","))
        except ValueError as error:
            raise SystemExit("contexts must be comma-separated integers") from error
        chip = detect_apple_chip_profile()
        versions = discover_runtime_versions(arguments.backend_version)
        environment_fingerprint = build_environment_fingerprint(
            platform=chip.platform,
            os_version=chip.os_version,
            toolchain_version=versions.toolchain_version,
            mlx_version=versions.mlx_version,
            backend_version=versions.backend_version,
        )
        profile = build_model_kernel_shape_profile(
            inspect_model(arguments.model),
            context_tiers=contexts,
            block_tokens=arguments.block_tokens,
        )
        report = tune_metal_shape_profile(
            profile,
            NativeMetalProbeAdapter(timeout_seconds=arguments.timeout),
            hardware_fingerprint=chip.hardware_fingerprint,
            environment_fingerprint=environment_fingerprint,
            samples=arguments.samples,
            maximum_shapes=arguments.maximum_shapes,
        )
        if arguments.stdout:
            _json(report.to_dict())
        else:
            print(save_metal_tuning_report(report, arguments.output))
        return 0
    if arguments.command == "vllm-metal-integration-inspect":
        try:
            inspection = inspect_vllm_metal_integration(arguments.source_root)
        except (OSError, ValueError) as error:
            _json({"compatible": False, "error": str(error)})
            return 2
        _json(inspection.to_dict())
        return 0 if inspection.compatible else 1
    if arguments.command == "vllm-metal-v2-tune":
        try:
            contexts = tuple(int(value) for value in arguments.contexts.split(","))
            chip = detect_apple_chip_profile()
            if chip.gpu_core_count is None:
                raise ValueError("GPU core count is unavailable")
            inspection = inspect_vllm_metal_integration(arguments.source_root)
            if not inspection.native_v2_detected:
                raise ValueError("vLLM-Metal native v2 topology was not detected")
            model_profile = build_model_kernel_shape_profile(
                inspect_model(arguments.model),
                context_tiers=contexts,
                block_tokens=arguments.block_tokens,
            )
            adapter = VLLMMetalV2MeasurementAdapter(
                arguments.helper,
                timeout_seconds=arguments.timeout,
            )
            capability = adapter.capability()
            if not capability["compatible"]:
                raise ValueError(f"native v2 capability unavailable: {capability['issue']}")
            profile = tune_v2_model_profile(
                model_profile,
                adapter.measure,
                hardware_fingerprint=chip.hardware_fingerprint,
                source_fingerprint=inspection.source_fingerprint,
                gpu_cores=chip.gpu_core_count,
                samples=arguments.samples,
                maximum_shapes=arguments.maximum_shapes,
                prefill_query_tokens=arguments.prefill_query_tokens,
                nax_available=not arguments.disable_nax,
            )
        except (OSError, ValueError, V2MeasurementAdapterError) as error:
            _json({"passed": False, "error": str(error)})
            return 2
        if arguments.stdout:
            _json(profile.to_dict())
        else:
            print(save_v2_tuning_profile(profile, arguments.output))
        return 0
    if arguments.command == "vllm-metal-v2-capability":
        try:
            capability = VLLMMetalV2MeasurementAdapter(
                arguments.helper,
                timeout_seconds=arguments.timeout,
            ).capability()
        except (ValueError, V2MeasurementAdapterError) as error:
            _json({"compatible": False, "error": str(error)})
            return 2
        _json(capability)
        return 0 if capability["compatible"] else 1
    if arguments.command == "phase-profile":
        try:
            chip = detect_apple_chip_profile()
            token = (
                _read_private_token(arguments.session_token_file)
                if arguments.session_token_file
                else None
            )
            result = run_phase_probe(
                PhaseProbeConfig(
                    base_url=arguments.url,
                    model=arguments.model,
                    hardware_fingerprint=arguments.hardware_fingerprint
                    or chip.hardware_fingerprint,
                    backend=arguments.backend,
                    samples=arguments.samples,
                    maximum_output_tokens=arguments.max_tokens,
                    timeout_seconds=arguments.timeout,
                    session_token=token,
                    target_pid=arguments.pid,
                    allow_remote=arguments.allow_remote,
                )
            )
        except (OSError, ValueError, PhaseProbeError) as error:
            code = error.code if isinstance(error, PhaseProbeError) else "invalid_configuration"
            _json({"passed": False, "error_code": code, "detail": str(error)})
            return 2
        _json(result)
        return 0
    if arguments.command == "long-context-evaluate":
        try:
            if arguments.memory_ceiling_gb <= 0:
                raise ValueError("memory ceiling must be positive")
            stages = tuple(int(value.strip()) for value in arguments.stages.split(","))
            chip = detect_apple_chip_profile()
            token = (
                _read_private_token(arguments.session_token_file)
                if arguments.session_token_file
                else None
            )
            probe = PhaseProbeConfig(
                base_url=arguments.url,
                model=arguments.model,
                hardware_fingerprint=arguments.hardware_fingerprint or chip.hardware_fingerprint,
                backend=arguments.backend,
                samples=1,
                maximum_output_tokens=arguments.max_tokens,
                timeout_seconds=arguments.timeout,
                session_token=token,
                target_pid=arguments.pid,
                allow_remote=arguments.allow_remote,
            )
            evaluator = LongContextEvaluator(
                model_id=arguments.model,
                hardware_fingerprint=probe.hardware_fingerprint,
                memory_ceiling_bytes=int(arguments.memory_ceiling_gb * GIB),
                backend=arguments.backend,
            )
            adapter_type = (
                MLXLongContextAdapter
                if arguments.backend == "mlx_lm"
                else VLLMLongContextAdapter
            )
            model_max_context = arguments.model_max_context
            if model_max_context is None:
                try:
                    model_max_context = inspect_model(
                        arguments.model
                    ).memory_spec.model_max_context
                except ModelInspectionError:
                    pass
            adapter = adapter_type(
                probe,
                state_bytes_per_token=arguments.state_bytes_per_token,
                load_peak_rss_bytes=arguments.load_peak_rss_bytes,
                model_max_context=model_max_context,
            )
            result = evaluator.evaluate(stages, adapter.measure)
            output = arguments.output
            if arguments.save:
                output = default_calibration_report_path(
                    arguments.model,
                    probe.hardware_fingerprint,
                    arguments.backend,
                    result["evaluation_id"],
                    application_support=arguments.application_support,
                )
            if output is not None:
                save_long_context_report(result, output)
        except (OSError, ValueError, LongContextEvaluationError) as error:
            code = (
                error.code
                if isinstance(error, LongContextEvaluationError)
                else "invalid_configuration"
            )
            _json({"passed": False, "error_code": code, "detail": str(error)})
            return 2
        _json(result)
        return 0 if result["passed"] else 1
    if arguments.command == "qualify-model":
        try:
            if arguments.max_rss_growth_mib < 0:
                raise ValueError("RSS growth limit cannot be negative")
            vllm_version = None
            if arguments.backend_kind == "vllm_metal":
                compatibility = inspect_backend(str(arguments.backend_executable))
                if not compatibility.compatible:
                    raise ValueError("incompatible backend: " + ", ".join(compatibility.issues))
                vllm_version = compatibility.vllm_version
            else:
                mlx_compatibility = inspect_mlx_lm_backend(arguments.backend_executable)
                if not mlx_compatibility.compatible:
                    raise ValueError(
                        "incompatible backend: " + ", ".join(mlx_compatibility.issues)
                    )
            result = qualify_model(
                QualificationConfig(
                    model=arguments.model,
                    executable=arguments.backend_executable,
                    port=arguments.backend_port,
                    max_model_len=arguments.max_model_len,
                    startup_timeout_seconds=arguments.startup_timeout,
                    duration_seconds=arguments.duration,
                    warmup_seconds=arguments.warmup,
                    concurrency=arguments.concurrency,
                    request_timeout_seconds=arguments.request_timeout,
                    max_rss_growth_bytes=int(arguments.max_rss_growth_mib * 1024 * 1024),
                    require_30_minute_window=not arguments.allow_short_run,
                    vllm_version=vllm_version,
                    allow_context_reduction=arguments.allow_context_reduction,
                    backend_kind=arguments.backend_kind,
                )
            )
            save_qualification_report(
                result,
                arguments.output or default_qualification_report_path(arguments.model),
            )
        except (OSError, ValueError, RuntimeError) as error:
            _json({"passed": False, "error_code": "qualification_failed", "detail": str(error)})
            return 2
        _json(result)
        return 0 if result["passed"] else 1
    if arguments.command == "qualification-preflight":
        result = run_qualification_preflight(arguments.backend_executable)
        _json(result.to_dict())
        return 0 if result.eligible else 1
    if arguments.command == "serve":
        try:
            serve(
                host=arguments.host,
                port=arguments.port,
                max_concurrent_requests=arguments.max_concurrent_requests,
                model=arguments.model,
                backend_executable=arguments.backend_executable,
                backend_port=arguments.backend_port,
                backend_startup_timeout=arguments.backend_startup_timeout,
                max_model_len=arguments.max_model_len,
                model_integrity_manifest=arguments.model_integrity_manifest,
                require_compatible_backend=not arguments.skip_backend_check,
                socket_path=arguments.socket_path,
                session_token=arguments.session_token,
                session_token_file=arguments.session_token_file,
                shutdown_grace_period=arguments.shutdown_grace_period,
                enable_runtime_probes=not arguments.skip_runtime_probes,
                enable_metal_tuning=not arguments.disable_metal_tuning,
                metal_tuning_report=arguments.metal_tuning_report,
                enable_kv_calibration=not arguments.disable_kv_calibration,
                kv_calibration_root=arguments.kv_calibration_root,
            )
        except (RuntimeError, ValueError) as error:
            print(f"vllm-apple: {error}", file=sys.stderr)
            return 2
        return 0
    return 2
