from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .artifact_admission import assess_artifact_admission_for_path
from .compat import assess_candidate_backend, inspect_backend, inspect_mlx_lm_backend
from .context import recommend_context
from .daemon import serve
from .execution_profile import detect_apple_chip_profile, save_chip_profile
from .hardware import detect_hardware
from .huggingface_metadata import HuggingFaceMetadataError, fetch_hugging_face_metadata
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
from .model import ModelInspectionError, inspect_model, inspect_model_metadata
from .model_integrity import (
    ModelIntegrityError,
    build_model_integrity_manifest,
    save_model_integrity_manifest,
    sign_model_integrity_manifest,
    verify_model_integrity,
    verify_signed_model_integrity,
)
from .model_recommendation import build_model_recommendation
from .phase_probe import PhaseProbeConfig, PhaseProbeError, run_phase_probe
from .profile import build_profile, save_profile
from .qualification import (
    QualificationConfig,
    default_qualification_report_path,
    qualify_model,
    save_qualification_report,
)
from .mlx_qwen4_readiness import inspect_mlx_qwen4_readiness
from .qualification_preflight import run_qualification_preflight
from .qwen4_cache_contract import run_qwen4_cache_fixture
from .qwen4_adapter_contract import build_qwen4_adapter_contract
from .qwen4_adapter_loader import inspect_qwen4_adapter_headers
from .qwen4_conversion_plan import build_qwen4_conversion_plan
from .qwen4_mlx_fixture import run_qwen4_mlx_fixture
from .qwen4_load_plan import build_qwen4_component_load_plan
from .qwen4_shard_stager import stage_qwen4_shards, verify_qwen4_stage
from .qwen4_weight_map import inspect_qwen4_weight_map
from .qualification_bundle import (
    QualificationBundleError,
    build_qualification_bundle,
    save_qualification_bundle,
    sign_qualification_bundle,
    verify_qualification_bundle,
    verify_signed_qualification_bundle,
)
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
from .vllm_metal_v2_observation import load_v2_observations
from .vllm_metal_v2_tuning import (
    build_v2_hardware_fingerprint,
    restore_quarantined_v2_profile,
    save_v2_tuning_profile,
    tune_v2_model_profile,
    tune_v2_observed_shapes,
)


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _candidate_versions(arguments: argparse.Namespace) -> tuple[str, str, str] | None:
    values = (
        arguments.candidate_vllm_version,
        arguments.candidate_vllm_metal_version,
        arguments.candidate_transformers_version,
    )
    if any(values) and not all(values):
        raise ValueError("all candidate stack versions must be provided together")
    return values if all(values) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-apple")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("hardware", help="detect Apple hardware and memory")

    artifact_admission = commands.add_parser(
        "artifact-admission", help="check memory and disk fit before downloading a model"
    )
    artifact_admission.add_argument("--model", required=True)
    artifact_size = artifact_admission.add_mutually_exclusive_group(required=True)
    artifact_size.add_argument("--artifact-gib", type=float)
    artifact_size.add_argument("--artifact-bytes", type=int)
    resident_size = artifact_admission.add_mutually_exclusive_group(required=True)
    resident_size.add_argument("--resident-gib", type=float)
    resident_size.add_argument("--resident-bytes", type=int)
    artifact_admission.add_argument("--target", type=Path, default=Path("models"))
    artifact_admission.add_argument("--staging-factor", type=float, default=1.05)

    doctor = commands.add_parser("doctor", help="inspect the vLLM-Metal environment")
    doctor.add_argument("--backend-executable")

    inspect = commands.add_parser(
        "inspect-model", help="inspect local metadata and recommend a safe configuration"
    )
    inspect.add_argument("model")
    inspect.add_argument("--backend", choices=("vllm_metal", "mlx_lm"), default="vllm_metal")
    inspect.add_argument("--feature", action="append", dest="features")
    inspect.add_argument(
        "--mode", action="append", choices=("text", "vision", "mtp", "yarn"), dest="modes"
    )

    metadata = commands.add_parser(
        "fetch-model-metadata",
        help="fetch bounded config metadata without downloading model weights",
    )
    metadata.add_argument("model")
    metadata.add_argument("--revision", default="main")
    metadata.add_argument("--timeout", type=float, default=10.0)

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
    integrity_verify.add_argument("--signature", type=Path)
    integrity_verify.add_argument("--trusted-ca", type=Path)
    integrity_verify.add_argument("--expected-signer-sha256")
    integrity_sign = commands.add_parser(
        "model-integrity-sign", help="create a detached CMS signature for a model manifest"
    )
    integrity_sign.add_argument("--manifest", required=True, type=Path)
    integrity_sign.add_argument("--certificate", required=True, type=Path)
    integrity_sign.add_argument("--private-key", required=True, type=Path)
    integrity_sign.add_argument("--output", required=True, type=Path)

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
    v2_tune.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    v2_tune.add_argument("--observed-shapes", type=Path)
    v2_output = v2_tune.add_mutually_exclusive_group()
    v2_output.add_argument("--output", type=Path)
    v2_output.add_argument("--stdout", action="store_true")

    v2_capability = commands.add_parser(
        "vllm-metal-v2-capability",
        help="probe the native-v2 measurement symbol without running a benchmark",
    )
    v2_capability.add_argument("--helper", required=True, type=Path)
    v2_capability.add_argument("--timeout", type=float, default=10)

    v2_restore = commands.add_parser(
        "vllm-metal-v2-restore",
        help="remeasure a quarantined native-v2 profile before restoring it",
    )
    v2_restore.add_argument("profile_id")
    v2_restore.add_argument("--source-root", required=True, type=Path)
    v2_restore.add_argument("--helper", required=True, type=Path)
    v2_restore.add_argument("--samples", type=int, default=3)
    v2_restore.add_argument("--timeout", type=float, default=30)
    v2_restore.add_argument("--disable-nax", action="store_true")
    v2_restore.add_argument("--profile-root", type=Path)

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
    qualify.add_argument("--phase-samples", type=int, default=3)
    qualify.add_argument("--phase-output-tokens", type=int, default=32)
    qualify.add_argument("--skip-quality-smoke", action="store_true")
    qualify.add_argument(
        "--mode",
        action="append",
        choices=("text", "vision", "mtp", "yarn"),
        dest="qualification_modes",
    )
    qualify.add_argument("--allow-short-run", action="store_true")
    qualify.add_argument("--allow-context-reduction", action="store_true")
    qualify.add_argument("--output", type=Path)
    qualify.add_argument("--candidate-vllm-version")
    qualify.add_argument("--candidate-vllm-metal-version")
    qualify.add_argument("--candidate-transformers-version")

    qualification_preflight = commands.add_parser(
        "qualification-preflight",
        help="verify an Apple Silicon runner and its selected vLLM-Metal platform",
    )
    qualification_preflight.add_argument("--backend-executable", required=True, type=Path)
    qualification_preflight.add_argument(
        "--backend-kind", choices=("vllm_metal", "mlx_lm"), default="vllm_metal"
    )
    qualification_preflight.add_argument("--candidate-vllm-version")
    qualification_preflight.add_argument("--candidate-vllm-metal-version")
    qualification_preflight.add_argument("--candidate-transformers-version")
    preflight_model = qualification_preflight.add_mutually_exclusive_group()
    preflight_model.add_argument("--model")
    preflight_model.add_argument(
        "--model-metadata",
        type=Path,
        help="inspect bounded local config metadata without requiring model weights",
    )
    qualification_preflight.add_argument("--max-model-len", type=int)
    qualification_preflight.add_argument(
        "--mode", action="append", choices=("text", "vision", "mtp", "yarn"), dest="preflight_modes"
    )
    mlx_qwen4_readiness = commands.add_parser(
        "mlx-qwen4-readiness",
        help="inspect reusable MLX Qwen4 components without importing MLX or allocating Metal",
    )
    mlx_qwen4_readiness.add_argument("--backend-executable", required=True, type=Path)
    qwen4_mlx_fixture = commands.add_parser(
        "qwen4-mlx-fixture",
        help="compare bounded MLX Qwen4 primitives with the dependency-free CPU reference",
    )
    qwen4_mlx_fixture.add_argument("--python-executable", required=True, type=Path)
    qwen4_weight_map = commands.add_parser(
        "qwen4-weight-map-inspect",
        help="validate a bounded Qwen4 safetensors index without loading weights",
    )
    qwen4_weight_map.add_argument("--model-metadata", required=True, type=Path)
    qwen4_weight_map.add_argument("--index", required=True, type=Path)
    qwen4_weight_map.add_argument(
        "--mode", action="append", choices=("text", "mtp", "vision"), dest="qwen4_weight_modes"
    )
    qwen4_cache_fixture = commands.add_parser(
        "qwen4-cache-fixture",
        help="verify Qwen4 prefill, segmented prefill, and decode cache boundaries without tensors",
    )
    qwen4_cache_fixture.add_argument("--model-metadata", required=True, type=Path)
    qwen4_conversion_plan = commands.add_parser(
        "qwen4-conversion-plan",
        help="build a one-shard-at-a-time Qwen4 MLX conversion plan without loading tensors",
    )
    qwen4_conversion_plan.add_argument("--model-metadata", required=True, type=Path)
    qwen4_conversion_plan.add_argument("--index", required=True, type=Path)
    qwen4_conversion_plan.add_argument(
        "--mode", action="append", choices=("text", "mtp", "vision"), dest="qwen4_plan_modes"
    )
    qwen4_shard_stage = commands.add_parser(
        "qwen4-shard-stage",
        help="atomically stage Qwen4 shards with bounded memory and digest-bound resume",
    )
    qwen4_shard_stage.add_argument("--source", required=True, type=Path)
    qwen4_shard_stage.add_argument("--output", required=True, type=Path)
    qwen4_shard_stage.add_argument("--maximum-output-bytes", required=True, type=int)
    qwen4_shard_stage.add_argument(
        "--mode", action="append", choices=("text", "mtp", "vision"), dest="qwen4_stage_modes"
    )
    qwen4_shard_stage.add_argument("--resume", action="store_true")
    qwen4_shard_stage.add_argument(
        "--execute", action="store_true", help="explicitly authorize copying the model artifact"
    )
    qwen4_stage_verify = commands.add_parser(
        "qwen4-stage-verify",
        help="verify every staged Qwen4 shard before adapter construction",
    )
    qwen4_stage_verify.add_argument("--stage", required=True, type=Path)
    qwen4_stage_verify.add_argument("--maximum-artifact-bytes", required=True, type=int)
    qwen4_stage_verify.add_argument(
        "--mode", action="append", choices=("text", "mtp", "vision"), dest="qwen4_verify_modes"
    )
    qwen4_adapter_contract = commands.add_parser(
        "qwen4-adapter-contract",
        help="build a component/shard execution contract from a verified Qwen4 stage",
    )
    qwen4_adapter_contract.add_argument("--stage", required=True, type=Path)
    qwen4_adapter_contract.add_argument("--maximum-artifact-bytes", required=True, type=int)
    qwen4_adapter_contract.add_argument(
        "--mode", action="append", choices=("text", "mtp", "vision"), dest="qwen4_adapter_modes"
    )
    qwen4_adapter_headers = commands.add_parser(
        "qwen4-adapter-headers",
        help="validate bounded safetensors headers against a verified Qwen4 adapter contract",
    )
    qwen4_adapter_headers.add_argument("--stage", required=True, type=Path)
    qwen4_adapter_headers.add_argument("--maximum-artifact-bytes", required=True, type=int)
    qwen4_adapter_headers.add_argument("--maximum-header-bytes", type=int)
    qwen4_adapter_headers.add_argument(
        "--mode", action="append", choices=("text", "mtp", "vision"), dest="qwen4_header_modes"
    )
    qwen4_load_plan = commands.add_parser(
        "qwen4-load-plan",
        help="calculate resident and on-demand Qwen4 component memory without allocating a backend",
    )
    qwen4_load_plan.add_argument("--stage", required=True, type=Path)
    qwen4_load_plan.add_argument("--maximum-artifact-bytes", required=True, type=int)
    qwen4_load_plan.add_argument("--target-dtype", choices=("BF16", "F16", "F32"), default="BF16")
    qwen4_load_plan.add_argument("--scratch-bytes-per-tensor", type=int, default=0)
    qwen4_load_plan.add_argument(
        "--mode", action="append", choices=("text", "mtp", "vision"), dest="qwen4_load_modes"
    )
    qualification_bundle = commands.add_parser(
        "qualification-bundle", help="build a bounded, tamper-evident promotion bundle"
    )
    qualification_bundle.add_argument("--reports", required=True, type=Path)
    qualification_bundle.add_argument("--output", required=True, type=Path)
    qualification_bundle_verify = commands.add_parser(
        "qualification-bundle-verify", help="verify a promotion bundle and its source evidence"
    )
    qualification_bundle_verify.add_argument("--reports", required=True, type=Path)
    qualification_bundle_verify.add_argument("--bundle", required=True, type=Path)
    qualification_bundle_verify.add_argument("--signature", type=Path)
    qualification_bundle_verify.add_argument("--trusted-ca", type=Path)
    qualification_bundle_verify.add_argument("--expected-signer-sha256")
    qualification_bundle_sign = commands.add_parser(
        "qualification-bundle-sign", help="sign a promotion bundle with detached CMS"
    )
    qualification_bundle_sign.add_argument("--reports", required=True, type=Path)
    qualification_bundle_sign.add_argument("--bundle", required=True, type=Path)
    qualification_bundle_sign.add_argument("--certificate", required=True, type=Path)
    qualification_bundle_sign.add_argument("--private-key", required=True, type=Path)
    qualification_bundle_sign.add_argument("--output", required=True, type=Path)

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
    server.add_argument("--vllm-metal-source-root", type=Path)
    server.add_argument("--vllm-metal-v2-helper", type=Path)
    server.add_argument("--disable-native-v2-idle-tuning", action="store_true")
    server.add_argument("--native-v2-preference-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "hardware":
        _json(detect_hardware().to_dict())
        return 0
    if arguments.command == "artifact-admission":
        try:
            gib_values = tuple(
                value
                for value in (arguments.artifact_gib, arguments.resident_gib)
                if value is not None
            )
            if any(not math.isfinite(value) or value <= 0 or value > 16_384 for value in gib_values):
                raise ValueError("artifact and resident sizes must be between 0 and 16384 GiB")
            byte_values = tuple(
                value
                for value in (arguments.artifact_bytes, arguments.resident_bytes)
                if value is not None
            )
            if any(value <= 0 or value > 16_384 * GIB for value in byte_values):
                raise ValueError("artifact and resident byte sizes are outside the supported range")
            artifact_bytes = arguments.artifact_bytes
            if artifact_bytes is None:
                artifact_bytes = math.ceil(arguments.artifact_gib * GIB)
            resident_bytes = arguments.resident_bytes
            if resident_bytes is None:
                resident_bytes = math.ceil(arguments.resident_gib * GIB)
            admission = assess_artifact_admission_for_path(
                model=arguments.model,
                artifact_bytes=artifact_bytes,
                estimated_resident_bytes=resident_bytes,
                hardware=detect_hardware(),
                target=arguments.target,
                staging_factor=arguments.staging_factor,
            )
        except (OSError, ValueError) as error:
            _json(
                {
                    "eligible": False,
                    "error_code": "artifact_admission_failed",
                    "detail": str(error),
                }
            )
            return 2
        _json(admission.to_dict())
        return 0 if admission.eligible else 1
    if arguments.command == "doctor":
        report = inspect_backend(arguments.backend_executable)
        _json(report.to_dict())
        return 0 if report.compatible else 1
    if arguments.command == "inspect-model":
        try:
            features = frozenset(arguments.features or ())
            if len(features) > 64 or any(not value or len(value) > 128 for value in features):
                raise ValueError("model feature declarations are invalid")
            report = build_model_recommendation(
                inspect_model(arguments.model),
                detect_hardware(),
                backend=arguments.backend,
                available_features=features,
                requested_modes=frozenset(arguments.modes or ("text",)),
            )
        except (OSError, ModelInspectionError, ValueError) as error:
            _json({"runnable": False, "error_code": "model_inspection_failed", "detail": str(error)})
            return 2
        _json(report.to_dict())
        return 0 if report.runnable else 1
    if arguments.command == "fetch-model-metadata":
        try:
            metadata = fetch_hugging_face_metadata(
                arguments.model,
                revision=arguments.revision,
                timeout_seconds=arguments.timeout,
            )
        except (HuggingFaceMetadataError, ValueError) as error:
            _json(
                {
                    "fetched": False,
                    "error_code": "model_metadata_fetch_failed",
                    "detail": str(error),
                }
            )
            return 2
        _json(metadata.to_dict())
        return 0
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
            signed_inputs = (
                arguments.signature,
                arguments.trusted_ca,
                arguments.expected_signer_sha256,
            )
            if any(value is not None for value in signed_inputs):
                if not all(value is not None for value in signed_inputs):
                    raise ModelIntegrityError("signed verification inputs must be provided together")
                manifest = verify_signed_model_integrity(
                    arguments.model,
                    arguments.manifest,
                    arguments.signature,
                    arguments.trusted_ca,
                    arguments.expected_signer_sha256,
                )
            else:
                manifest = verify_model_integrity(arguments.model, arguments.manifest)
        except (OSError, ModelIntegrityError) as error:
            _json({"passed": False, "error_code": "model_integrity_failed", "detail": str(error)})
            return 1
        _json({"passed": True, "root_sha256": manifest["root_sha256"]})
        return 0
    if arguments.command == "model-integrity-sign":
        try:
            output = sign_model_integrity_manifest(
                arguments.manifest,
                arguments.certificate,
                arguments.private_key,
                arguments.output,
            )
        except (OSError, ModelIntegrityError) as error:
            _json({"passed": False, "error_code": "model_integrity_signing_failed", "detail": str(error)})
            return 2
        _json({"passed": True, "signature": str(output)})
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
            hardware_fingerprint = build_v2_hardware_fingerprint(detect_hardware())
            adapter = VLLMMetalV2MeasurementAdapter(
                arguments.helper,
                timeout_seconds=arguments.timeout,
            )
            capability = adapter.capability()
            if not capability["compatible"]:
                raise ValueError(f"native v2 capability unavailable: {capability['issue']}")
            if arguments.observed_shapes is not None:
                observed = load_v2_observations(
                    arguments.observed_shapes,
                    hardware_fingerprint=hardware_fingerprint,
                    source_fingerprint=inspection.source_fingerprint,
                )
                profile = tune_v2_observed_shapes(
                    observed,
                    adapter.measure,
                    hardware_fingerprint=hardware_fingerprint,
                    source_fingerprint=inspection.source_fingerprint,
                    samples=arguments.samples,
                    maximum_shapes=arguments.maximum_shapes,
                    nax_available=not arguments.disable_nax,
                )
            else:
                model_profile = build_model_kernel_shape_profile(
                    inspect_model(arguments.model),
                    context_tiers=contexts,
                    block_tokens=arguments.block_tokens,
                )
                profile = tune_v2_model_profile(
                    model_profile,
                    adapter.measure,
                    hardware_fingerprint=hardware_fingerprint,
                    source_fingerprint=inspection.source_fingerprint,
                    gpu_cores=chip.gpu_core_count,
                    samples=arguments.samples,
                    maximum_shapes=arguments.maximum_shapes,
                    prefill_query_tokens=arguments.prefill_query_tokens,
                    nax_available=not arguments.disable_nax,
                    floating_dtype=arguments.dtype,
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
    if arguments.command == "vllm-metal-v2-restore":
        try:
            inspection = inspect_vllm_metal_integration(arguments.source_root)
            if not inspection.native_v2_detected:
                raise ValueError("vLLM-Metal native v2 topology was not detected")
            adapter = VLLMMetalV2MeasurementAdapter(
                arguments.helper,
                timeout_seconds=arguments.timeout,
            )
            capability = adapter.capability()
            if not capability["compatible"]:
                raise ValueError(f"native v2 capability unavailable: {capability['issue']}")
            profile, path = restore_quarantined_v2_profile(
                arguments.profile_id,
                adapter.measure,
                hardware_fingerprint=build_v2_hardware_fingerprint(detect_hardware()),
                source_fingerprint=inspection.source_fingerprint,
                samples=arguments.samples,
                nax_available=not arguments.disable_nax,
                root=arguments.profile_root,
            )
        except (OSError, ValueError, V2MeasurementAdapterError) as error:
            _json({"passed": False, "error": str(error)})
            return 2
        _json(
            {
                "passed": True,
                "profile_id": profile.profile_id,
                "restored_from_profile_id": arguments.profile_id,
                "path": str(path),
            }
        )
        return 0
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
            candidate_versions = _candidate_versions(arguments)
            if candidate_versions is not None and arguments.backend_kind != "vllm_metal":
                raise ValueError("candidate stack qualification requires vllm_metal")
            vllm_version = None
            backend_versions: dict[str, str | None] | None = None
            architecture_features: tuple[str, ...] = ()
            if arguments.backend_kind == "vllm_metal":
                compatibility = inspect_backend(str(arguments.backend_executable))
                candidate_issues = (
                    assess_candidate_backend(
                        compatibility,
                        expected_vllm=candidate_versions[0],
                        expected_vllm_metal=candidate_versions[1],
                        expected_transformers=candidate_versions[2],
                    )
                    if candidate_versions is not None
                    else compatibility.issues
                )
                if candidate_issues:
                    raise ValueError("incompatible backend: " + ", ".join(candidate_issues))
                vllm_version = compatibility.vllm_version
                backend_versions = {
                    "vllm": compatibility.vllm_version,
                    "vllm_metal": compatibility.vllm_metal_version,
                    "transformers": compatibility.transformers_version,
                }
                architecture_features = compatibility.architecture_features
            else:
                mlx_compatibility = inspect_mlx_lm_backend(arguments.backend_executable)
                if not mlx_compatibility.compatible:
                    raise ValueError(
                        "incompatible backend: " + ", ".join(mlx_compatibility.issues)
                    )
                architecture_features = mlx_compatibility.architecture_features
                backend_versions = {"mlx_lm": mlx_compatibility.mlx_lm_version}
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
                    architecture_features=architecture_features,
                    phase_samples=arguments.phase_samples,
                    phase_output_tokens=arguments.phase_output_tokens,
                    quality_smoke=not arguments.skip_quality_smoke,
                    requested_modes=tuple(arguments.qualification_modes or ("text",)),
                    backend_versions=backend_versions,
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
        try:
            candidate_versions = _candidate_versions(arguments)
        except ValueError as error:
            _json({"eligible": False, "error_code": "invalid_candidate_stack", "detail": str(error)})
            return 2
        result = run_qualification_preflight(
            arguments.backend_executable,
            backend_kind=arguments.backend_kind,
            candidate_versions=candidate_versions,
            model=arguments.model,
            model_metadata=arguments.model_metadata,
            max_model_len=arguments.max_model_len,
            requested_modes=tuple(arguments.preflight_modes or ("text",)),
        )
        _json(result.to_dict())
        return 0 if result.eligible else 1
    if arguments.command == "mlx-qwen4-readiness":
        try:
            result = inspect_mlx_qwen4_readiness(arguments.backend_executable)
        except ValueError as error:
            _json({"ready": False, "error_code": "mlx_qwen4_readiness_failed", "detail": str(error)})
            return 2
        _json(result)
        return 0 if result["ready"] else 1
    if arguments.command == "qwen4-mlx-fixture":
        try:
            result = run_qwen4_mlx_fixture(arguments.python_executable)
        except ValueError as error:
            _json({"passed": False, "error_code": "qwen4_mlx_fixture_failed", "detail": str(error)})
            return 2
        _json(result)
        return 0 if result["passed"] else 1
    if arguments.command == "qwen4-weight-map-inspect":
        try:
            result = inspect_qwen4_weight_map(
                arguments.model_metadata,
                arguments.index,
                requested_modes=tuple(arguments.qwen4_weight_modes or ("text",)),
            )
        except (ModelInspectionError, ValueError) as error:
            _json({"compatible": False, "error_code": "qwen4_weight_map_failed", "detail": str(error)})
            return 2
        _json(result)
        return 0 if result["compatible"] else 1
    if arguments.command == "qwen4-cache-fixture":
        try:
            config, _ = inspect_model_metadata(arguments.model_metadata)
            result = run_qwen4_cache_fixture(config)
        except (ModelInspectionError, ValueError) as error:
            _json({"passed": False, "error_code": "qwen4_cache_fixture_failed", "detail": str(error)})
            return 2
        _json(result)
        return 0 if result["passed"] else 1
    if arguments.command == "qwen4-conversion-plan":
        try:
            result = build_qwen4_conversion_plan(
                arguments.model_metadata,
                arguments.index,
                requested_modes=tuple(arguments.qwen4_plan_modes or ("text",)),
            )
        except (ModelInspectionError, ValueError) as error:
            _json({"passed": False, "error_code": "qwen4_conversion_plan_failed", "detail": str(error)})
            return 2
        _json(result)
        return 0
    if arguments.command == "qwen4-shard-stage":
        if not arguments.execute:
            _json(
                {
                    "completed": False,
                    "error_code": "explicit_execution_required",
                    "detail": "Pass --execute after reviewing the source, output, and byte ceiling.",
                }
            )
            return 2
        try:
            result = stage_qwen4_shards(
                arguments.source,
                arguments.output,
                maximum_output_bytes=arguments.maximum_output_bytes,
                requested_modes=tuple(arguments.qwen4_stage_modes or ("text",)),
                resume=arguments.resume,
            )
        except (OSError, ModelInspectionError, ValueError) as error:
            _json({"completed": False, "error_code": "qwen4_shard_stage_failed", "detail": str(error)})
            return 2
        _json(result)
        return 0
    if arguments.command == "qwen4-stage-verify":
        try:
            result = verify_qwen4_stage(
                arguments.stage,
                requested_modes=tuple(arguments.qwen4_verify_modes or ("text",)),
                maximum_artifact_bytes=arguments.maximum_artifact_bytes,
            )
        except (OSError, ModelInspectionError, ValueError) as error:
            _json({"verified": False, "error_code": "qwen4_stage_verify_failed", "detail": str(error)})
            return 2
        _json(result)
        return 0
    if arguments.command == "qwen4-adapter-contract":
        try:
            result = build_qwen4_adapter_contract(
                arguments.stage,
                requested_modes=tuple(arguments.qwen4_adapter_modes or ("text",)),
                maximum_artifact_bytes=arguments.maximum_artifact_bytes,
            )
        except (OSError, ModelInspectionError, ValueError) as error:
            _json(
                {"passed": False, "error_code": "qwen4_adapter_contract_failed", "detail": str(error)}
            )
            return 2
        _json(result)
        return 0
    if arguments.command == "qwen4-adapter-headers":
        keyword_arguments = {}
        if arguments.maximum_header_bytes is not None:
            keyword_arguments["maximum_header_bytes"] = arguments.maximum_header_bytes
        try:
            result = inspect_qwen4_adapter_headers(
                arguments.stage,
                requested_modes=tuple(arguments.qwen4_header_modes or ("text",)),
                maximum_artifact_bytes=arguments.maximum_artifact_bytes,
                **keyword_arguments,
            )
        except (OSError, ModelInspectionError, ValueError) as error:
            _json({"passed": False, "error_code": "qwen4_adapter_headers_failed", "detail": str(error)})
            return 2
        _json(result)
        return 0
    if arguments.command == "qwen4-load-plan":
        try:
            result = build_qwen4_component_load_plan(
                arguments.stage,
                requested_modes=tuple(arguments.qwen4_load_modes or ("text",)),
                maximum_artifact_bytes=arguments.maximum_artifact_bytes,
                target_dtype=arguments.target_dtype,
                scratch_bytes_per_tensor=arguments.scratch_bytes_per_tensor,
            )
        except (OSError, ModelInspectionError, ValueError) as error:
            _json({"passed": False, "error_code": "qwen4_load_plan_failed", "detail": str(error)})
            return 2
        _json(result)
        return 0
    if arguments.command == "qualification-bundle":
        try:
            bundle = build_qualification_bundle(arguments.reports)
            output = save_qualification_bundle(bundle, arguments.output)
        except (OSError, ValueError, QualificationBundleError) as error:
            _json({"passed": False, "error_code": "qualification_bundle_failed", "detail": str(error)})
            return 2
        _json({"passed": True, "bundle_id": bundle["bundle_id"], "output": str(output)})
        return 0
    if arguments.command == "qualification-bundle-verify":
        try:
            signed = (
                arguments.signature,
                arguments.trusted_ca,
                arguments.expected_signer_sha256,
            )
            if any(value is not None for value in signed) and not all(
                value is not None for value in signed
            ):
                raise QualificationBundleError(
                    "signed qualification bundle inputs must be provided together"
                )
            bundle = (
                verify_signed_qualification_bundle(
                    arguments.reports,
                    arguments.bundle,
                    arguments.signature,
                    arguments.trusted_ca,
                    arguments.expected_signer_sha256,
                )
                if all(value is not None for value in signed)
                else verify_qualification_bundle(arguments.reports, arguments.bundle)
            )
        except (OSError, ValueError, QualificationBundleError) as error:
            _json({"passed": False, "error_code": "qualification_bundle_failed", "detail": str(error)})
            return 1
        _json({"passed": True, "bundle_id": bundle["bundle_id"]})
        return 0
    if arguments.command == "qualification-bundle-sign":
        try:
            output = sign_qualification_bundle(
                arguments.reports,
                arguments.bundle,
                arguments.certificate,
                arguments.private_key,
                arguments.output,
            )
        except (OSError, ValueError, QualificationBundleError) as error:
            _json({"passed": False, "error_code": "qualification_bundle_signing_failed", "detail": str(error)})
            return 2
        _json({"passed": True, "signature": str(output)})
        return 0
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
                enable_native_v2_idle_tuning=not arguments.disable_native_v2_idle_tuning,
                vllm_metal_source_root=arguments.vllm_metal_source_root,
                vllm_metal_v2_helper=arguments.vllm_metal_v2_helper,
                native_v2_preference_path=arguments.native_v2_preference_path,
            )
        except (RuntimeError, ValueError) as error:
            print(f"vllm-apple: {error}", file=sys.stderr)
            return 2
        return 0
    return 2
