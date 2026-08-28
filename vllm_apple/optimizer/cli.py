from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

from ..hardware import detect_hardware
from ..model import ModelInspectionError, inspect_model
from ..types import GIB
from .adapters import (
    AdapterUnavailableError,
    MLXExportReport,
    MLXOptimizationAdapter,
    builtin_adapter_registry,
    persist_artifact_manifest,
)
from .checkpoint import CheckpointLeaseError, CheckpointStore
from .errors import OptimizerErrorCode, OptimizerFailure, Recoverability
from .evaluation import (
    compare_perplexity_reports,
    load_perplexity_report,
    persist_evaluation_report,
)
from .generation_evaluation import compare_generation_reports, load_generation_report
from .planner import build_dry_run_plan
from .profiler import OptimizationPerformanceProfile, profile_optimizer_io
from .safety import OptimizationPathError
from .types import OptimizationObjective, ResourceBudget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-apple-optimize")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="create a side-effect-free optimization plan")
    plan.add_argument("model")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument(
        "--objective",
        choices=[value.value for value in OptimizationObjective],
        default="balanced",
    )
    plan.add_argument("--max-memory-gb", type=float)
    plan.add_argument("--max-disk-gb", type=float)
    plan.add_argument("--max-duration-seconds", type=int)
    plan.add_argument("--license")
    plan.add_argument("--performance-profile", type=Path)
    profile = commands.add_parser("profile", help="measure bounded model I/O throughput")
    profile.add_argument("model")
    profile.add_argument("--workspace", type=Path, required=True)
    profile.add_argument("--sample-mib", type=int, default=64)
    capabilities = commands.add_parser(
        "capabilities",
        help="detect versioned optimization adapter capabilities without importing them",
    )
    capabilities.add_argument("model")
    export = commands.add_parser(
        "export",
        help="plan or explicitly execute a checkpointed MLX weight export",
    )
    export.add_argument("model")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--checkpoint-root", type=Path, required=True)
    export.add_argument("--plan-id", required=True)
    export.add_argument("--bits", type=int, choices=(4, 8), default=4)
    export.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    export.add_argument("--max-output-gb", type=float, required=True)
    export.add_argument("--manifest-output", type=Path)
    export.add_argument("--license")
    export.add_argument("--timeout-seconds", type=float)
    export.add_argument("--resume", action="store_true")
    export.add_argument("--execute", action="store_true")
    evaluate = commands.add_parser(
        "evaluate",
        help="evaluate one local MLX-compatible model with bounded JSONL perplexity",
    )
    evaluate.add_argument("model")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--max-samples", type=int, default=256)
    evaluate.add_argument("--max-tokens-per-sample", type=int, default=512)
    evaluate.add_argument("--max-total-tokens", type=int, default=131_072)
    gate = commands.add_parser(
        "quality-gate",
        help="compare baseline and candidate perplexity reports",
    )
    gate.add_argument("--baseline", type=Path, required=True)
    gate.add_argument("--candidate", type=Path, required=True)
    gate.add_argument("--max-perplexity-regression", type=float, required=True)
    gate.add_argument("--output", type=Path)
    generation = commands.add_parser(
        "generate-evaluate",
        help="evaluate deterministic local generation with bounded JSONL prompts",
    )
    generation.add_argument("model")
    generation.add_argument("--dataset", type=Path, required=True)
    generation.add_argument("--output", type=Path)
    generation.add_argument("--max-samples", type=int, default=32)
    generation.add_argument("--max-prompt-tokens", type=int, default=16_384)
    generation.add_argument("--max-new-tokens", type=int, default=32)
    generation.add_argument("--chat-template", action="store_true")
    generation.add_argument(
        "--domain",
        action="append",
        default=[],
        help="include one domain; repeat to select multiple domains",
    )
    generation.add_argument(
        "--language",
        action="append",
        default=[],
        help="include one language; repeat to select multiple languages",
    )
    generation_gate = commands.add_parser(
        "generation-quality-gate",
        help="compare baseline and candidate deterministic generation reports",
    )
    generation_gate.add_argument("--baseline", type=Path, required=True)
    generation_gate.add_argument("--candidate", type=Path, required=True)
    generation_gate.add_argument("--min-token-agreement", type=float, required=True)
    generation_gate.add_argument(
        "--max-expectation-regression",
        type=float,
        required=True,
    )
    generation_gate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "quality-gate":
            baseline = load_perplexity_report(arguments.baseline)
            candidate = load_perplexity_report(arguments.candidate)
            report = compare_perplexity_reports(
                baseline,
                candidate,
                arguments.max_perplexity_regression,
            )
            payload = report.to_dict()
            if arguments.output is not None:
                persist_evaluation_report(payload, arguments.output)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report.approved else 1
        if arguments.command == "generation-quality-gate":
            baseline = load_generation_report(arguments.baseline)
            candidate = load_generation_report(arguments.candidate)
            report = compare_generation_reports(
                baseline,
                candidate,
                minimum_token_agreement=arguments.min_token_agreement,
                maximum_expectation_regression=arguments.max_expectation_regression,
            )
            payload = report.to_dict()
            if arguments.output is not None:
                persist_evaluation_report(payload, arguments.output)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report.approved else 1
        model = inspect_model(arguments.model)
        if arguments.command == "evaluate":
            from .mlx_evaluate import evaluate_mlx_perplexity

            report = evaluate_mlx_perplexity(
                model.path,
                arguments.dataset,
                maximum_samples=arguments.max_samples,
                maximum_tokens_per_sample=arguments.max_tokens_per_sample,
                maximum_total_tokens=arguments.max_total_tokens,
            )
            payload = report.to_dict()
            if arguments.output is not None:
                persist_evaluation_report(payload, arguments.output)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if arguments.command == "generate-evaluate":
            from .mlx_generate_evaluate import evaluate_mlx_generation

            report = evaluate_mlx_generation(
                model.path,
                arguments.dataset,
                maximum_samples=arguments.max_samples,
                maximum_prompt_tokens=arguments.max_prompt_tokens,
                maximum_new_tokens=arguments.max_new_tokens,
                use_chat_template=arguments.chat_template,
                domains=tuple(arguments.domain),
                languages=tuple(arguments.language),
            )
            payload = report.to_dict()
            if arguments.output is not None:
                persist_evaluation_report(payload, arguments.output)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if arguments.command == "capabilities":
            report = builtin_adapter_registry().detect(model)
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if arguments.command == "export":
            if arguments.resume and not arguments.execute:
                raise ValueError("resume requires explicit --execute")
            if not math.isfinite(arguments.max_output_gb) or arguments.max_output_gb <= 0:
                raise ValueError("maximum output size must be finite and positive")
            maximum_output_bytes = int(arguments.max_output_gb * GIB)
            adapter = MLXOptimizationAdapter()
            invocation = adapter.build_export_invocation(
                model,
                arguments.output,
                target_weight_bits=arguments.bits,
                group_size=arguments.group_size,
                maximum_output_bytes=maximum_output_bytes,
                allow_existing_output=arguments.resume,
                require_executable=arguments.execute,
            )
            if not arguments.execute:
                print(
                    json.dumps(
                        invocation.to_dict(),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            from .worker import IsolatedConversionWorker

            result = adapter.execute_export(
                invocation,
                plan_id=arguments.plan_id,
                worker=IsolatedConversionWorker(),
                checkpoint_store=CheckpointStore(arguments.checkpoint_root),
                resume=arguments.resume,
                timeout_seconds=arguments.timeout_seconds,
            )
            if result.state.value != "completed":
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
                return 1
            manifest = adapter.build_artifact_manifest(
                invocation,
                result,
                plan_id=arguments.plan_id,
                license_name=arguments.license,
            )
            manifest_path, persisted_manifest = persist_artifact_manifest(
                manifest,
                arguments.manifest_output
                or arguments.output.with_name(f"{arguments.output.name}.manifest.json"),
                source_path=model.path,
            )
            report = MLXExportReport(result, persisted_manifest, str(manifest_path))
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        hardware = detect_hardware()
        if arguments.command == "profile":
            profile = profile_optimizer_io(
                model.path,
                arguments.workspace,
                hardware,
                arguments.sample_mib * 1024 * 1024,
            )
            print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
            return 0
        memory_budget = (
            int(arguments.max_memory_gb * GIB)
            if arguments.max_memory_gb is not None
            else hardware.memory.available_bytes
        )
        disk_budget = (
            int(arguments.max_disk_gb * GIB)
            if arguments.max_disk_gb is not None
            else shutil.disk_usage(_existing_parent(arguments.output)).free
        )
        plan = build_dry_run_plan(
            model=model,
            hardware=hardware,
            output_path=arguments.output,
            objective=OptimizationObjective(arguments.objective),
            resource_budget=ResourceBudget(
                memory_budget,
                disk_budget,
                arguments.max_duration_seconds,
            ),
            license_name=arguments.license,
            performance_profile=(
                OptimizationPerformanceProfile.from_dict(
                    json.loads(arguments.performance_profile.read_text(encoding="utf-8"))
                )
                if arguments.performance_profile
                else None
            ),
        )
    except (
        ImportError,
        ModelInspectionError,
        OptimizationPathError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        failure = _structured_failure(error, arguments.command)
        print(
            json.dumps(
                {"error": failure.to_dict()},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise ValueError("no existing parent for artifact output")
        candidate = parent
    return candidate


def _structured_failure(error: Exception, command: str) -> OptimizerFailure:
    if isinstance(error, AdapterUnavailableError):
        code = OptimizerErrorCode.ADAPTER_UNAVAILABLE
        message_key = "optimizer.error.adapter_unavailable"
    elif isinstance(error, CheckpointLeaseError):
        code = OptimizerErrorCode.CHECKPOINT_CONFLICT
        message_key = "optimizer.error.checkpoint_conflict"
    elif isinstance(error, ModelInspectionError):
        code = OptimizerErrorCode.INVALID_MODEL
        message_key = "optimizer.error.invalid_model"
    elif isinstance(error, OptimizationPathError):
        code = OptimizerErrorCode.UNSAFE_PATH
        message_key = "optimizer.error.unsafe_path"
    elif isinstance(error, OSError) and command == "profile":
        code = OptimizerErrorCode.PROFILER_FAILED
        message_key = "optimizer.error.profiler_failed"
    elif command in {"evaluate", "generate-evaluate", "generation-quality-gate"}:
        code = OptimizerErrorCode.EVALUATION_FAILED
        message_key = "optimizer.error.evaluation_failed"
    else:
        code = OptimizerErrorCode.INVALID_PLAN
        message_key = "optimizer.error.invalid_plan"
    return OptimizerFailure(
        code=code,
        message_key=message_key,
        recoverability=(
            Recoverability.RETRYABLE
            if code in {
                OptimizerErrorCode.PROFILER_FAILED,
                OptimizerErrorCode.CHECKPOINT_CONFLICT,
            }
            else Recoverability.USER_ACTION_REQUIRED
        ),
        detail=str(error),
    )


if __name__ == "__main__":
    raise SystemExit(main())
