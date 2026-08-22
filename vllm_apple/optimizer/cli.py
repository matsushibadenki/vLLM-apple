from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ..hardware import detect_hardware
from ..model import ModelInspectionError, inspect_model
from ..types import GIB
from .errors import OptimizerErrorCode, OptimizerFailure, Recoverability
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
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        model = inspect_model(arguments.model)
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
    except (ModelInspectionError, OptimizationPathError, OSError, ValueError) as error:
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
    if isinstance(error, ModelInspectionError):
        code = OptimizerErrorCode.INVALID_MODEL
        message_key = "optimizer.error.invalid_model"
    elif isinstance(error, OptimizationPathError):
        code = OptimizerErrorCode.UNSAFE_PATH
        message_key = "optimizer.error.unsafe_path"
    elif isinstance(error, OSError) and command == "profile":
        code = OptimizerErrorCode.PROFILER_FAILED
        message_key = "optimizer.error.profiler_failed"
    else:
        code = OptimizerErrorCode.INVALID_PLAN
        message_key = "optimizer.error.invalid_plan"
    return OptimizerFailure(
        code=code,
        message_key=message_key,
        recoverability=(
            Recoverability.RETRYABLE
            if code == OptimizerErrorCode.PROFILER_FAILED
            else Recoverability.USER_ACTION_REQUIRED
        ),
        detail=str(error),
    )


if __name__ == "__main__":
    raise SystemExit(main())
