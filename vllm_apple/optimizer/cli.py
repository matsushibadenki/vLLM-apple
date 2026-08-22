from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ..hardware import detect_hardware
from ..model import ModelInspectionError, inspect_model
from ..types import GIB
from .planner import build_dry_run_plan
from .safety import OptimizationPathError
from .types import OptimizationObjective, ResourceBudget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-apple-optimize")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="create a side-effect-free optimization plan")
    plan.add_argument("model")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--objective", choices=[value.value for value in OptimizationObjective], default="balanced")
    plan.add_argument("--max-memory-gb", type=float)
    plan.add_argument("--max-disk-gb", type=float)
    plan.add_argument("--license")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        model = inspect_model(arguments.model)
        hardware = detect_hardware()
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
            resource_budget=ResourceBudget(memory_budget, disk_budget),
            license_name=arguments.license,
        )
    except (ModelInspectionError, OptimizationPathError, OSError, ValueError) as error:
        print(f"vllm-apple-optimize: {error}", file=sys.stderr)
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


if __name__ == "__main__":
    raise SystemExit(main())
