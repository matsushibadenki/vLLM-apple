"""Promote strict heterogeneous benchmark reports into a placement plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .device_benchmark import load_profile_device_benchmark
from .device_placement import (
    build_device_placement_plan,
    default_device_placement_paths,
    promote_device_placement_plan,
)
from .device_selection import select_measured_device_backend
from .execution import ExecutionBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-apple-device-placement")
    parser.add_argument("--hardware-fingerprint", required=True)
    parser.add_argument("--environment-fingerprint", required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, action="append", required=True)
    parser.add_argument("--current-plan", type=Path)
    parser.add_argument("--last-known-good-plan", type=Path)
    parser.add_argument("--application-support", type=Path)
    parser.add_argument("--ttl-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--cold-load-amortization-runs", type=int, default=1)
    parser.add_argument("--minimum-improvement-ratio", type=float, default=0.05)
    parser.add_argument("--require-peak-memory", action="store_true")
    parser.add_argument("--language", choices=("en", "ja", "zh-Hans"), default="en")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    reports = tuple(
        load_profile_device_benchmark(
            path,
            hardware_fingerprint=arguments.hardware_fingerprint,
            environment_fingerprint=arguments.environment_fingerprint,
        )
        for path in (arguments.baseline_report, *arguments.candidate_report)
    )
    if reports[0].config.backend is not ExecutionBackend.CPU:
        raise ValueError("device placement baseline report must use CPU")
    decision = select_measured_device_backend(
        reports,
        minimum_improvement_ratio=arguments.minimum_improvement_ratio,
        cold_load_amortization_runs=arguments.cold_load_amortization_runs,
        require_peak_memory=arguments.require_peak_memory,
    )
    plan = build_device_placement_plan(
        reports,
        decision,
        ttl_seconds=arguments.ttl_seconds,
    )
    defaults = default_device_placement_paths(
        arguments.hardware_fingerprint,
        arguments.environment_fingerprint,
        application_support=arguments.application_support,
    )
    current = arguments.current_plan or defaults[0]
    last_good = arguments.last_known_good_plan or defaults[1]
    if (arguments.current_plan is None) != (arguments.last_known_good_plan is None):
        raise ValueError("custom placement paths must be provided together")
    promote_device_placement_plan(plan, current, last_good)
    messages = {
        "en": "Device placement plan promoted.",
        "ja": "デバイス配置プランを昇格しました。",
        "zh-Hans": "设备调度方案已升级。",
    }
    print(json.dumps({
        "plan_id": plan.plan_id,
        "selected_backend": decision.selected.value,
        "baseline_backend": decision.baseline.value,
        "improvement_ratio": decision.improvement_ratio,
        "current_plan": str(current),
        "last_known_good_plan": str(last_good),
        "message_key": "device_placement_plan_promoted",
        "message": messages[arguments.language],
        "language": arguments.language,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
