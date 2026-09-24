#!/usr/bin/env python3
"""Qualify Qwen prompt encoding and denoising in disjoint processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.encode_mflux_qwen_prompt_phase import PROMPTS  # noqa: E402
from scripts.inspect_mflux_qwen_streaming import _deployable_tree_identity  # noqa: E402
from vllm_apple.hardware import detect_hardware  # noqa: E402
from vllm_apple.mflux_qwen_promotion import load_mflux_qwen_promotion  # noqa: E402
from vllm_apple.mflux_qwen_prompt_handoff import (  # noqa: E402
    MANIFEST_NAME,
    PAYLOAD_NAME,
)
from vllm_apple.mflux_qwen_streaming_plan import (  # noqa: E402
    inspect_mflux_qwen_text_encoder_staging,
)
from vllm_apple.qualification import save_qualification_report  # noqa: E402


def _run_bounded(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if len(result.stdout.encode()) > 64 * 1024 or len(result.stderr.encode()) > 64 * 1024:
        raise RuntimeError("Qwen staged child output exceeds the bounded limit")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(32, 64, 128, 160, 192, 256, 512), default=32)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--promotion-baseline-report", type=Path)
    arguments = parser.parse_args()
    if not 2 <= arguments.steps <= 20:
        parser.error("--steps must be in 2..20")
    if not 1.0 <= arguments.guidance <= 10.0:
        parser.error("--guidance must be in 1..10")
    root = arguments.model.expanduser().resolve(strict=True)
    plan = inspect_mflux_qwen_text_encoder_staging(root)
    root_digest, file_count, artifact_bytes = _deployable_tree_identity(root)
    before = detect_hardware()
    minimum_available = plan.static_plus_maximum_layer_bytes * 4
    if arguments.size >= 512:
        minimum_available = max(minimum_available, 20_000_000_000)
    elif arguments.size >= 256:
        minimum_available = max(minimum_available, 14_000_000_000)
    elif arguments.size >= 192:
        minimum_available = max(minimum_available, 12_000_000_000)
    elif arguments.size >= 160:
        minimum_available = max(minimum_available, 11_000_000_000)
    elif arguments.size >= 128:
        minimum_available = max(minimum_available, 10_000_000_000)
    promotion = None
    if arguments.promotion_baseline_report is not None:
        promotion = load_mflux_qwen_promotion(
            arguments.promotion_baseline_report,
            artifact_root_sha256=root_digest,
            target_size=arguments.size,
        )
        minimum_available = min(minimum_available, promotion.minimum_available_bytes)
    if (
        before.memory.pressure.value != "normal"
        or before.memory.available_bytes < minimum_available
    ):
        raise RuntimeError("Qwen staged qualification rejected before weight load")
    plan_sha256 = hashlib.sha256(
        f"qwen-image-2512-mflux|{root_digest}|text-to-image|handoff-v1".encode()
    ).hexdigest()
    prompt_hashes = {
        role: hashlib.sha256(prompt.encode()).hexdigest() for role, prompt in PROMPTS.items()
    }
    workspace = Path(__file__).resolve().parents[1]
    phase_script = Path(__file__).with_name("encode_mflux_qwen_prompt_phase.py")
    denoise_script = Path(__file__).with_name("qualify_mflux_qwen_streamed_denoise.py")
    started = time.perf_counter_ns()
    encoder_reports: list[dict[str, object]] = []
    transformer_report: dict[str, object] = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_streamed_denoising_smoke",
        "passed": False,
        "failure": "generation_not_started",
    }
    child_returncode = None
    child_failure = None
    child_consumed_handoff_cleanup_verified = False

    with tempfile.TemporaryDirectory(prefix="qwen2512-staged-", dir=workspace) as directory:
        private_root = Path(directory)
        roles = ["positive"] + (["negative"] if arguments.guidance > 1.0 else [])
        manifests: dict[str, Path] = {}
        for role in roles:
            handoff_root = private_root / role
            handoff_root.mkdir(mode=0o700)
            phase = _run_bounded(
                [
                    sys.executable,
                    str(phase_script),
                    str(root),
                    "--handoff-root",
                    str(handoff_root),
                    "--role",
                    role,
                ],
                timeout=180,
            )
            if phase.returncode:
                child_failure = f"{role}_encoder_phase_failed"
                break
            phase_report = json.loads(phase.stdout)
            if (
                phase_report.get("passed") is not True
                or phase_report.get("role") != role
                or phase_report.get("plan_sha256") != plan_sha256
                or phase_report.get("prompt_sha256") != prompt_hashes[role]
                or phase_report.get("layer_execution_count") != 28
            ):
                child_failure = f"{role}_encoder_phase_contract_failed"
                break
            encoder_reports.append(phase_report)
            manifests[role] = handoff_root / MANIFEST_NAME

        if child_failure is None:
            child_args = [
                sys.executable,
                str(denoise_script),
                str(root),
                "--manifest",
                str(manifests["positive"]),
                "--plan-sha256",
                plan_sha256,
                "--prompt-sha256",
                prompt_hashes["positive"],
                "--size",
                str(arguments.size),
                "--steps",
                str(arguments.steps),
                "--guidance",
                str(arguments.guidance),
            ]
            if "negative" in manifests:
                child_args.extend(
                    [
                        "--negative-manifest",
                        str(manifests["negative"]),
                        "--negative-prompt-sha256",
                        prompt_hashes["negative"],
                    ]
                )
            if arguments.output is not None:
                child_args.extend(["--output", str(arguments.output)])
            if arguments.promotion_baseline_report is not None:
                child_args.extend(
                    [
                        "--promotion-baseline-report",
                        str(arguments.promotion_baseline_report),
                    ]
                )
            child_report_path = private_root / "transformer-report.json"
            child_args.extend(["--report", str(child_report_path)])
            child = _run_bounded(child_args, timeout=900)
            child_returncode = child.returncode
            if child_report_path.is_file():
                transformer_report = json.loads(child_report_path.read_text(encoding="utf-8"))
            if child.returncode or not child_report_path.is_file():
                child_failure = (
                    "child_memory_admission_rejected_before_weight_load"
                    if "rejected before weight load" in child.stderr
                    else "generation_phase_failed"
                )
            child_consumed_handoff_cleanup_verified = all(
                not manifest.exists() and not manifest.with_name(PAYLOAD_NAME).exists()
                for manifest in manifests.values()
            )
    private_handoff_cleanup_verified = not private_root.exists()
    after = detect_hardware()
    expected_encoder_phases = 2 if arguments.guidance > 1.0 else 1
    passed = (
        child_failure is None
        and len(encoder_reports) == expected_encoder_phases
        and transformer_report.get("passed") is True
        and len(transformer_report.get("step_reports", [])) == arguments.steps
        and transformer_report.get("decoded_shape") == [1, 3, 1, arguments.size, arguments.size]
        and transformer_report.get("uses_real_prompt") is True
        and transformer_report.get("prompt_sha256") == prompt_hashes["positive"]
        and child_consumed_handoff_cleanup_verified
        and private_handoff_cleanup_verified
        and after.memory.pressure.value == "normal"
        and after.thermal_state.value in {"nominal", "fair"}
    )
    report = {
        "schema_version": 1,
        "scope": "mflux_qwen_image_staged_real_prompt_streamed_denoising",
        "candidate_id": "qwen-image-2512",
        "artifact_root_sha256": root_digest,
        "artifact_file_count": file_count,
        "artifact_bytes": artifact_bytes,
        "promotion_baseline_sha256": promotion.report_sha256 if promotion else None,
        "promotion_baseline_size": promotion.baseline_size if promotion else None,
        "minimum_available_bytes": minimum_available,
        "plan_sha256": plan_sha256,
        "prompt_sha256": prompt_hashes["positive"],
        "negative_prompt_sha256": prompt_hashes["negative"] if arguments.guidance > 1.0 else None,
        "encoder_phases": encoder_reports,
        "encoder_layer_execution_count": sum(
            int(item["layer_execution_count"]) for item in encoder_reports
        ),
        "encoder_peak_mlx_bytes": max(
            (int(item["peak_mlx_bytes"]) for item in encoder_reports), default=0
        ),
        "encoder_peak_process_rss_bytes": max(
            (int(item["peak_process_rss_bytes"]) for item in encoder_reports), default=0
        ),
        "transformer": transformer_report,
        "child_returncode": child_returncode,
        "child_failure": child_failure,
        "child_consumed_handoff_cleanup_verified": child_consumed_handoff_cleanup_verified,
        "private_handoff_cleanup_verified": private_handoff_cleanup_verified,
        "elapsed_nanoseconds": max(1, time.perf_counter_ns() - started),
        "initial_memory_pressure": before.memory.pressure.value,
        "final_memory_pressure": after.memory.pressure.value,
        "final_thermal_state": after.thermal_state.value,
        "denoise_steps": arguments.steps,
        "guidance": arguments.guidance,
        "uses_disjoint_encoder_processes": True,
        "uses_real_prompt": True,
        "uses_true_cfg": arguments.guidance > 1.0,
        "width": arguments.size,
        "height": arguments.size,
        "stores_prompt": False,
        "stores_output": transformer_report.get("stores_output") is True,
        "output_sha256": transformer_report.get("output_sha256"),
        "image_generation_qualified": False,
        "passed": passed,
    }
    save_qualification_report(report, arguments.report)
    print(
        json.dumps(
            {
                "passed": passed,
                "encoder_phase_count": len(encoder_reports),
                "transformer_completed_blocks": sum(
                    step.get("completed_blocks", 0)
                    for step in transformer_report.get("step_reports", [])
                ),
                "child_failure": child_failure,
                "private_handoff_cleanup_verified": private_handoff_cleanup_verified,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
