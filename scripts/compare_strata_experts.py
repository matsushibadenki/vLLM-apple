"""Isolated synthetic Qwen3 block comparison; never a model/P3 qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(policy, repeat, profile=False, preload=False, grouped=False):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    from vllm_apple.expert_execution import ResidentExpertExecutor
    from vllm_apple.expert_residency import ExpertKey, ExpertResidencyManager
    from vllm_apple.expert_timing import ExpertTimings
    from vllm_apple.mlx_expert_backend import MLXFileExpertBackend
    from vllm_apple.mlx_expert_export import export_switch_experts
    from vllm_apple.qwen3_moe_residency import install_qwen3_moe_residency

    mx.set_default_device(mx.cpu)
    mx.random.seed(11)
    block = Qwen3MoeSparseMoeBlock(SimpleNamespace(
        hidden_size=64, moe_intermediate_size=64, num_experts=4,
        num_experts_per_tok=2, norm_topk_prob=True,
    ))
    nn.quantize(block, group_size=32, bits=4)
    inputs = [mx.cos(mx.arange(64).reshape(1, 1, 64) + offset)
              for offset in (0, 1, 0, 2, 0, 1, 0, 3)]
    references = [block(x) for x in inputs]
    mx.eval(*references)
    manager = None
    timings = ExpertTimings() if profile else None
    with tempfile.TemporaryDirectory() as directory:
        if policy != "baseline":
            root = Path(directory) / "experts"
            fixture_id = hashlib.sha256(b"synthetic-qwen3-seed11-d64-e4-top2-q4-g32").hexdigest()
            export_switch_experts(
                {0: block.switch_mlp}, root, model_sha256=fixture_id,
                maximum_expert_bytes=65536, maximum_total_bytes=262144,
            )
            backend = MLXFileExpertBackend.from_manifest(
                root, expected_model_sha256=fixture_id, maximum_file_bytes=65536, timings=timings,
            )
            manager = ExpertResidencyManager(
                backend, maximum_entries=4 if preload else 2,
                maximum_bytes=50000, eviction_policy=policy,
            )
            executor = ResidentExpertExecutor(manager, timings=timings)
            install_qwen3_moe_residency(
                block, backend, executor, layer=0, phase="decode", grouped=grouped,
            )
            if preload:
                for expert in range(4):
                    manager.acquire(ExpertKey(0, expert)).release()
        initial_residency = manager.snapshot() if manager else None
        latencies = []
        errors = []
        request_phases = []
        try:
            for x, reference in zip(inputs, references):
                before = timings.snapshot() if timings else None
                started = time.perf_counter_ns()
                actual = block(x)
                mx.eval(actual)
                latencies.append(time.perf_counter_ns() - started)
                if timings:
                    after = timings.snapshot()
                    request_phases.append({phase: {
                        field: after[phase][field] - before[phase][field]
                        for field in ("count", "nanoseconds")
                    } for phase in after})
                errors.append(float(mx.max(mx.abs(actual - reference)).item()))
            snapshot = manager.snapshot() if manager else None
        finally:
            if manager:
                manager.close()
        return dict(policy=policy, repeat=repeat, fixture="synthetic", device="cpu",
                    request_nanoseconds=latencies, first_request_nanoseconds=latencies[0],
                    subsequent_median_nanoseconds=statistics.median(latencies[1:]),
                    maximum_absolute_error=max(errors), parity_passed=max(errors) < 1e-5,
                    residency=snapshot,
                    profiling_enabled=profile, request_phases=request_phases,
                    preloaded=preload, initial_residency=initial_residency,
                    grouped=grouped,
                    closed_resident_bytes=manager.snapshot()["resident_bytes"] if manager else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--policy", choices=("baseline", "lru", "cost_frequency"), default="baseline")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--preload", action="store_true",
                        help="Ablation: keep all four experts resident before request timing")
    parser.add_argument("--grouped", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.policy, args.repeat, args.profile, args.preload, args.grouped)))
        return
    if args.output is None or args.output.exists():
        parser.error("--output must identify a new report file")
    trials = []
    policies = ("baseline", "lru", "cost_frequency")
    for repeat in range(3):
        for policy in policies[repeat:] + policies[:repeat]:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--policy", policy,
                       "--repeat", str(repeat)]
            if args.profile:
                command.append("--profile")
            if args.preload:
                command.append("--preload")
            if args.grouped:
                command.append("--grouped")
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=60,
            )
            if result.returncode:
                raise RuntimeError(f"comparison worker failed: {result.stderr[-2000:]}")
            trial = json.loads(result.stdout)
            if not trial["parity_passed"]:
                raise RuntimeError("synthetic expert parity failed")
            trials.append(trial)
    sources = [Path(__file__), *[ROOT / "vllm_apple" / name for name in (
        "qwen3_moe_residency.py", "mlx_expert_backend.py", "mlx_expert_export.py",
        "expert_manifest.py", "expert_residency.py", "expert_execution.py",
        "expert_timing.py",
    )]]
    report = dict(schema_version=1, scope="synthetic_qwen3_block_cpu",
                  model_qualification=False, standard_adoption_eligible=False,
                  automatic_application=False, trials=trials,
                  profiling_enabled=args.profile,
                  preloaded=args.preload,
                  grouped=args.grouped,
                  notes=["Export and model initialization excluded from request timing",
                         "OS file cache is not reset; this is not cold-disk performance",
                         "Baseline has resident source weights; candidates use file residency",
                         "Phase spans are host wall times; acquire includes nested loading spans",
                         "Router eval includes gate/top-k execution, not just a synchronization wait",
                         "MLX load and eval may include file I/O; checksum_read is separate",
                         "Opt-in instrumentation overhead is included in request timings",
                         "No RSS, GPU, E2E chat, quality-suite or P3 certification"],
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources})
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(args.output), "trials": len(trials),
                      "all_parity_passed": True, "standard_adoption_eligible": False}))


if __name__ == "__main__":
    main()
