"""Isolated synthetic Qwen3 block comparison; never a model/P3 qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(policy, repeat, profile=False, preload=False, grouped=False, gather=False, cycles=1,
           warmup_cycles=0, resident_bytes=50000, handshake=False):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    from vllm_apple.expert_execution import ResidentExpertExecutor
    from vllm_apple.expert_residency import ExpertKey, ExpertResidencyManager
    from vllm_apple.expert_timing import ExpertTimings
    from vllm_apple.mlx_expert_backend import MLXFileExpertBackend
    from vllm_apple.mlx_expert_export import export_switch_experts
    from vllm_apple.qwen3_moe_gather import install_qwen3_moe_gather
    from vllm_apple.qwen3_moe_residency import install_qwen3_moe_residency

    mx.set_default_device(mx.cpu)
    mx.random.seed(11)
    block = Qwen3MoeSparseMoeBlock(SimpleNamespace(
        hidden_size=64, moe_intermediate_size=64, num_experts=4,
        num_experts_per_tok=2, norm_topk_prob=True,
    ))
    nn.quantize(block, group_size=32, bits=4)
    base_inputs = [mx.cos(mx.arange(64).reshape(1, 1, 64) + offset)
                   for offset in (0, 1, 0, 2, 0, 1, 0, 3)]
    base_references = [block(x) for x in base_inputs]
    mx.eval(*base_references)
    inputs, references = base_inputs * cycles, base_references * cycles
    manager = None
    gather_hook = None
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
            if gather:
                gather_hook = install_qwen3_moe_gather(
                    block, backend, layer=0, resident_budget_bytes=resident_bytes,
                    startup_budget_bytes=200000,
                )
            else:
                manager = ExpertResidencyManager(
                    backend, maximum_entries=4,
                    maximum_bytes=resident_bytes, eviction_policy=policy,
                )
                executor = ResidentExpertExecutor(manager, timings=timings)
                install_qwen3_moe_residency(
                    block, backend, executor, layer=0, phase="decode", grouped=grouped,
                )
                if preload:
                    for expert in range(4):
                        manager.acquire(ExpertKey(0, expert)).release()
        latencies = []
        errors = []
        request_phases = []
        try:
            for x in base_inputs * warmup_cycles:
                mx.eval(block(x))
            initial_residency = manager.snapshot() if manager else None
            if handshake:
                print("READY", flush=True)
                if sys.stdin.readline().strip() != "GO":
                    raise RuntimeError("missing request start signal")
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
            if handshake:
                print("DONE", flush=True)
                if sys.stdin.readline().strip() != "ACK":
                    raise RuntimeError("missing request completion acknowledgement")
            snapshot = manager.snapshot() if manager else None
            gather_bytes = gather_hook.resident_bytes if gather_hook else None
            if snapshot and (snapshot["active_leases"] != 0
                             or snapshot["peak_resident_bytes"] > resident_bytes):
                raise RuntimeError("resident byte budget or lease contract violated")
        finally:
            if manager:
                manager.close()
            if gather_hook:
                gather_hook.close()
        return dict(policy=policy, repeat=repeat, fixture="synthetic", device="cpu",
                    request_nanoseconds=latencies, first_request_nanoseconds=latencies[0],
                    subsequent_median_nanoseconds=statistics.median(latencies[1:]),
                    maximum_absolute_error=max(errors), parity_passed=max(errors) < 1e-5,
                    residency=snapshot,
                    profiling_enabled=profile, request_phases=request_phases,
                    preloaded=preload, initial_residency=initial_residency,
                    grouped=grouped,
                    gather=gather, cycles=cycles, gather_resident_bytes=gather_bytes,
                    resident_budget_bytes=resident_bytes, warmup_cycles=warmup_cycles,
                    closed_gather_bytes=gather_hook.resident_bytes if gather_hook else None,
                    closed_resident_bytes=manager.snapshot()["resident_bytes"] if manager else None)


def expect_signal(process, expected, timeout=60):
    if not select.select([process.stdout], [], [], timeout)[0]:
        raise TimeoutError(f"worker did not send {expected}")
    signal = process.stdout.readline().strip()
    if signal != expected:
        raise RuntimeError(f"expected {expected}, received {signal!r}")


def phased_trial(command, *, quiet_timeout=30):
    from scripts.strata_measurement import HostLoadMonitor, quiet, wait_for_quiet

    process = subprocess.Popen(command + ["--handshake"], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with HostLoadMonitor(os.getpid()) as startup:
            expect_signal(process, "READY")
        gate = wait_for_quiet(os.getpid(), timeout=quiet_timeout)
        if not gate["passed"]:
            process.terminate()
            process.communicate(timeout=10)
            return None, gate
        with HostLoadMonitor(os.getpid(), interval=0.1) as requests:
            process.stdin.write("GO\n")
            process.stdin.flush()
            expect_signal(process, "DONE")
        process.stdin.write("ACK\n")
        process.stdin.flush()
        stdout, stderr = process.communicate(timeout=60)
        if process.returncode:
            raise RuntimeError(f"comparison worker failed: {stderr[-2000:]}")
        trial = json.loads(stdout)
        def summary(monitor):
            return dict(observations=monitor.observations, sampling_error=monitor.error,
                        passed=bool(monitor.observations) and not monitor.error
                        and all(quiet(o) for o in monitor.observations))
        trial["startup_host_load"] = summary(startup)
        trial["request_start_gate"] = gate
        trial["request_host_load"] = summary(requests)
        trial["host_load"] = trial["request_host_load"]
        return trial, gate
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--handshake", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--policy", choices=("baseline", "lru", "cost_frequency"), default="baseline")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--preload", action="store_true",
                        help="Ablation: keep all four experts resident before request timing")
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--gather", action="store_true")
    parser.add_argument("--variant-suite", action="store_true")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--warmup-cycles", type=int, default=0)
    parser.add_argument("--resident-bytes", type=int, default=50000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--memory-suite", action="store_true")
    parser.add_argument("--quiet-host", action="store_true",
                        help="Require three low-load samples before each worker; record load during it")
    parser.add_argument("--quiet-timeout", type=int, default=30)
    args = parser.parse_args()
    if (not 1 <= args.cycles <= 512 or not 0 <= args.warmup_cycles <= 32
            or not 3 <= args.runs <= 12 or args.resident_bytes <= 0
            or not 3 <= args.quiet_timeout <= 180
            or args.memory_suite and (args.variant_suite or args.preload or args.gather)):
        parser.error("invalid workload limits or incompatible memory-suite options")
    if args.worker:
        print(json.dumps(worker(args.policy, args.repeat, args.profile, args.preload,
                                args.grouped, args.gather, args.cycles,
                                args.warmup_cycles, args.resident_bytes, args.handshake)))
        return
    if args.output is None or args.output.exists():
        parser.error("--output must identify a new report file")
    trials = []
    host_gates = []
    campaign_completed = True
    policies = (("baseline", "baseline"), ("lru", "sequential"),
                ("lru", "grouped"), ("lru", "gather")) if args.variant_suite else (
                    ("baseline", "baseline"), ("lru", "selected"), ("cost_frequency", "selected"))
    if args.memory_suite:
        policies = (("baseline", "baseline"), ("lru", "sequential"),
                    ("lru", "grouped"), ("cost_frequency", "grouped"))
    for repeat in range(args.runs):
        offset = repeat % len(policies)
        for policy, variant in policies[offset:] + policies[:offset]:
            if args.quiet_host:
                from scripts.strata_measurement import wait_for_quiet

                gate = wait_for_quiet(os.getpid(), timeout=args.quiet_timeout)
                gate.update(repeat=repeat, policy=policy, variant=variant)
                host_gates.append(gate)
                if not gate["passed"]:
                    campaign_completed = False
                    break
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--policy", policy,
                       "--repeat", str(repeat), "--cycles", str(args.cycles),
                       "--warmup-cycles", str(args.warmup_cycles),
                       "--resident-bytes", str(args.resident_bytes)]
            if args.profile:
                command.append("--profile")
            if args.preload:
                command.append("--preload")
            if args.grouped or variant == "grouped":
                command.append("--grouped")
            if args.gather or variant == "gather":
                command.append("--gather")
            if args.quiet_host:
                trial, request_gate = phased_trial(command, quiet_timeout=args.quiet_timeout)
                if trial is None:
                    host_gates.append(dict(request_gate, phase="ready_wait", repeat=repeat,
                                           policy=policy, variant=variant))
                    campaign_completed = False
                    break
            else:
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                if result.returncode:
                    raise RuntimeError(f"comparison worker failed: {result.stderr[-2000:]}")
                trial = json.loads(result.stdout)
            trial["variant"] = variant
            if not trial["parity_passed"]:
                raise RuntimeError("synthetic expert parity failed")
            trials.append(trial)
        if not campaign_completed:
            break
    sources = [Path(__file__), ROOT / "scripts/strata_measurement.py",
               *[ROOT / "vllm_apple" / name for name in (
        "qwen3_moe_residency.py", "mlx_expert_backend.py", "mlx_expert_export.py",
        "expert_manifest.py", "expert_residency.py", "expert_execution.py",
        "expert_timing.py",
        "qwen3_moe_gather.py",
    )]]
    report = dict(schema_version=1, scope="synthetic_qwen3_block_cpu",
                  model_qualification=False, standard_adoption_eligible=False,
                  automatic_application=False, trials=trials,
                  profiling_enabled=args.profile,
                  preloaded=args.preload,
                  grouped=args.grouped,
                  gather=args.gather, variant_suite=args.variant_suite, cycles=args.cycles,
                  memory_suite=args.memory_suite, runs=args.runs,
                  warmup_cycles=args.warmup_cycles, resident_budget_bytes=args.resident_bytes,
                  quiet_host_requested=args.quiet_host, host_gates=host_gates,
                  host_load_window="requests_only" if args.quiet_host else None,
                  quiet_timeout_seconds=args.quiet_timeout,
                  campaign_completed=campaign_completed,
                  host_load_rule="3 consecutive 1s samples: total<=200%, max process<=75%, no PID churn",
                  notes=["Export and model initialization excluded from request timing",
                         "Host gate checks sampled CPU counters, not disk/GPU contention",
                         "Aggregate CPU includes benchmark; per-process gate excludes worker tree",
                         "Startup includes import/export/preload/warmup; ready wait is separate",
                         "Request host window includes parity checks; individual latencies exclude them",
                         "Request host sampling uses approx 100ms cadence; startup uses approx 1s",
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
                      "all_parity_passed": bool(trials) and all(t["parity_passed"] for t in trials),
                      "campaign_completed": campaign_completed,
                      "standard_adoption_eligible": False}))


if __name__ == "__main__":
    main()
