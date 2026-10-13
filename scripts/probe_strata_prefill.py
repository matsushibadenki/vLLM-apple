"""Diverse authoritative prefill fixtures; speed and memory use separate workers."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IDENTITY = hashlib.sha256(b"strata-prefill-d512-i1024-e4-q4-g32-seed11").hexdigest()
CASES = ("repeat", "mixed_runs", "alternating", "reversed", "random")


def selections(name, rows):
    if name == "repeat":
        return [(0, 1)] * rows
    if name == "alternating":
        return [(0, 1) if i % 2 == 0 else (2, 3) for i in range(rows)]
    if name == "reversed":
        return [(0, 1) if i % 2 == 0 else (1, 0) for i in range(rows)]
    if name == "random":
        rng = random.Random(11)
        return [tuple(rng.sample(range(4), 2)) for _ in range(rows)]
    if name == "mixed_runs":
        result = []
        while len(result) < rows:
            for i, length in enumerate((1, 2, 3, 5, 8, 13)):
                result.extend([(0, 1) if i % 2 == 0 else (2, 3)] * length)
        return result[:rows]
    raise ValueError("unknown fixture")


def arguments():
    return SimpleNamespace(hidden_size=512, moe_intermediate_size=1024,
                           num_experts=4, num_experts_per_tok=2, norm_topk_prob=True)


def prepare(root):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    from vllm_apple.mlx_expert_export import export_switch_experts

    mx.set_default_device(mx.cpu)
    mx.random.seed(11)
    block = Qwen3MoeSparseMoeBlock(arguments())
    nn.quantize(block, bits=4, group_size=32)
    export_switch_experts({0: block.switch_mlp}, root / "experts", model_sha256=IDENTITY,
                          maximum_expert_bytes=8 << 20, maximum_total_bytes=32 << 20)
    for name in CASES:
        rows = 32 if name == "reversed" else 128
        x = mx.sin(mx.arange(rows * 512).reshape(1, rows, 512) * 0.013)
        if name == "random":
            x = mx.random.normal((1, rows, 512))
        indices = mx.array([selections(name, rows)])
        expected = block.switch_mlp(x, indices)
        mx.eval(x, indices, expected)
        mx.save_safetensors(str(root / f"{name}.safetensors"),
                            dict(x=x, indices=indices, expected=expected))


def worker(root, device, group_rows, measure, repeat, exact_payload_budget=False):
    import mlx.core as mx
    import mlx.nn as nn
    import psutil
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    from scripts.strata_measurement import RSSSampler
    from vllm_apple.expert_execution import ResidentExpertExecutor
    from vllm_apple.expert_residency import ExpertKey, ExpertResidencyManager
    from vllm_apple.expert_timing import ExpertTimings
    from vllm_apple.mlx_expert_backend import MLXFileExpertBackend
    from vllm_apple.qwen3_moe_residency import install_qwen3_moe_residency

    if device == "gpu" and not mx.metal.is_available():
        return dict(status="unavailable", device=device)
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    results = []
    order = CASES[repeat % len(CASES):] + CASES[:repeat % len(CASES)]
    for name in order:
        tensors = mx.load(str(root / f"{name}.safetensors"))
        x, indices, expected = (tensors[key] for key in ("x", "indices", "expected"))
        mx.eval(x, indices, expected)
        block = Qwen3MoeSparseMoeBlock(arguments())
        nn.quantize(block, bits=4, group_size=32)
        timings = ExpertTimings() if measure == "memory" else None
        backend = MLXFileExpertBackend.from_manifest(
            root / "experts", expected_model_sha256=IDENTITY,
            maximum_file_bytes=8 << 20, timings=timings,
        )
        payload = backend.resident_bytes_upper_bound(ExpertKey(0, 0))
        budget = (2 * payload if exact_payload_budget else
                  2 * max(size for size, _ in backend._manifest.entries.values()))
        if not payload * 2 <= budget < payload * 3:
            raise RuntimeError("fixture does not enforce two-expert partial residency")
        manager = ExpertResidencyManager(backend, maximum_entries=4, maximum_bytes=budget)
        executor = ResidentExpertExecutor(manager, timings=timings)
        hook = install_qwen3_moe_residency(block, backend, executor, layer=0,
                                         phase="prefill", grouped=True,
                                         maximum_group_rows=group_rows)
        gc.collect()
        mx.synchronize()
        if measure == "memory":
            mx.clear_cache()
        latencies, errors, memory = [], [], []
        try:
            if measure == "speed":
                mx.eval(hook(x, indices))
            for iteration in range(4 if measure == "speed" else 2):
                if measure == "memory":
                    process = psutil.Process()
                    before = dict(rss_bytes=process.memory_info().rss,
                                  mlx_active_bytes=mx.get_active_memory(),
                                  mlx_cache_bytes=mx.get_cache_memory())
                    phases_before = timings.snapshot()
                    mx.reset_peak_memory()
                    with RSSSampler() as sampler:
                        actual = hook(x, indices)
                        mx.eval(actual)
                        mx.synchronize()
                    if sampler.error:
                        raise RuntimeError(sampler.error)
                    phases_after = timings.snapshot()
                    memory.append(dict(iteration=iteration, before=before,
                                       sampled_rss_peak_bytes=sampler.peak,
                                       rss_samples=sampler.samples,
                                       process_hwm_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                       mlx_peak_active_bytes=mx.get_peak_memory(),
                                       mlx_cache_bytes=mx.get_cache_memory(),
                                       counts={key: phases_after[key]["count"] - phases_before[key]["count"]
                                               for key in phases_after}))
                else:
                    started = time.perf_counter_ns()
                    actual = hook(x, indices)
                    mx.eval(actual)
                    latencies.append(time.perf_counter_ns() - started)
                error = float(mx.max(mx.abs(actual - expected)).item())
                errors.append(error)
                if error >= 1e-5:
                    raise RuntimeError(f"{name}: parity failed: {error}")
                del actual
            snapshot = manager.snapshot()
            if (snapshot["active_leases"] or snapshot["peak_reserved_bytes"] > budget):
                raise RuntimeError("residency contract failed")
        finally:
            manager.close()
        results.append(dict(case=name, rows=x.shape[1], latencies_ns=latencies,
                            maximum_absolute_error=max(errors), memory=memory,
                            residency=snapshot, budget_bytes=budget, payload_bytes=payload,
                            closed_resident_bytes=manager.snapshot()["resident_bytes"]))
        del tensors, x, indices, expected, hook, block, executor, manager, backend
        gc.collect()
        mx.synchronize()
        if measure == "memory":
            mx.clear_cache()
    return dict(status="passed", device=device, maximum_group_rows=group_rows,
                measure=measure, repeat=repeat, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--group-rows", type=int, choices=(1, 8), default=1)
    parser.add_argument("--measure", choices=("speed", "memory"), default="speed")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--exact-payload-budget", action="store_true")
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
        return
    if args.worker:
        print(json.dumps(worker(args.worker, args.device, args.group_rows, args.measure, args.repeat,
                                args.exact_payload_budget)))
        return
    if args.output is None or args.output.exists() or sys.platform != "darwin":
        parser.error("new --output and macOS required")
    script = str(Path(__file__).resolve())
    trials = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        subprocess.run([sys.executable, script, "--prepare", str(root)], check=True, timeout=90)
        campaigns = (("memory", ("cpu", "gpu"), 2),) if args.memory_only else (
            ("speed", ("cpu",), 3), ("memory", ("cpu", "gpu"), 2))
        for measure, devices, repeats in campaigns:
            for repeat in range(repeats):
                for device in devices:
                    for rows in ((1, 8) if repeat % 2 == 0 else (8, 1)):
                        result = subprocess.run([sys.executable, script, "--worker", str(root),
                                                 "--device", device, "--group-rows", str(rows),
                                                 "--measure", measure, "--repeat", str(repeat),
                                                 *(["--exact-payload-budget"]
                                                   if args.exact_payload_budget else [])],
                                                capture_output=True, text=True, timeout=120)
                        if result.returncode:
                            raise RuntimeError(result.stderr[-3000:])
                        trials.append(json.loads(result.stdout))
    sources = [Path(__file__), ROOT / "vllm_apple/qwen3_moe_residency.py",
               ROOT / "vllm_apple/mlx_expert_backend.py", ROOT / "vllm_apple/expert_residency.py",
               ROOT / "scripts/strata_measurement.py"]
    report = dict(scope="synthetic_authoritative_prefill_switch_cpu_metal", trials=trials,
                  speed_qualified=False, real_oom_qualified=False,
                  exact_payload_budget=args.exact_payload_budget, memory_only=args.memory_only,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources},
                  notes=["Five synthetic ordered-selection fixtures, not text/tokenizer/real-model quality",
                         ("Budget is exactly two payloads" if args.exact_payload_budget else
                          "Budget includes two safetensors files and excludes a third payload"),
                         "Native SwitchGLU references and experts prepared in a separate process",
                         "Workers discard temporary source bank before measurement",
                         "4 measured CPU requests after 1 warmup per case; no host qualification",
                         "Memory workers separate from speed; phase counters enabled only for memory",
                         "2 memory requests per case: first cold manager, then existing cache/residency",
                         "Reference/input/output arrays contribute to baseline and observed peaks",
                         "RSS sampling requested at 1ms; sampled peak may miss transients",
                         "RSS HWM is lifetime-wide; MLX peak reset per request",
                         "MLX allocator is process-wide unified memory, not dedicated/system GPU VRAM",
                         "Keep default allocator cache between requests; test-only clear between memory cases"])
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(report=str(args.output), trials=len(trials))))


if __name__ == "__main__":
    main()
