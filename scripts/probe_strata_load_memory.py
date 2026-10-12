"""Fresh-process CPU/Metal loading peaks on a bounded synthetic expert fixture."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURE_ID = hashlib.sha256(b"strata-memory-d1024-i2048-e4-q4-g32-seed11").hexdigest()


def prepare(root):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    from vllm_apple.mlx_expert_export import export_switch_experts

    mx.set_default_device(mx.cpu)
    mx.random.seed(11)
    block = Qwen3MoeSparseMoeBlock(SimpleNamespace(
        hidden_size=1024, moe_intermediate_size=2048, num_experts=4,
        num_experts_per_tok=2, norm_topk_prob=True,
    ))
    nn.quantize(block, bits=4, group_size=32)
    export_switch_experts({0: block.switch_mlp}, root, model_sha256=FIXTURE_ID,
                          maximum_expert_bytes=8 << 20, maximum_total_bytes=32 << 20)


def worker(root, device, mode, capacity):
    import mlx.core as mx
    import psutil

    from scripts.strata_measurement import RSSSampler
    from vllm_apple.expert_residency import ExpertKey, ExpertResidencyManager
    from vllm_apple.mlx_expert_backend import MLXFileExpertBackend

    if device == "gpu" and not mx.metal.is_available():
        return dict(device=device, status="unavailable")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    backend = MLXFileExpertBackend.from_manifest(
        root, expected_model_sha256=FIXTURE_ID, maximum_file_bytes=8 << 20,
    )
    payload = backend.resident_bytes_upper_bound(ExpertKey(0, 0))
    if mode == "postload":
        # Control deliberately omits the optional reservation capability.
        adapter = SimpleNamespace(load_expert=backend.load_expert,
                                  release_expert=backend.release_expert)
    else:
        adapter = backend
    manager = ExpertResidencyManager(adapter, maximum_entries=4, maximum_bytes=payload * capacity)
    process = psutil.Process()
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    steps = []
    try:
        for expert in (0, 1, 2, 3, 0, 1):
            mx.synchronize()
            before = dict(rss_bytes=process.memory_info().rss,
                          process_hwm_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory())
            mx.reset_peak_memory()
            with RSSSampler() as sampler:
                lease = manager.acquire(ExpertKey(0, expert))
                try:
                    assert lease.resource.resident_bytes == payload
                    mx.synchronize()
                finally:
                    lease.release()
            if sampler.error:
                raise RuntimeError(f"RSS sampler failed: {sampler.error}")
            steps.append(dict(expert=expert, before=before,
                              sampled_rss_peak_bytes=sampler.peak, rss_samples=sampler.samples,
                              process_hwm_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              mlx_peak_active_bytes=mx.get_peak_memory(),
                              mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
                              residency=manager.snapshot()))
    finally:
        manager.close()
    mx.synchronize()
    gc.collect()
    closed = dict(residency=manager.snapshot(), rss_bytes=process.memory_info().rss,
                  mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory())
    mx.clear_cache()
    return dict(status="passed", device=device, mode=mode, capacity=capacity,
                mlx_version=importlib.metadata.version("mlx"),
                metal_device_info=mx.metal.device_info() if mx.metal.is_available() else None,
                payload_bytes=payload, budget_bytes=payload * capacity, steps=steps, closed=closed,
                after_clear_cache=dict(rss_bytes=process.memory_info().rss,
                                       mlx_active_bytes=mx.get_active_memory(),
                                       mlx_cache_bytes=mx.get_cache_memory()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--mode", choices=("preload", "postload"), default="preload")
    parser.add_argument("--capacity", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
        return
    if args.worker:
        print(json.dumps(worker(args.worker, args.device, args.mode, args.capacity)))
        return
    if args.output is None or args.output.exists() or sys.platform != "darwin":
        parser.error("new --output path and macOS required (ru_maxrss is bytes)")
    trials = []
    script = str(Path(__file__).resolve())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "experts"
        subprocess.run([sys.executable, script, "--prepare", str(root)], check=True, timeout=60)
        for repeat in range(2):
            jobs = [(device, mode, capacity) for device in ("cpu", "gpu")
                    for capacity in (1, 2) for mode in ("postload", "preload")]
            for device, mode, capacity in jobs[repeat:] + jobs[:repeat]:
                result = subprocess.run([sys.executable, script, "--worker", str(root),
                                         "--device", device, "--mode", mode,
                                         "--capacity", str(capacity)],
                                        capture_output=True, text=True, timeout=60, check=True)
                trial = json.loads(result.stdout)
                trial["repeat"] = repeat
                trials.append(trial)
    sources = [Path(__file__), ROOT / "scripts/strata_measurement.py",
               ROOT / "vllm_apple/expert_residency.py", ROOT / "vllm_apple/mlx_expert_backend.py"]
    report = dict(scope="synthetic_expert_loading_cpu_metal", trials=trials,
                  real_oom_qualified=False, speed_qualified=False,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources},
                  notes=["Fixture exported in another process; workers retain no source model",
                         "Current RSS sampled at requested 1ms interval; short peaks may be missed",
                         "ru_maxrss is process lifetime high-water, not an isolated load peak",
                         "MLX peak reset per acquire; synchronization surrounds each load",
                         "MLX statistics cover process allocator, not total GPU/OS memory",
                         "GPU workers select Metal as default device; no inference kernels measured",
                         "Apple Silicon uses unified memory; RSS and MLX values overlap",
                         "Default allocator cache retained between loads; cleared at start/end",
                         "Checksums, file loading and weight eval included; inference excluded",
                         "Other host workloads may affect RSS sample cadence; no speed claim"])
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(report=str(args.output), trials=len(trials))))


if __name__ == "__main__":
    main()
