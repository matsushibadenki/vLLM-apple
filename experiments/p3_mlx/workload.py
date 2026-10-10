"""Bounded real Gemma 2 workload for Metal System Trace; no HTTP qualification."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    import sys

    from .conditions import observe_conditions, require_matching_conditions
    from .config import local_identity
    conditions = observe_conditions()
    require_matching_conditions(conditions, conditions)
    before = local_identity(args.model, Path(sys.executable))
    import mlx.core as mx
    import mlx_lm.models.gemma2 as gemma2
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from experiments.p2_mlx.precision import install_norm_weight_reuse
    from vllm_apple.mlx_gemma2_compat import install_gemma2_batch_mask_fix
    install_gemma2_batch_mask_fix()
    mx.set_memory_limit(8*1024**3)
    mx.set_cache_limit(256*1024**2)
    model, _ = load(str(args.model.resolve()))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    install_norm_weight_reuse(gemma2, mx)
    rows = []
    # Warmup precedes the explicitly timed windows. Loading is not inference.
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([[11, 12, 13, 14]]), cache=cache))
    for phase, count in [('prefill', 512), ('decode', 1)]:
        cache = make_prompt_cache(model)
        if phase == 'decode':
            mx.eval(model(mx.array([[11]*512]), cache=cache))
        for index in range(12):
            if phase == 'prefill':
                cache = make_prompt_cache(model)
            cpu = time.process_time_ns()
            start = time.perf_counter_ns()
            logits = model(mx.array([[11]*count]), cache=cache)
            encoded = time.perf_counter_ns()
            mx.eval(logits)
            done = time.perf_counter_ns()
            rows.append(dict(phase=phase, index=index, start_monotonic_ns=start,
                end_monotonic_ns=done, graph_encode_ns=encoded-start,
                evaluation_and_wait_ns=done-encoded, wall_ns=done-start,
                cpu_ns=time.process_time_ns()-cpu, finite=bool(mx.all(mx.isfinite(logits)).item())))
    conditions_after = observe_conditions()
    require_matching_conditions(conditions, conditions_after)
    args.output.write_text(json.dumps(dict(scope='float32 Gemma2; 512-token prefill and batch-one decode; diagnostic overhead, not E2E speed qualification',
        operating_conditions=conditions, conditions_after=conditions_after,
        identity=before, identity_unchanged=before == local_identity(args.model, Path(sys.executable)),
        samples=rows, allocator_active_bytes=mx.get_active_memory(),
        allocator_peak_bytes=mx.get_peak_memory(), allocator_cache_bytes=mx.get_cache_memory(),
        qualification=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
