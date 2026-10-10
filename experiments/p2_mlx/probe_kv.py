"""Real-model cache cloning, exact-prefix reuse and logit parity probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compute-dtype', choices=('float16', 'float32'), default='float32')
    parser.add_argument('--disable-norm-reuse', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache

    from vllm_apple.mlx_gemma2_compat import install_gemma2_batch_mask_fix

    from .cache import IdentityPromptCache
    from .server import _sha

    install_gemma2_batch_mask_fix()
    model, tokenizer = load(args.model)
    model.set_dtype(getattr(mx, args.compute_dtype))
    mx.eval(model.parameters())
    if not args.disable_norm_reuse:
        from mlx_lm.models import gemma2

        from .precision import install_norm_weight_reuse
        install_norm_weight_reuse(gemma2, mx)
    tokens = tokenizer.encode('The numbers are 1, 2, 3, 4, 5. Continue the sequence:')
    prefix, suffix = tokens[:-4], tokens[-4:]
    owner_cache = make_prompt_cache(model)
    prefix_logits = model(mx.array([prefix]), cache=owner_cache)
    mx.eval(prefix_logits)
    lru = IdentityPromptCache(LRUPromptCache(4, 256*1024**2), 'fixed-real-model-test')
    lru.insert_cache('model', prefix, owner_cache, cache_type='user')
    results = []
    for tail in (suffix, tokenizer.encode(' 8 9')):
        request = prefix+tail
        copy, remaining = lru.fetch_nearest_cache('model', request)
        reused = model(mx.array([remaining]), cache=copy)[0, -1]
        reference_cache = make_prompt_cache(model)
        reference_prefix = model(mx.array([prefix]), cache=reference_cache)
        mx.eval(reference_prefix)
        reference = model(mx.array([tail]), cache=reference_cache)[0, -1]
        fresh = model(mx.array([request]), cache=make_prompt_cache(model))[0, -1]
        error = mx.max(mx.abs(reused-fresh)).item()
        equal = mx.allclose(reused, fresh, atol=1e-3, rtol=1e-3).item()
        results.append(dict(prompt_tokens=len(request), reused_tokens=len(prefix),
                            remaining_tokens=len(remaining), maximum_logit_error=error,
                            argmax_equal=mx.argmax(reused).item() == mx.argmax(fresh).item(),
                            segmented_reference_error=mx.max(mx.abs(reused-reference)).item(),
                            cache_copy_allclose=bool(mx.allclose(reused, reference, atol=1e-3, rtol=1e-3).item()),
                            allclose=bool(equal)))
    # Both resumed requests grew their copies; the owner's prefix stays intact.
    restored, remaining = lru.fetch_nearest_cache('model', tokens)
    offsets = [getattr(cache, 'offset', None) for cache in restored]
    lru.trim_to(n_bytes=0)
    miss = lru.fetch_nearest_cache('model', tokens)[0] is None
    report = dict(report_kind='p2_real_model_kv_parity', cases=results,
                  compute_dtype=args.compute_dtype,
                  norm_weight_reuse=not args.disable_norm_reuse,
                  cache_types=[type(cache).__name__ for cache in owner_cache],
                  cache_dtypes=[str(cache.keys.dtype) for cache in owner_cache],
                  restored_offsets=offsets, expected_prefix_length=len(prefix),
                  eviction_releases_entries=miss, model_config_sha256=_sha(Path(args.model)/'config.json'),
                  passed=all(r['allclose'] and r['argmax_equal'] for r in results)
                         and all(offset == len(prefix) for offset in offsets) and miss,
                  qualification=False, performance_qualification=False,
                  scope='Gemma 2 fixed short prefixes; not SWA wraparound or long-context certification')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
