"""Real KV parity through 1025 tokens, branching and mixed-prefix batch merge."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.models import gemma2
    from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache

    from experiments.p2_mlx.cache import IdentityPromptCache
    from experiments.p2_mlx.precision import install_norm_weight_reuse
    from experiments.p2_mlx.server import _sha
    from vllm_apple.mlx_gemma2_compat import install_gemma2_batch_mask_fix

    install_gemma2_batch_mask_fix()
    mx.set_memory_limit(8*1024**3)
    mx.set_cache_limit(256*1024**2)
    model, tokenizer = load(str(args.model))
    packed = {name: value for name, value in tree_flatten(model.parameters())
              if not mx.issubdtype(value.dtype, mx.floating)}
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    after = dict(tree_flatten(model.parameters()))
    packed_unchanged = all(after[name] is value for name, value in packed.items())
    norms = install_norm_weight_reuse(gemma2, mx)
    tokens = tokenizer.encode('alpha beta gamma delta. '*240)
    results, owners = [], []
    lengths = (19, 127, 511, 513, 1025)
    if len(tokens) < max(lengths):
        raise RuntimeError('fixed probe requires at least 1025 reference tokens')
    for length in lengths:
        prefix = tokens[:length]
        cache = make_prompt_cache(model)
        mx.eval(model(mx.array([prefix]), cache=cache))
        owner = IdentityPromptCache(LRUPromptCache(4, 256*1024**2), 'fixed-probe-float32')
        owner.insert_cache('model', prefix, cache, cache_type='user')
        for tail in ([11], [12, 13, 14, 15]):
            branch, remaining = owner.fetch_nearest_cache('model', prefix+tail)
            resumed = model(mx.array([remaining]), cache=branch)[0, -1]
            fresh = model(mx.array([prefix+tail]), cache=make_prompt_cache(model))[0, -1]
            results.append(dict(prefix_tokens=length, suffix_tokens=len(tail),
                maximum_logit_error=mx.max(mx.abs(resumed-fresh)).item(),
                allclose=bool(mx.allclose(resumed, fresh, atol=1e-3, rtol=1e-3).item()),
                argmax_equal=mx.argmax(resumed).item() == mx.argmax(fresh).item(),
                owner_unchanged=all(x.offset == length for x in cache)))
        owners.append(cache)
    for indices in ((0, 1), (0, 1, 2, 3)):
        chosen = [copy.deepcopy(owners[i]) for i in indices]
        merged = [layer[0].merge(list(layer)) for layer in zip(*chosen)]
        tails = [[11, 12, 13, 14] for _ in indices]
        output = model(mx.array(tails), cache=merged)[:, -1]
        for row, index in enumerate(indices):
            length = lengths[index]
            fresh = model(mx.array([tokens[:length]+tails[row]]), cache=make_prompt_cache(model))[0, -1]
            results.append(dict(batch_width=len(indices), prefix_tokens=length,
                maximum_logit_error=mx.max(mx.abs(output[row]-fresh)).item(),
                allclose=bool(mx.allclose(output[row], fresh, atol=1e-3, rtol=1e-3).item()),
                argmax_equal=mx.argmax(output[row]).item() == mx.argmax(fresh).item()))
    report = dict(report_kind='p2_extended_kv_parity', cases=results, compute_dtype='float32',
        atol=1e-3, rtol=1e-3, packed_integer_parameter_count=len(packed),
        packed_integer_parameters_unchanged=packed_unchanged, norm_weight_reuse=dict(norms),
        model_files={p.name: _sha(p) for p in sorted(args.model.iterdir()) if p.is_file()},
        sources={str(p): _sha(p) for p in (Path(__file__), Path(gemma2.__file__),
                 Path('experiments/p2_mlx/precision.py'), Path('experiments/p2_mlx/cache.py'))},
        passed=packed_unchanged and all(x['allclose'] and x['argmax_equal']
            and x.get('owner_unchanged', True) for x in results),
        qualification=False, performance_qualification=False,
        scope='fixed float32 exact-prefix cases through 1025 tokens; mixed left-padding native KV merge widths 2/4; not SWA wraparound or general quality certification')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], maximum_error=max(x['maximum_logit_error'] for x in results))))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
