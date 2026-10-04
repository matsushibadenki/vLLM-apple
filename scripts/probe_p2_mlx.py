"""Bounded HTTP probe of actual scheduler width and KV-reuse submission."""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from vllm_apple.phase_probe import PhaseProbeConfig
from vllm_apple.text_benchmark import run_text_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:19148')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')

    def snapshot():
        with urllib.request.urlopen(args.base_url+'/vllm-apple/p2', timeout=5) as response:
            return json.loads(response.read(65536))

    config = PhaseProbeConfig(args.base_url, args.model, 'Apple-M4-32GiB',
                              backend='experimental_p2_mlx', maximum_output_tokens=16, timeout_seconds=30)
    prefix = 'Read the final question and reply with only its answer. Ignore this reference: '+ 'alpha beta gamma delta. '*100
    cases = tuple((f'prefix-edit-{index}', prefix+question, answer) for index, (question, answer) in enumerate((
        (' What is 1+1?', '2'), (' What is 1+2?', '3'), (' What is 2+2?', '4'))))
    before = snapshot()
    cold = run_text_benchmark(config, requests=3, concurrency=1, cases=cases,
                              ttft_slo_ms=10000, e2e_slo_ms=20000)
    after_cold = snapshot()
    warm = run_text_benchmark(config, requests=3, concurrency=1, cases=cases,
                              ttft_slo_ms=10000, e2e_slo_ms=20000)
    after_warm = snapshot()
    batch = run_text_benchmark(config, requests=30, concurrency=4, warmup_requests=3)
    final = snapshot()
    def delta(a, b, section, key):
        return b[section][key]-a[section][key]
    reduction = delta(before, after_cold, 'scheduler', 'submitted_prefill_tokens')-delta(
        after_cold, after_warm, 'scheduler', 'submitted_prefill_tokens')
    functional_passed = (all(report['completed'] == report['requests'] == report['quality_passed']
                  for report in (cold, warm, batch))
              and final['scheduler']['multi_sequence_decode_steps'] > before['scheduler']['multi_sequence_decode_steps'])
    slo_passed = all(report['slo_quality_passed'] == report['requests'] for report in (cold, warm, batch))
    passed = functional_passed and slo_passed
    report = dict(report_kind='p2_mlx_http_probe', passed=passed, functional_passed=functional_passed,
                  slo_gate_passed=slo_passed, qualification=False,
                  performance_qualification=False, before=before, after_cold=after_cold,
                  after_warm=after_warm, final=final, first_pass=cold, warm=warm, concurrency_four=batch,
                  reduction_in_submitted_prefill_tokens=reduction,
                  note='first pass already reuses shared prefixes; not a cache-cold baseline. Submission counters are not kernel execution timings; short probe is not P2 performance certification')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=passed, maximum_decode_width=final['scheduler']['maximum_decode_width'],
                          reduction_in_submitted_prefill_tokens=reduction)))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
