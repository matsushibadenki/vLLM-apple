"""Measure retained allocator memory after fixed requests for each P1 candidate."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qualify_gemma2_batch_mask import _idle_resources, _long_prefix_cases
from vllm_apple.backend import BackendConfig, BackendProcess
from vllm_apple.phase_probe import PhaseProbeConfig, measure_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    root = Path(__file__).resolve().parents[1]
    observations = []
    for name in ('baseline', 'responsive', 'compact'):
        os.environ.update(VLLM_APPLE_P1_PROFILE='1', VLLM_APPLE_P1_EFFICIENCY=name,
                          PYTHONPATH=str(root), HF_HUB_OFFLINE='1')
        backend = BackendProcess(BackendConfig(model=str(args.model.resolve()),
            executable=args.python.absolute(), python_module='vllm_apple.mlx_gemma2_compat',
            backend_kind='mlx_lm', host='127.0.0.1', port=19166, startup_timeout=90,
            extra_arguments=('--decode-concurrency', '2', '--prompt-concurrency', '2',
                             '--prefill-step-size', '512', '--prompt-cache-size', '4')))
        row = dict(candidate=name, samples=[], process_stopped=False)
        observations.append(row)
        try:
            backend.start()
            if not backend.ready:
                raise RuntimeError('backend did not become ready')
            config = PhaseProbeConfig('http://127.0.0.1:19166', str(args.model.resolve()),
                                      'Apple-M4-32GiB', maximum_output_tokens=16,
                                      target_pid=backend.pid, timeout_seconds=30)
            cases = [('short', 'What is 1+1? Reply with only the digit.', '2'), *_long_prefix_cases()]
            for index, (_, prompt, expected) in enumerate(cases * 2):
                result = measure_stream(replace(config, prompt=prompt), expected_text=expected,
                                        expected_match_mode='trimmed_exact')
                resources = _idle_resources(19166)
                row['samples'].append(dict(index=index, quality_passed=result.expected_text_matched,
                    e2e_ns=(result.measurement.stream_done_ns-result.measurement.started_ns
                            if result.measurement.stream_done_ns is not None else None), rss_bytes=result.steady_memory_bytes,
                    allocator=resources['allocator'], efficiency=resources.get('efficiency')))
        finally:
            backend.stop()
            row['process_stopped'] = not backend.running
            args.output.write_text(json.dumps(dict(scope='short fixed requests; no long-run or energy qualification',
                watts_measured=False, observations=observations), indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
