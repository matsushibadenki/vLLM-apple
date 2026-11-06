"""Bounded same-worker attribution of post-long-request RSS changes."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qualify_gemma2_batch_mask import (
    _atomic_json,
    _cancel_stream,
    _idle_resources,
    _long_prefix_cases,
    _resources,
    _stream_request,
    _timeout_stream,
)
from vllm_apple.backend import BackendConfig, BackendProcess
from vllm_apple.benchmark_gpu_lock import BenchmarkGpuLock
from vllm_apple.phase_probe import PhaseProbeConfig, measure_stream
from vllm_apple.process_memory import process_activity, resident_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compare-completion', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    root = Path(__file__).resolve().parents[1]
    os.environ.update(PYTHONPATH=str(root), HF_HUB_OFFLINE='1',
                      VLLM_APPLE_P1_PROFILE='1', VLLM_APPLE_P1_EFFICIENCY='baseline')
    report = dict(passed=False, observations=[], scope='one worker; short diagnostic; non-atomic RSS and allocator counters',
                  energy_per_token=None, energy_reason='electrical energy unavailable in this probe')
    with BenchmarkGpuLock() as lock:
        report['gpu_lock'] = lock.snapshot()
        backend = BackendProcess(BackendConfig(
            model=str(args.model.resolve()), executable=args.python.absolute(),
            python_module='vllm_apple.mlx_gemma2_compat', backend_kind='mlx_lm',
            port=19276, startup_timeout=90,
            extra_arguments=('--decode-concurrency', '2', '--prompt-concurrency', '2',
                             '--prefill-step-size', '512', '--prompt-cache-size', '4')))
        try:
            backend.start()
            pid = backend.pid
            report['pid'] = pid
            config = PhaseProbeConfig('http://127.0.0.1:19276', str(args.model.resolve()),
                'Apple-M4-32GiB', maximum_output_tokens=16, target_pid=pid, timeout_seconds=30)

            def sample():
                if backend.pid != pid or not backend.running:
                    raise RuntimeError('worker changed')
                return dict(rss_bytes=resident_bytes(pid), activity=process_activity(pid),
                            monotonic_ns=time.monotonic_ns())

            if args.compare_completion:
                for cycle in range(6):
                    order = ('normal', 'timeout') if cycle % 2 == 0 else ('timeout', 'normal')
                    for mode in order:
                        _, prompt, expected = _long_prefix_cases()[cycle % 3]
                        quality = measure_stream(replace(config, prompt=prompt), expected_text=expected,
                                                 expected_match_mode='trimmed_exact')
                        resources_before = _idle_resources(19276)
                        before = sample()
                        points = []
                        stop = threading.Event()

                        def monitor():
                            while not stop.wait(.02):
                                points.append(sample())

                        thread = threading.Thread(target=monitor, daemon=True)
                        thread.start()
                        done = False
                        tokens = None
                        connection = None
                        try:
                            connection, response = _stream_request(
                                19276, str(args.model.resolve()),
                                request_id='completion-control', max_tokens=512, seed=11,
                                timeout_ms=100 if mode == 'timeout' else None)
                            for line in response:
                                if line.strip() == b'data: [DONE]':
                                    done = True
                                    break
                                if line.startswith(b'data: '):
                                    event = json.loads(line[6:])
                                    usage = event.get('usage')
                                    if usage:
                                        tokens = usage.get('completion_tokens')
                        finally:
                            stop.set()
                            thread.join()
                            if connection:
                                connection.close()
                        immediate = sample()
                        resources_after = _idle_resources(19276)
                        drained = sample()
                        time.sleep(1)
                        settled = sample()
                        report['observations'].append(dict(
                            cycle=cycle, operation=mode, before=before, immediate=immediate,
                            drained=drained, settled=settled, during=points,
                            peak_rss_bytes=max([before['rss_bytes'], immediate['rss_bytes'],
                                                *[p['rss_bytes'] for p in points]]),
                            generation_delta_bytes=immediate['rss_bytes']-before['rss_bytes'],
                            post_stop_delta_bytes=settled['rss_bytes']-immediate['rss_bytes'],
                            completion_tokens=tokens, passed=done,
                            quality_passed=quality.expected_text_matched,
                            allocator_before=resources_before['allocator'],
                            allocator_after=resources_after['allocator']))
                        _atomic_json(args.output, report)
                report['passed'] = all(r['passed'] and r['quality_passed']
                                       for r in report['observations'])
                return 0 if report['passed'] else 1

            for cycle in range(6):
                for name in ('cancel', 'timeout', 'resources'):
                    _, prompt, expected = _long_prefix_cases()[cycle % 3]
                    result = measure_stream(replace(config, prompt=prompt), expected_text=expected,
                                            expected_match_mode='trimmed_exact')
                    drained = _idle_resources(19276)
                    before = sample()
                    if name == 'cancel':
                        evidence = _cancel_stream(19276, str(args.model.resolve()))
                        passed = evidence.get('completion_observed') is True
                    elif name == 'timeout':
                        evidence = _timeout_stream(19276, str(args.model.resolve()))
                        passed = evidence.get('completion_observed') is True
                    else:
                        evidence = [_resources(19276) for _ in range(10)]
                        passed = True
                    immediate = sample()
                    after_resources = _idle_resources(19276)
                    after = sample()
                    report['observations'].append(dict(
                        cycle=cycle, operation=name, quality_passed=result.expected_text_matched,
                        passed=passed, before=before, immediate=immediate, after=after,
                        rss_delta_bytes=after['rss_bytes']-before['rss_bytes'],
                        allocator_before=drained['allocator'],
                        allocator_after=after_resources['allocator'], evidence=evidence))
                    _atomic_json(args.output, report)
            report['passed'] = all(r['passed'] and r['quality_passed']
                                       for r in report['observations'])
        finally:
            backend.stop()
            report['shutdown_clean'] = not backend.running
            _atomic_json(args.output, report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
