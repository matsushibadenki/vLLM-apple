"""Compare two bounded P1 prefill settings across fresh workers; no promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
from pathlib import Path

ORDERS = ((512, 256), (256, 512), (512, 256))


def summarize(directory):
    rows = []
    identities = set()
    for step in (512, 256):
        trials = []
        for repeat in range(1, 4):
            path = directory / f'prefill-{step}-{repeat}.json'
            data = json.loads(path.read_text())
            if data.get('status') != 'complete' or data.get('efficiency_candidate') != 'compact':
                raise ValueError('incomplete trial or incorrect efficiency')
            command = data['command']
            if command[command.index('--prefill-step-size') + 1] != str(step):
                raise ValueError('wrong prefill identity')
            identities.add(hashlib.sha256(json.dumps({k: data[k] for k in
                ('runtime_sources', 'runner_source_sha256', 'model_files')}, sort_keys=True).encode()).hexdigest())
            long = data['long_prefix_edit']
            short = data['sustained']
            trials.append({
                'report': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'passed': data['passed'], 'shutdown_clean': data['shutdown_clean'],
                'identity_unchanged': data['runtime_identity_unchanged'] and data['model_identity_unchanged'],
                'long_e2e_mean_ms': long['latency_distributions']['e2e']['mean_ms'],
                'short_e2e_mean_ms': short['latency_distributions']['e2e']['mean_ms'],
                'quality_passed': long['quality_passed'] + short['quality_passed'],
                'slo_passed': long['slo_quality_passed'] + short['slo_quality_passed'],
                'requests': long['requests'] + short['requests'],
                'allocator_peak_bytes': data['http_exhaustion']['snapshot']['allocator']['peak_bytes'],
            })
        rows.append({'prefill_step': step, 'trials': trials,
                     'long_e2e_median_ms': statistics.median(x['long_e2e_mean_ms'] for x in trials),
                     'long_e2e_range_ms': [min(x['long_e2e_mean_ms'] for x in trials),
                                           max(x['long_e2e_mean_ms'] for x in trials)],
                     'short_e2e_median_ms': statistics.median(x['short_e2e_mean_ms'] for x in trials)})
    if len(identities) != 1:
        raise ValueError('runtime/runner/model identity changed across trials')
    return {'scope': 'Three fresh workers per setting; compact, concurrency two, arithmetic and faults',
            'candidates': rows, 'same_runtime_runner_model': True,
            'performance_qualified': False, 'standard_promotion': False,
            'limitations': ['Short trials do not qualify P1 stability.',
                            'Means across trials are not per-request median/p95/p99.',
                            'No independent quality matrix, TPOT p95 or energy measurement.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--port', type=int, default=19167)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_directory.mkdir(exist_ok=False)
    for repeat, order in enumerate(ORDERS, 1):
        for step in order:
            path = args.output_directory.resolve() / f'prefill-{step}-{repeat}.json'
            command = [sys.executable, str(root / 'scripts/qualify_gemma2_batch_mask.py'),
                       '--python', str(args.python.absolute()), '--model', str(args.model.resolve()),
                       '--output', str(path), '--port', str(args.port), '--sustained-requests', '30',
                       '--long-requests', '12', '--decode-concurrency', '2', '--prompt-concurrency', '2',
                       '--prefill-step-size', str(step), '--long-concurrency', '1', '--p1-profile']
            with path.with_suffix('.runner.log').open('w') as log:
                process = subprocess.Popen(command, cwd=root, stdout=log, stderr=log,
                    env=dict(os.environ, VLLM_APPLE_P1_EFFICIENCY='compact'))
                try:
                    code = process.wait(timeout=180)
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        process.wait(timeout=45)
            data = json.loads(path.read_text())
            print(f'repeat={repeat} prefill={step} exit={code} passed={data.get("passed")}', flush=True)
    summary = summarize(args.output_directory)
    with (args.output_directory / 'summary.json').open('x') as output:
        json.dump(summary, output, indent=2)
        output.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
