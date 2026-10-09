"""Three alternating fresh workers per KV-entry setting; never auto-promote."""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from vllm_apple.trial_process import run_trial

ORDERS = ((4, 3), (3, 4), (4, 3))


def summarize(directory):
    identities = set()
    candidates = []
    for entries in (4, 3):
        trials = []
        for repeat in range(1, 4):
            path = directory / f'entries-{entries}-{repeat}.json'
            data = json.loads(path.read_text())
            command = data['command']
            if (data.get('status') != 'complete' or data.get('efficiency_candidate') != 'compact'
                    or command[command.index('--prompt-cache-size')+1] != str(entries)
                    or command[command.index('--prefill-step-size')+1] != '512'):
                raise ValueError('incomplete or mismatched trial')
            identities.add(hashlib.sha256(json.dumps({key: data[key] for key in
                ('runtime_sources', 'runner_source_sha256', 'model_files')}, sort_keys=True).encode()).hexdigest())
            snapshot = data.get('normal_memory_points', {}).get('after_short')
            if not isinstance(snapshot, dict) or snapshot.get('same_worker') is not True:
                raise ValueError('missing same-worker normal-workload memory point')
            trials.append(dict(report=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                passed=data['passed'], shutdown_clean=data['shutdown_clean'],
                identity_unchanged=data['runtime_identity_unchanged'] and data['model_identity_unchanged'],
                warmup_slo=data['warmup']['slo_quality_passed'],
                cache=snapshot['prompt_cache'], rss_bytes=snapshot['process_activity']['resident_size'],
                benchmarks={key: {name: data[key][name] for name in
                    ('requests', 'quality_passed', 'slo_quality_passed', 'latency_distributions', 'prompt_cache_usage')}
                    for key in ('long_prefix_edit', 'sustained')}))
        candidates.append(dict(entries=entries, trials=trials,
            short_mean_median_ms=statistics.median(t['benchmarks']['sustained']['latency_distributions']['e2e']['mean_ms'] for t in trials),
            short_max_range_ms=[min(t['benchmarks']['sustained']['latency_distributions']['e2e']['max_ms'] for t in trials),
                                max(t['benchmarks']['sustained']['latency_distributions']['e2e']['max_ms'] for t in trials)],
            retained_kv_median_bytes=statistics.median(t['cache']['accounted_bytes'] for t in trials)))
    if len(identities) != 1:
        raise ValueError('runtime/runner/model identity changed')
    return dict(candidates=candidates, same_runtime_runner_model=True, performance_qualified=False,
                stability_qualified=False, default_entries=4,
                scope='Three fresh workers each; medians of run means, not per-request medians; RSS after normal workload is not a trend')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19167)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_directory.mkdir(exist_ok=False)
    for repeat, order in enumerate(ORDERS, 1):
        for entries in order:
            path = args.output_directory.resolve()/f'entries-{entries}-{repeat}.json'
            command = [sys.executable, str(root/'scripts/qualify_gemma2_batch_mask.py'),
                       '--python', str(args.python.absolute()), '--model', str(args.model.resolve()),
                       '--output', str(path), '--port', str(args.port), '--sustained-requests', '30',
                       '--long-requests', '12', '--decode-concurrency', '2', '--prompt-concurrency', '2',
                       '--prefill-step-size', '512', '--long-concurrency', '1', '--p1-profile',
                       '--prompt-cache-entries', str(entries)]
            with path.with_suffix('.runner.log').open('w') as log:
                try:
                    code = run_trial(command, cwd=root, stdout=log,
                        env=dict(os.environ, VLLM_APPLE_P1_EFFICIENCY='compact'))
                except subprocess.TimeoutExpired:
                    path.with_suffix('.timeout.json').write_text(json.dumps(dict(status='timeout', qualification=False)))
                    return 1
            print(f'repeat={repeat} entries={entries} exit={code}', flush=True)
            if code:
                return code
    with (args.output_directory/'summary.json').open('x') as output:
        json.dump(summarize(args.output_directory), output, indent=2)
        output.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
