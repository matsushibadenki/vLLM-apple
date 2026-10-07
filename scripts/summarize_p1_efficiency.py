"""Summarize nine P1 short trials; never promote an execution profile."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
identities = set()
for name in ('baseline', 'responsive', 'compact'):
    trials = []
    for repeat in range(1, 4):
        path = root/f'{name}-{repeat}.json'
        data = json.loads(path.read_text())
        if data.get('status') != 'complete' or data.get('efficiency_candidate') != name:
            raise ValueError(f'incomplete or mismatched trial: {path}')
        identities.add(hashlib.sha256(json.dumps({k: data[k] for k in
            ('runtime_sources', 'runner_source_sha256', 'model_files')}, sort_keys=True).encode()).hexdigest())
        short = data['sustained']
        long = data['long_prefix_edit']
        trials.append(dict(report=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            passed=data['passed'], identity_unchanged=data['runtime_identity_unchanged'] and data['model_identity_unchanged'],
            shutdown_clean=data['shutdown_clean'],
            short_e2e_mean_ms=short['latency_distributions']['e2e']['mean_ms'],
            short_ttft_mean_ms=short['latency_distributions']['ttft']['mean_ms'],
            long_e2e_mean_ms=long['latency_distributions']['e2e']['mean_ms'],
            quality_passed=short['quality_passed']+long['quality_passed'],
            completed=short['completed']+long['completed'],
            slo_quality_passed=short['slo_quality_passed']+long['slo_quality_passed'],
            allocator_peak_bytes=data['http_exhaustion']['snapshot']['allocator']['peak_bytes']))
    times = [x['short_e2e_mean_ms'] for x in trials]
    rows.append(dict(candidate=name, trials=trials, short_e2e_median_ms=statistics.median(times),
                     short_e2e_range_ms=[min(times), max(times)],
                     relative_spread=(max(times)-min(times))/statistics.median(times),
                     long_e2e_median_ms=statistics.median(x['long_e2e_mean_ms'] for x in trials)))
if len(identities) != 1:
    raise ValueError('runtime/runner/model identity differs across trials')
baseline = rows[0]
for row in rows:
    row['descriptive_short_e2e_change_percent'] = round(100*(row['short_e2e_median_ms']/baseline['short_e2e_median_ms']-1), 2)
    row['range_separated_from_baseline'] = row['short_e2e_range_ms'][1] < baseline['short_e2e_range_ms'][0]
summary = dict(scope='three rotated independent workers per candidate, short arithmetic and fault checks',
               same_runtime_model_runner=True, candidates=rows, performance_qualified=False,
               standard_promotion=False, watts_measured=False,
               limitations=['No long-run qualification; R9 failed P1.',
                            'Arithmetic is not general model quality.',
                            'TPOT p95 and electrical power were not measured.',
                            'Descriptive medians do not establish causal speedup or P3 qualification.'])
(root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps([{k:v for k,v in row.items() if k!='trials'} for row in rows], indent=2))
