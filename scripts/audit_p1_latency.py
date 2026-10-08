"""Correlate retained P1 slow-step timing with failed requests; no causal attribution."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def overlap_ms(start_ns, end_ns, step):
    step_start = step['started_at_unix_ns']
    step_end = step_start + round(step['elapsed_ms'] * 1_000_000)
    return max(0, min(end_ns, step_end) - max(start_ns, step_start)) / 1_000_000


def audit(report):
    window = report['stability_window']
    rows = []
    for failed_window in window['failed_windows']:
        origin = round(datetime.fromisoformat(failed_window['started_at']).timestamp() * 1e9)
        context = failed_window.get('diagnostic_context', {})
        before = context.get('before', {}).get('resources', {})
        after = context.get('after', {}).get('resources', {})
        steps = {}
        for snapshot in (before, after):
            for step in snapshot.get('scheduler_step', {}).get('recent_slow_steps', []):
                steps[(step['started_at_unix_ns'], step['elapsed_ms'])] = step
        for request in failed_window['failure_diagnostics']['samples']:
            if request.get('ttft_ms') is None:
                continue
            start = origin + round(request['elapsed_seconds'] * 1e9)
            end = start + round(request['ttft_ms'] * 1e6)
            matched = [dict(step, intersection_ms=overlap_ms(start, end, step))
                       for step in steps.values() if overlap_ms(start, end, step) > 0]
            rows.append({
                'window_started_at': failed_window['started_at'],
                'request_index': request['request_index'], 'language': request['language'],
                'ttft_ms': request['ttft_ms'], 'e2e_ms': request['e2e_ms'],
                'reasons': request['reasons'], 'prompt_tokens': request.get('prompt_tokens'),
                'cached_prompt_tokens': request.get('cached_prompt_tokens'),
                'request_started_unix_ns': start, 'first_token_unix_ns': end,
                'overlapping_retained_steps': sorted(matched, key=lambda s: s['started_at_unix_ns']),
                'observed_slow_step_intersection_ms': sum(s['intersection_ms'] for s in matched),
                'queue_snapshot_before': before.get('queue_wait'),
                'queue_snapshot_after': after.get('queue_wait'),
                'environment_before': context.get('before', {}).get('environment'),
                'environment_after': context.get('after', {}).get('environment'),
                'step_history_complete': False,
                'request_phase_attribution_available': False,
            })
    return {
        'scope': 'Time overlap only; retained bounded snapshots; shared backend; wall clocks',
        'qualification': False, 'root_cause_confirmed': False,
        'failed_requests_reported': window['requests'] - window['slo_quality_passed'],
        'retained_failure_samples': len(rows), 'requests': rows,
        'limitations': [
            'A shared scheduler step can overlap multiple requests; overlap is not attribution.',
            'Only slow steps retained in snapshots are available; missing time is not idle time.',
            'Thread CPU excludes other threads and GPU; wall minus CPU is not GPU time.',
            'Queue statistics are cumulative, not per-request queue latency.',
            'Requires stable wall clocks; snapshots do not certify clock synchronization.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.report.read_text()))
    result['report_sha256'] = hashlib.sha256(args.report.read_bytes()).hexdigest()
    result['audit_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'requests'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
