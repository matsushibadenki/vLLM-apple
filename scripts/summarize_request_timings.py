"""Summarize recorded request host phases without filling missing observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

PHASES = {
    'queue_creation_to_dequeue': (None, 'scheduler_dequeued'),
    'dequeue_to_first_response_ready': ('scheduler_dequeued', 'first_response_ready'),
    'dequeue_to_tokenize_start': ('scheduler_dequeued', 'tokenize_started'),
    'tokenize': ('tokenize_started', 'tokenize_finished'),
    'tokenize_end_to_first_response_ready': ('tokenize_finished', 'first_response_ready'),
    'ready_to_http_receipt': ('first_response_ready', 'first_response_received'),
    'first_sse_write': ('first_sse_write_started', 'first_sse_write_returned'),
}


def summarize(records):
    result = {}
    for name, (start, end) in PHASES.items():
        values = []
        for record in records:
            stages = record['stages_ms']
            if end not in stages or (start is not None and start not in stages):
                continue
            value = stages[end] - (stages[start] if start is not None else 0)
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid monotonic phase interval')
            values.append(value)
        values.sort()
        result[name] = {'samples': len(values), 'missing': len(records) - len(values),
                        'median_ms': statistics.median(values) if values else None,
                        'p95_ms': values[math.ceil(.95 * len(values)) - 1] if values else None,
                        'p99_ms': values[math.ceil(.99 * len(values)) - 1] if values else None,
                        'max_ms': max(values) if values else None,
                        'p99_reference_only': len(values) < 100}
    return {'records': len(records), 'phases': result, 'qualification': False,
            'scope': 'Retained completed host traces, including faults; no workload grouping',
            'interpretation': 'After-tokenize interval includes backend work and waits, not GPU-only time'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    snapshot = report['http_exhaustion']['snapshot']['request_timings']
    result = summarize(snapshot['records'])
    result['report_sha256'] = hashlib.sha256(args.report.read_bytes()).hexdigest()
    result['summary_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
