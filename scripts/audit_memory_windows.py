"""Audit retained first/last workload windows without running a model."""
import argparse
import hashlib
import json
from pathlib import Path

from vllm_apple.process_activity_delta import memory_activity_delta


def audit(path):
    if path.stat().st_size > 4*1024*1024:
        raise ValueError('report exceeds bounded size')
    data = json.loads(path.read_text())
    stability = data.get('stability_window', {})
    rows = []
    for name in ('first_window', 'last_window'):
        window = stability.get(name)
        if not isinstance(window, dict):
            continue
        context = window.get('diagnostic_context', {})
        rows.append({'window': name, 'workload_sha256': window.get('workload_sha256'),
                     'activity': memory_activity_delta(context.get('before', {}).get('resources'),
                                                      context.get('after', {}).get('resources'))})
    return {'report': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'windows': rows, 'scope': 'retained first/last windows only; not full-run memory attribution',
            'performance_qualified': False, 'stability_qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps([audit(path) for path in args.reports], indent=2))


if __name__ == '__main__':
    main()
