"""Compare source-pinned P1 efficiency candidates without automatic promotion.

Every trial uses a new worker. Order rotates across three repeats. This is a
short arithmetic/cancellation comparison, not P1/P3 qualification or a watt test.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ORDERS = (('baseline', 'responsive', 'compact'),
          ('compact', 'baseline', 'responsive'),
          ('responsive', 'compact', 'baseline'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19166)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_directory.mkdir(parents=False, exist_ok=False)
    rows = []
    for repeat, order in enumerate(ORDERS, 1):
        for candidate in order:
            report = args.output_directory.resolve() / f'{candidate}-{repeat}.json'
            command = [sys.executable, str(root/'scripts/qualify_gemma2_batch_mask.py'),
                       '--python', str(args.python.absolute()), '--model', str(args.model.resolve()),
                       '--output', str(report), '--port', str(args.port), '--sustained-requests', '60',
                       '--long-requests', '12', '--decode-concurrency', '2', '--prompt-concurrency', '2',
                       '--prefill-step-size', '512', '--long-concurrency', '1', '--p1-profile']
            with report.with_suffix('.runner.log').open('w') as log:
                process = subprocess.Popen(command, cwd=root, stdout=log, stderr=log,
                    env=dict(os.environ, VLLM_APPLE_P1_EFFICIENCY=candidate))
                try:
                    code = process.wait(timeout=240)
                finally:
                    if process.poll() is None:
                        # Let the runner's finally block stop its backend.
                        process.send_signal(signal.SIGINT)
                        process.wait(timeout=45)
            evidence = json.loads(report.read_text()) if report.exists() else {}
            rows.append(dict(candidate=candidate, repeat=repeat, returncode=code,
                             passed=evidence.get('passed'), report=str(report)))
            (args.output_directory/'progress.json').write_text(json.dumps(rows, indent=2)+'\n')
            print(candidate, repeat, code, evidence.get('passed'), flush=True)
    return 0 if all(row['passed'] for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
