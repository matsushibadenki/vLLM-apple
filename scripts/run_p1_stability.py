"""Run P1 30-minute and eight-hour qualification sequentially, with checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from scripts.qualify_gemma2_batch_mask import _atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--port', type=int, default=19146)
    args = parser.parse_args()
    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f'test orchestration received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    root = Path(__file__).resolve().parents[1]
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', passed=False, pid=os.getpid(), stages=[],
                 scope='P1 awake mixed-load candidate; sleep/wake remains unqualified',
                 automatic_promotion=False)
    _atomic_json(output/'state.json', state)
    inhibitor = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(os.getpid())])
    try:
        for name, duration, gate, crash_interval in (
            ('30min', 1800, '--require-30-minute-window', '0'),
            ('8hour', 28800, '--require-8-hour-window', '3600'),
        ):
            report = output/f'{name}.json'
            command = [sys.executable, str(root/'scripts/qualify_gemma2_batch_mask.py'),
                '--python', str(args.python.absolute()), '--model', str(args.model.resolve()),
                '--output', str(report), '--port', str(args.port), '--sustained-requests', '100',
                '--long-requests', '12', '--duration-seconds', str(duration), gate,
                '--decode-concurrency', '2', '--prompt-concurrency', '2', '--prefill-step-size', '512',
                '--long-concurrency', '1', '--fault-check-interval-cycles', '10',
                '--worker-crash-interval-seconds', crash_interval, '--p1-profile']
            stage = dict(name=name, status='running', command=command, report=str(report))
            state['stages'].append(stage)
            _atomic_json(output/'state.json', state)
            with (output/f'{name}.log').open('w') as log:
                child = subprocess.Popen(command, cwd=root, env=dict(os.environ, PYTHONPATH=str(root)),
                                         stdout=log, stderr=log)
                try:
                    returncode = child.wait(timeout=duration+600)
                finally:
                    if child.poll() is None:
                        child.send_signal(signal.SIGINT)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=5)
            evidence = json.loads(report.read_text()) if report.is_file() else {}
            stage.update(status='complete' if returncode == 0 else 'failed',
                         passed=evidence.get('passed') is True, returncode=returncode)
            if returncode != 0 or evidence.get('passed') is not True:
                raise RuntimeError(f'{name} qualification failed; later stages will not start')
            _atomic_json(output/'state.json', state)
        state.update(status='complete', passed=True)
    except BaseException as error:
        state.update(status='failed', passed=False, error_type=type(error).__name__, error=str(error)[:2048])
        raise
    finally:
        inhibitor.terminate()
        inhibitor.wait(timeout=5)
        _atomic_json(output/'state.json', state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
