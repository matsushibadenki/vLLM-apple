"""Read P1 checkpoints without treating a stale running marker as healthy."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Callable


def _runner_alive(pid: int) -> bool:
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'command='],
                            capture_output=True, text=True, timeout=5)
    expected = str(Path(__file__).resolve().with_name('run_p1_stability.py'))
    return result.returncode == 0 and expected in result.stdout.split()


def inspect_status(directory: Path, *, now: float | None = None,
                   runner_alive: Callable[[int], bool] = _runner_alive) -> dict:
    state_file = directory/'state.json'
    if state_file.stat().st_size > 4*1024*1024:
        raise ValueError('state exceeds bounded size')
    state = json.loads(state_file.read_text())
    status = state.get('status')
    pid = state.get('pid')
    active = status == 'running' and type(pid) is int and pid > 0 and runner_alive(pid)
    current = time.time() if now is None else now
    age = current-state_file.stat().st_mtime
    report_status = None
    stages = state.get('stages', [])
    if stages and stages[-1].get('status') == 'running':
        report_file = Path(stages[-1]['report'])
        if report_file.parent.resolve() != directory.resolve():
            raise ValueError('stage report must be in its evidence directory')
        if report_file.exists():
            if report_file.stat().st_size > 4*1024*1024:
                raise ValueError('report exceeds bounded size')
            report = json.loads(report_file.read_text())
            age = current-report_file.stat().st_mtime
            report_status = report.get('status')
    observed = 'interrupted' if status == 'running' and not active else (
        'stale_checkpoint' if active and age > 300 else status)
    return dict(recorded_status=status, observed_status=observed, runner_alive=active,
                checkpoint_age_seconds=round(max(0, age), 3), stage_status=report_status,
                recorded_passed=state.get('passed'), qualification=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    status = inspect_status(args.directory)
    print(json.dumps(status, indent=2))
    return 1 if status['observed_status'] in {'interrupted', 'stale_checkpoint', 'failed'} else 0


if __name__ == '__main__':
    raise SystemExit(main())
