"""Compare ps/native RSS costs and agreement on stable live child allocations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm_apple.process_memory import _ps_resident_bytes, resident_bytes

CHILD = '''import sys
import sysconfig
payload = None
for line in sys.stdin:
    payload = bytearray(int(line) * 1024**2)
    for index in range(0, len(payload), 4096):
        payload[index] = 1
    print("ready", flush=True)
'''


def distribution(values):
    values = sorted(values)
    return dict(samples=len(values), median_ms=statistics.median(values)/1e6,
                p95_ms=values[math.ceil(.95*len(values))-1]/1e6,
                p99_ms=values[math.ceil(.99*len(values))-1]/1e6,
                max_ms=max(values)/1e6, p99_reference_only=len(values)<1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sys.platform != 'darwin':
        parser.error('requires macOS and a new output path')
    rows = []
    child = subprocess.Popen([sys.executable, '-u', '-c', CHILD], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
    try:
        for mib in (8, 64, 256, 512):
            child.stdin.write(str(mib)+'\n')
            child.stdin.flush()
            if child.stdout.readline().strip() != 'ready':
                raise RuntimeError('child allocation failed')
            agreement = [(resident_bytes(child.pid), _ps_resident_bytes(child.pid)) for _ in range(5)]
            delta = max(abs(a-b) for a,b in agreement)
            if delta > 64*1024:
                raise RuntimeError(f'RSS mismatch: {delta} bytes')
            timings = {'ps': [], 'native': []}
            costs = []
            for repeat in range(3):
                for name in (('ps', 'native') if repeat%2==0 else ('native', 'ps')):
                    method = _ps_resident_bytes if name == 'ps' else resident_bytes
                    with patch('subprocess.run', wraps=subprocess.run) as spawn:
                        before = resource.getrusage(resource.RUSAGE_CHILDREN)
                        cpu = time.process_time_ns()
                        for _ in range(100):
                            start = time.perf_counter_ns()
                            method(child.pid)
                            timings[name].append(time.perf_counter_ns()-start)
                        cpu = time.process_time_ns()-cpu
                        after = resource.getrusage(resource.RUSAGE_CHILDREN)
                        count = spawn.call_count
                    if name == 'native' and count:
                        raise RuntimeError('native reader fell back to subprocess')
                    costs.append(dict(method=name, repeat=repeat+1, parent_cpu_ms=cpu/1e6,
                        child_cpu_ms=1000*(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime),
                        subprocesses=count))
            rows.append(dict(allocation_mib=mib, rss_agreement_pairs=agreement,
                             maximum_difference_bytes=delta, timings={k:distribution(v) for k,v in timings.items()}, costs=costs))
    finally:
        child.stdin.close()
        try:
            child.wait(timeout=5)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stdout.close()
    try:
        resident_bytes(child.pid)
    except OSError:
        exited_pid_rejected = True
    else:
        raise RuntimeError('exited PID returned a sample')
    args.output.write_text(json.dumps(dict(scope='RSS sampling only; no inference/GPU/power qualification',
        rows=rows, exited_pid_rejected=exited_pid_rejected, watts_measured=False,
        python_version=platform.python_version(), python_debug_build=bool(sysconfig.get_config_var('Py_DEBUG')),
        system=platform.platform(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sampler_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'vllm_apple/process_memory.py').read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps([dict(allocation_mib=r['allocation_mib'], timings=r['timings'],
                          maximum_difference_bytes=r['maximum_difference_bytes']) for r in rows],indent=2))


if __name__ == '__main__':
    main()
