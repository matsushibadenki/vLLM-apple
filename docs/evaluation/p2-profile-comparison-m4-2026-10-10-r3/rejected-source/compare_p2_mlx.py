"""Six fresh-worker P2 profile trials with bounded client latency samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from vllm_apple import text_benchmark
from vllm_apple.phase_probe import PhaseProbeConfig
from vllm_apple.phase_profile import ExecutionPhaseProfiler


class SampledProfiler(ExecutionPhaseProfiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latencies = []

    def record(self, measurement):
        super().record(measurement)
        if len(self.latencies) >= 1000:
            raise RuntimeError('bounded latency sample limit exceeded')
        self.latencies.append(dict(ttft_ms=measurement.ttft_ns/1e6,
            tpot_ms=(measurement.decode_ns/measurement.token_intervals/1e6
                     if measurement.token_intervals else None)))

    def snapshot(self):
        report = super().snapshot()
        report['client_latency_samples'] = self.latencies
        report['sample_scope'] = 'request TTFT and request-average client TPOT, not individual token intervals'
        return report


def percentile(report, field):
    values = sorted(x[field] for x in report['phase_profile']['client_latency_samples'] if x[field] is not None)
    return values[math.ceil(len(values)*.95)-1] if values else None


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19148)
    parser.add_argument('--requests', type=int, default=60)
    args = parser.parse_args()
    if not 30 <= args.requests <= 1000 or not 1024 <= args.port <= 65535:
        parser.error('requests must be 30..1000 and port 1024..65535')
    root = Path(__file__).resolve().parents[1]
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = [*sorted((root/'experiments/p2_mlx').glob('*.py')),
             *sorted((root/'vllm_apple').glob('*.py')), Path(__file__).resolve(),
             *sorted(p for p in args.model.resolve().iterdir() if p.is_file())]
    identity = {str(p): sha(p) for p in files}
    text_benchmark.ExecutionPhaseProfiler = SampledProfiler
    rows = []
    base_url = f'http://127.0.0.1:{args.port}'
    for index, profile in enumerate(('baseline', 'optimized', 'optimized', 'baseline', 'baseline', 'optimized')):
        # Resolving a venv executable's symlink loses its package environment.
        command = [str(args.python.absolute()), '-m', 'experiments.p2_mlx.server',
            '--model', str(args.model.resolve()), '--host', '127.0.0.1', '--port', str(args.port),
            '--decode-concurrency', '4', '--prompt-concurrency', '2',
            '--prefill-step-size', '512', '--prompt-cache-size', '4']
        if profile == 'baseline':
            command += ['--disable-prefix-cache', '--compute-dtype', 'float16',
                        '--scheduler-budget-ms', '500', '--disable-spm-reuse']
            command += ['--disable-mlp-compilation']
        row = dict(index=index, profile=profile, command=command, passed=False)
        with (output/f'{index}-{profile}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=log,
                env=dict(os.environ, PYTHONPATH=str(root), HF_HUB_OFFLINE='1', VLLM_APPLE_P1_PROFILE='0'),
                start_new_session=True)
            try:
                deadline = time.monotonic()+90
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'worker exited {process.returncode}; see trial log')
                    try:
                        with urllib.request.urlopen(base_url+'/v1/models', timeout=1) as response:
                            if response.status == 200:
                                break
                    except (OSError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError('worker readiness timed out')
                        time.sleep(.1)
                config = PhaseProbeConfig(base_url, str(args.model.resolve()), 'Apple-M4-32GiB',
                    backend='experimental_p2_mlx', maximum_output_tokens=16, target_pid=process.pid,
                    timeout_seconds=30)
                row['warmup'] = text_benchmark.run_text_benchmark(config, requests=3)
                for concurrency in (1, 4):
                    row[f'c{concurrency}'] = text_benchmark.run_text_benchmark(
                        config, requests=args.requests, concurrency=concurrency)
                    print(json.dumps(dict(trial=index, profile=profile, concurrency=concurrency,
                        quality=row[f'c{concurrency}']['quality_passed'],
                        slo=row[f'c{concurrency}']['slo_quality_passed'])), flush=True)
                with urllib.request.urlopen(base_url+'/vllm-apple/p2', timeout=5) as response:
                    row['backend'] = json.loads(response.read(65536))
                row['passed'] = all(row[f'c{c}']['quality_passed'] == args.requests
                    and row[f'c{c}']['slo_quality_passed'] == args.requests for c in (1, 4))
            finally:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                row['exit_code'] = process.returncode
                row['identity_unchanged'] = all(sha(Path(p)) == h for p, h in identity.items())
                (output/f'{index}-{profile}.json').write_text(json.dumps(row, indent=2)+'\n')
        rows.append(row)
    baseline = [r for r in rows if r['profile'] == 'baseline']
    optimized = [r for r in rows if r['profile'] == 'optimized']
    metrics = {}
    for name, group in (('baseline', baseline), ('optimized', optimized)):
        metrics[name] = dict(c4_goodput=sum(r['c4']['goodput_tokens_per_second'] for r in group)/3,
            c1_ttft_p95_ms=[percentile(r['c1'], 'ttft_ms') for r in group],
            c1_tpot_p95_ms=[percentile(r['c1'], 'tpot_ms') for r in group],
            c4_ttft_p95_ms=[percentile(r['c4'], 'ttft_ms') for r in group],
            c1_e2e_mean_ms=[r['c1']['latency_distributions']['e2e']['mean_ms'] for r in group],
            c4_e2e_mean_ms=[r['c4']['latency_distributions']['e2e']['mean_ms'] for r in group])
    b, o = metrics['baseline'], metrics['optimized']
    performance_gate = (all(r['passed'] and r['identity_unchanged'] and r['exit_code'] == 0 for r in optimized)
        and all(r['identity_unchanged'] and r['exit_code'] == 0 for r in baseline)
        and o['c4_goodput'] >= b['c4_goodput']*1.2
        and max(o['c1_ttft_p95_ms']) <= min(b['c1_ttft_p95_ms'])*1.05
        and max(o['c1_tpot_p95_ms']) <= min(b['c1_tpot_p95_ms'])*1.05)
    report = dict(report_kind='p2_profile_comparison', metrics=metrics,
        source_and_model_identity=identity, performance_gate_passed=performance_gate,
        qualification=False, scope='three fresh workers per profile; fixed arithmetic EN/JA/ZH, c1/c4; not broad P2 or P1 certification',
        comparison='old float16/500ms/cache-off/no-SPM-reuse versus float32/50ms/cache-on/shared-SPM; profile effect, not cache-only attribution')
    (output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['metrics']), flush=True)
    return 0 if performance_gate else 1


if __name__ == '__main__':
    raise SystemExit(main())
