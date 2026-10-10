"""Three fresh uncached P2 workers against a recorded same-precision baseline."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from scripts.compare_p2_mlx import SampledProfiler, percentile, sha
from vllm_apple import text_benchmark
from vllm_apple.phase_probe import PhaseProbeConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    comparison = args.comparison.resolve()
    reference = json.loads(comparison.read_text())
    if reference['baseline_compute_dtype'] != 'float32':
        parser.error('a same-float32 baseline is required')
    identity = reference['source_and_model_identity']
    if not all(sha(Path(p)) == h for p, h in identity.items()):
        parser.error('comparison source/model identity no longer matches')
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    template = json.loads((comparison.parent/'1-optimized.json').read_text())
    requests = template['c1']['requests']
    command = [*template['command'], '--disable-prefix-cache']
    port = command[command.index('--port')+1]
    model = command[command.index('--model')+1]
    base_url = f'http://127.0.0.1:{port}'
    tool_hash = sha(Path(__file__))
    text_benchmark.ExecutionPhaseProfiler = SampledProfiler
    rows = []
    for index in range(3):
        row = dict(command=command, passed=False)
        with (output/f'{index}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=log,
                env=dict(os.environ, PYTHONPATH=str(root), HF_HUB_OFFLINE='1', VLLM_APPLE_P1_PROFILE='0'),
                start_new_session=True)
            try:
                deadline = time.monotonic()+90
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('worker exited; see trial log')
                    try:
                        with urllib.request.urlopen(base_url+'/v1/models', timeout=1):
                            break
                    except (OSError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError('worker readiness timed out')
                        time.sleep(.1)
                config = PhaseProbeConfig(base_url, model, 'Apple-M4-32GiB',
                    backend='experimental_p2_mlx', maximum_output_tokens=16,
                    target_pid=process.pid, timeout_seconds=30)
                row['warmup'] = text_benchmark.run_text_benchmark(config, requests=3)
                row['c1'] = text_benchmark.run_text_benchmark(config, requests=requests)
                with urllib.request.urlopen(base_url+'/vllm-apple/p2', timeout=5) as response:
                    row['backend'] = json.loads(response.read(65536))
                row['passed'] = (row['c1']['quality_passed'] == requests
                    and row['c1']['slo_quality_passed'] == requests
                    and row['backend']['cache']['reused_tokens'] == 0
                    and row['backend']['cache']['cache_entries'] == 0
                    and not row['backend']['cache']['enabled']
                    and row['backend']['norm_weight_reuse']['weights_built'] == 105
                    and row['backend']['spm_tokenmap_reuse']['tables_built'] == 1)
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
                row['identity_unchanged'] = (sha(Path(__file__)) == tool_hash
                    and all(sha(Path(p)) == h for p, h in identity.items()))
                (output/f'{index}.json').write_text(json.dumps(row, indent=2)+'\n')
        rows.append(row)
        print(json.dumps(dict(trial=index, passed=row['passed'],
            ttft_p95_ms=percentile(row['c1'], 'ttft_ms'),
            tpot_p95_ms=percentile(row['c1'], 'tpot_ms'))), flush=True)
    ttft = [percentile(r['c1'], 'ttft_ms') for r in rows]
    tpot = [percentile(r['c1'], 'tpot_ms') for r in rows]
    b = reference['metrics']['baseline']
    passed = (all(r['passed'] and r['identity_unchanged'] and r['exit_code'] == 0 for r in rows)
        and max(ttft) <= min(b['c1_ttft_p95_ms'])*1.05
        and max(tpot) <= min(b['c1_tpot_p95_ms'])*1.05)
    report = dict(report_kind='p2_uncached_single_request_latency',
        reference_path=str(comparison), reference_sha256=sha(comparison),
        tool_sha256=tool_hash, source_and_model_identity=identity,
        ttft_p95_ms=ttft, tpot_p95_ms=tpot, passed=passed, qualification=False,
        scope='three fresh float32 workers; prefix cache disabled; fixed arithmetic EN/JA/ZH; not float16 replacement or broad P2 certification')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
