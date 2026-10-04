"""Reproduce the bounded M4 P0 baseline, using fresh backend processes."""
import argparse
import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from vllm_apple.phase_probe import PhaseProbeConfig
from vllm_apple.text_benchmark import run_text_benchmark

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True, help='New evidence directory; never overwrite')
args = parser.parse_args()
chip = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
memory = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
if chip != 'Apple M4' or memory != 32 * 1024**3:
    parser.error('This protocol is bound to Apple M4 / 32 GiB; define a separate hardware protocol')
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
for port in (19140, 19141):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
python = '/opt/homebrew/opt/vllm-metal/libexec/bin/python'
site = Path('/opt/homebrew/opt/vllm-metal/libexec/lib/python3.12/site-packages')
def digest_files(paths, base):
    entries = []
    for path in sorted(paths):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(1048576), b''):
                    digest.update(chunk)
            entries.append({'path': str(path.relative_to(base)), 'sha256': digest.hexdigest()})
    return entries, hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
model_entries, model_digest = digest_files((root / 'models/gemma-2-2b-it-4bit').iterdir(), root)
versions = json.loads(subprocess.check_output([python, '-c',
    'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ("mlx","mlx-lm","vllm","vllm-metal")}))'], text=True))
builds = {}
for backend, packages in [('mlx_lm', ['mlx_lm']), ('vllm_metal', ['vllm', 'vllm_metal'])]:
    entries, digest = digest_files([p for name in packages for p in (site/name).rglob('*.py')], site)
    builds[backend] = {'source_files': entries, 'source_sha256': digest}
    builds[backend]['build_sha256'] = hashlib.sha256(json.dumps(
        dict(versions=versions, source_sha256=digest), sort_keys=True).encode()).hexdigest()
identity = dict(model_files=model_entries, artifact_identity_sha256=model_digest,
                builds=builds, versions=versions, platform=platform.platform(),
                machine=platform.machine(), model_config=json.loads(
                    (root/'models/gemma-2-2b-it-4bit/config.json').read_text()),
                quality_policy='arithmetic_trimmed_exact', requests_per_run=102,
                concurrency=1, warmup_requests=3, maximum_output_tokens=16,
                backend_order=['vllm_metal','mlx_lm','mlx_lm','vllm_metal','vllm_metal','mlx_lm'])
(out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
counts = dict(mlx_lm=0, vllm_metal=0)
for index, backend in enumerate(identity['backend_order']):
    counts[backend] += 1
    number = counts[backend]
    port = 19140 if backend == 'vllm_metal' else 19141
    command = ([str(Path(python).parent/'vllm'), 'serve', 'models/gemma-2-2b-it-4bit',
                '--host','127.0.0.1','--port',str(port),'--max-model-len','1024','--max-num-seqs','1']
               if backend == 'vllm_metal' else [python,'-m','mlx_lm.server','--model',
                 'models/gemma-2-2b-it-4bit','--host','127.0.0.1','--port',str(port)])
    process = None
    pid = None
    with (out/f'{backend}-r{number}.log').open('w') as log:
        try:
            process = subprocess.Popen(command, cwd=root, env=dict(os.environ, HF_HUB_OFFLINE='1'),
                                       stdout=log, stderr=log, start_new_session=True)
            pid = process.pid
            deadline = time.monotonic()+120
            while True:
                if process is not None and process.poll() is not None:
                    raise RuntimeError('backend startup failed')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=1):
                        break
                except OSError:
                    if time.monotonic()>deadline:
                        raise RuntimeError('backend readiness deadline exceeded')
                    time.sleep(.5)
            config = PhaseProbeConfig(f'http://127.0.0.1:{port}', 'models/gemma-2-2b-it-4bit',
                'Apple-M4-32GiB', backend=backend, maximum_output_tokens=16, timeout_seconds=10,
                target_pid=pid)
            report = run_text_benchmark(config, requests=102, warmup_requests=3,
                collect_operating_context=True, artifact_identity_sha256=model_digest,
                backend_build_sha256=builds[backend]['build_sha256'])
            report['reproduction'] = dict(command=command, independent_process=True,
                                         run_number=number, order_index=index, HF_HUB_OFFLINE=True)
            (out/f'{backend}-r{number}.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(dict(backend=backend,run=number,completed=report['completed'],
                quality=report['quality_passed'],slo=report['slo_quality_passed'],
                goodput=report['goodput_tokens_per_second'])), flush=True)
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(5)
