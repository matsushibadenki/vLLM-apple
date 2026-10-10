"""Apply a freshly revalidated measured choice only to an explicit local P3 launch."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from vllm_apple.phase_probe import PhaseProbeConfig
from vllm_apple.text_benchmark import run_text_benchmark

from .conditions import observe_conditions, require_matching_conditions
from .config import local_identity, selected_settings
from .trace import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--port',type=int,default=19150)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.evidence.stat().st_size > 4*1024**2 or args.receipt.exists():
        parser.error('invalid port, oversized evidence or receipt already exists')
    payload = json.loads(args.evidence.read_text())
    conditions = observe_conditions()
    decision, settings = selected_settings(payload,local_identity(args.model,args.python), conditions)
    root = Path(__file__).resolve().parents[2]
    command = [str(args.python.absolute()),'-m','experiments.p3_mlx.server','--model',
        str(args.model.resolve()),'--host','127.0.0.1','--port',str(args.port),*settings,'--log-level','ERROR']
    receipt = dict(evidence_sha256=sha(args.evidence),decision=decision,command=command,
        operating_conditions=conditions,
        opt_in_application=True,standard_adoption=False,applied_settings_verified=False)
    with args.receipt.open('x') as stream:
        stream.write(json.dumps(receipt,indent=2)+'\n')
    process = subprocess.Popen(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root),
        HF_HUB_OFFLINE='1',VLLM_APPLE_P1_PROFILE='0'),start_new_session=True)
    try:
        base_url = f'http://127.0.0.1:{args.port}'
        deadline = time.monotonic()+90
        while True:
            if process.poll() is not None:
                raise RuntimeError('selected worker exited during startup')
            try:
                with urllib.request.urlopen(base_url+'/v1/models',timeout=1):
                    break
            except (OSError,TimeoutError):
                if time.monotonic() >= deadline:
                    raise RuntimeError('selected worker readiness timed out')
                time.sleep(.1)
        if args.smoke:
            config = PhaseProbeConfig(base_url,str(args.model.resolve()),'Apple-M4-32GiB',
                backend='p3_selected_mlx',maximum_output_tokens=16,target_pid=process.pid,timeout_seconds=30)
            receipt['smoke'] = run_text_benchmark(config,requests=9,concurrency=4)
            with urllib.request.urlopen(base_url+'/vllm-apple/p3',timeout=5) as response:
                receipt['effective_settings'] = json.loads(response.read(65536))
            measured = receipt['effective_settings']
            receipt['applied_settings_verified'] = (
                measured['prefill_step_size']==int(settings[settings.index('--prefill-step-size')+1])
                and measured['decode_concurrency']==4 and measured['prompt_concurrency']==2
                and measured['cache_enabled'] is False)
            if not receipt['applied_settings_verified'] or receipt['smoke']['slo_quality_passed'] != 9:
                raise RuntimeError('selected setting smoke failed')
        else:
            process.wait()
    finally:
        try:
            os.killpg(process.pid,signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=5)
        receipt['exit_code'] = process.returncode
        receipt['identity_unchanged'] = payload['scope']['identity'] == local_identity(args.model,args.python)
        receipt['conditions_after'] = observe_conditions()
        try:
            require_matching_conditions(conditions, receipt['conditions_after'])
            receipt['conditions_unchanged'] = True
        except ValueError:
            receipt['conditions_unchanged'] = False
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    if process.returncode != 0 or not receipt['identity_unchanged'] or not receipt['conditions_unchanged']:
        raise RuntimeError('selected worker shutdown or identity check failed')


if __name__ == '__main__':
    main()
