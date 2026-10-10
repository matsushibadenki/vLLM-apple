"""GPU-guided real HTTP trials, preserving failures and recomputing the P3 gate."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from scripts.compare_p2_mlx import SampledProfiler, percentile
from vllm_apple import text_benchmark
from vllm_apple.p3_selection import SelectionPolicy
from vllm_apple.phase_probe import PhaseProbeConfig

from .conditions import observe_conditions, require_matching_conditions
from .config import SETTINGS, digest, evaluate, local_identity


def cases():
    prefix = 'Reference marker. '*128
    return tuple((lang,prefix+prompt,'2') for lang,prompt in (
        ('en','Ignore the markers. What is 1+1? Reply only with the digit.'),
        ('ja','参照情報は無視。1+1は？数字だけで答えてください。'),
        ('zh','忽略标记。1+1等于几？只回答数字。')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--gpu-profile',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--port',type=int,default=19149)
    parser.add_argument('--requests',type=int,default=30)
    args = parser.parse_args()
    if not 30 <= args.requests <= 120 or not 1024 <= args.port <= 65535:
        parser.error('requests 30..120 and loopback port 1024..65535 required')
    profile = json.loads(args.gpu_profile.read_text())
    identity = local_identity(args.model,args.python)
    conditions = observe_conditions()
    require_matching_conditions(profile.get('operating_conditions'), conditions)
    if profile.get('identity') != identity or profile.get('identity_unchanged') is not True:
        parser.error('GPU profile identity does not match')
    if profile.get('suggested_tuning_dimension') != 'prefill_step_size':
        parser.error('GPU evidence does not support the fixed candidate dimension')
    output = args.output_directory.resolve()
    output.mkdir(parents=True,exist_ok=False)
    scope = dict(identity=identity,operating_conditions=conditions,gpu_profile_sha256=digest(profile),
        workload_sha256=digest(cases()),requests=args.requests,concurrency=[1,4],
        precision='float32',sampling='temperature-zero',cache_policy='disabled',
        memory_measurement='MLX allocator peak active; not process RSS or physical peak',
        shape_scope='fixed three-language reference-prefix arithmetic; output<=16')
    policy = SelectionPolicy(digest(scope),('en','ja','zh','slo','warmup','identity','shutdown','effective_settings'),8*1024**3)
    envelope = dict(policy=asdict(policy),scope=scope,gpu_profile=profile,
        baseline=dict(candidate_id='baseline-512',kind='baseline',trials=[]),
        candidates=[dict(candidate_id='candidate-128',kind='kernel',trials=[])],
        independent_acquisition_verified=False,prerequisites_verified=False)
    # Write policy before acquiring any candidate measurements.
    (output/'policy.json').write_text(json.dumps(envelope,indent=2)+'\n')
    root = Path(__file__).resolve().parents[2]
    base_url = f'http://127.0.0.1:{args.port}'
    text_benchmark.ExecutionPhaseProfiler = SampledProfiler
    for index, candidate in enumerate(('baseline-512','candidate-128','candidate-128','baseline-512','baseline-512','candidate-128')):
        require_matching_conditions(conditions, observe_conditions())
        step = SETTINGS[candidate]
        command = [str(args.python.absolute()),'-m','experiments.p3_mlx.server',
            '--model',str(args.model.resolve()),'--host','127.0.0.1','--port',str(args.port),
            '--compute-dtype','float32','--scheduler-budget-ms','50','--decode-concurrency','4',
            '--prompt-concurrency','2','--prefill-step-size',str(step),
            '--prompt-cache-size','4','--disable-prefix-cache','--log-level','ERROR']
        row = dict(index=index,candidate_id=candidate,command=command)
        with (output/f'{index}.log').open('w') as log:
            process = subprocess.Popen(command,cwd=root,stdout=log,stderr=log,
                env=dict(os.environ,PYTHONPATH=str(root),HF_HUB_OFFLINE='1',VLLM_APPLE_P1_PROFILE='0'),start_new_session=True)
            try:
                deadline = time.monotonic()+90
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('worker exited; see trial log')
                    try:
                        with urllib.request.urlopen(base_url+'/v1/models',timeout=1):
                            break
                    except (OSError,TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError('worker readiness timed out')
                        time.sleep(.1)
                config = PhaseProbeConfig(base_url,str(args.model.resolve()),'Apple-M4-32GiB',
                    backend='p3_experimental_mlx',maximum_output_tokens=16,target_pid=process.pid,timeout_seconds=30)
                row['warmup'] = text_benchmark.run_text_benchmark(config,requests=3,cases=cases())
                for concurrency in (1,4):
                    row[f'c{concurrency}'] = text_benchmark.run_text_benchmark(config,
                        requests=args.requests,concurrency=concurrency,cases=cases())
                    print(json.dumps(dict(index=index,candidate=candidate,concurrency=concurrency,
                        quality=row[f'c{concurrency}']['quality_passed'],slo=row[f'c{concurrency}']['slo_quality_passed'])),flush=True)
                with urllib.request.urlopen(base_url+'/vllm-apple/p3',timeout=5) as response:
                    row['memory_and_settings'] = json.loads(response.read(65536))
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
                row['exit_code'] = process.returncode
                row['identity_unchanged'] = identity == local_identity(args.model,args.python)
                row['conditions_after'] = observe_conditions()
                row['conditions_unchanged'] = row['conditions_after'] == conditions and conditions['thermal_state'] == 'nominal'
                (output/f'{index}.json').write_text(json.dumps(row,indent=2)+'\n')
        memory = row['memory_and_settings']
        if type(memory.get('peak_active_bytes')) is not int or memory['peak_active_bytes'] <= 0:
            raise ValueError('real allocator peak is missing')
        quality = [[lang,all(r['languages'][lang]['quality_passed']==r['languages'][lang]['attempted'] for r in (row['c1'],row['c4']))] for lang in ('en','ja','zh')]
        quality += [['slo',all(row[f'c{c}']['slo_quality_passed']==args.requests for c in (1,4))],
            ['warmup',row['warmup']['quality_passed']==3 and row['warmup']['slo_quality_passed']==3],
            ['identity',row['identity_unchanged'] and row['conditions_unchanged']],['shutdown',row['exit_code']==0],
            ['effective_settings',memory['prefill_step_size']==step and memory['decode_concurrency']==4 and memory['prompt_concurrency']==2 and memory['cache_enabled'] is False]]
        trial = dict(acquisition_id=f'{output.name}:{index}:pid-{process.pid}',
            scope_sha256=policy.scope_sha256,policy_sha256=policy.digest,
            e2e_seconds=row['c4']['elapsed_seconds'],
            ttft_p95_seconds=max(percentile(row[f'c{c}'],'ttft_ms') for c in (1,4))/1000,
            tpot_p95_seconds=max(percentile(row[f'c{c}'],'tpot_ms') for c in (1,4))/1000,
            peak_memory_bytes=memory['peak_active_bytes'],quality=quality)
        target = envelope['baseline'] if candidate=='baseline-512' else envelope['candidates'][0]
        target['trials'].append(trial)
        (output/'evidence.json').write_text(json.dumps(envelope,indent=2)+'\n')
    # Serial fresh workers, independent model/cache construction and clean exits are audited.
    # This is not a claim that the host was interference-free.
    envelope['independent_acquisition_verified'] = True
    envelope['acquisition_scope'] = 'six serial fresh processes; cold caches; workload fixed before acquisition; host exclusivity unverified'
    (output/'evidence.json').write_text(json.dumps(envelope,indent=2)+'\n')
    decision = evaluate(envelope)
    (output/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(decision,indent=2))


if __name__ == '__main__':
    main()
