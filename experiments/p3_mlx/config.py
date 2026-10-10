"""Fixed, opt-in P3 settings; recompute the gate instead of trusting a winner label."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from vllm_apple.p3_selection import Candidate, SelectionPolicy, Trial, select_candidate

SETTINGS = {'baseline-512': 512, 'candidate-128': 128}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def local_identity(model, python):
    from vllm_apple.hardware import _ioreg_gpu_core_count, _positive_int, _sysctl

    from .trace import sha
    root = Path(__file__).resolve().parents[2]
    files = [*sorted((root/'experiments/p2_mlx').glob('*.py')),
        *sorted((root/'experiments/p3_mlx').glob('*.py')),
        *sorted((root/'vllm_apple').glob('*.py')),
        *sorted(p for p in Path(model).resolve().iterdir() if p.is_file()),
        Path(python).absolute()]
    # Package source/metadata drift matters even when the interpreter is unchanged.
    site = Path(python).absolute().parent.parent/'lib/python3.12/site-packages'
    files += sorted((site/'mlx_lm').rglob('*.py'))
    files += sorted((site/'mlx').rglob('*.so'))
    files += sorted((site/'mlx').rglob('*.dylib'))
    files += sorted(site.glob('mlx*-*.dist-info/METADATA'))
    return dict(files={str(p.resolve()):sha(p) for p in files},
        hardware=dict(architecture=platform.machine(), os=platform.mac_ver()[0],
            soc=subprocess.check_output(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
            memory_bytes=int(subprocess.check_output(['/usr/sbin/sysctl','-n','hw.memsize'],text=True)),
            os_build=subprocess.check_output(['/usr/sbin/sysctl','-n','kern.osversion'],text=True).strip(),
            gpu_core_count=_positive_int(_sysctl('hw.perflevel0.gpu_count')) or _ioreg_gpu_core_count()))


def evaluate(payload):
    policy = dict(payload['policy'])
    policy['quality_slices'] = tuple(policy['quality_slices'])
    policy = SelectionPolicy(**policy)

    def candidate(value):
        trials = []
        for raw in value['trials']:
            raw = dict(raw)
            raw['quality'] = tuple(tuple(pair) for pair in raw['quality'])
            trials.append(Trial(**raw))
        return Candidate(value['candidate_id'], value['kind'], tuple(trials))

    if payload['baseline']['candidate_id'] != 'baseline-512' or [c['candidate_id'] for c in payload['candidates']] != ['candidate-128']:
        raise ValueError('unsupported setting candidates')
    if payload['baseline']['kind'] != 'baseline' or payload['candidates'][0]['kind'] != 'kernel':
        raise ValueError('unsupported candidate kind')
    if policy.scope_sha256 != digest(payload['scope']):
        raise ValueError('scope digest mismatch')
    if (payload['scope']['gpu_profile_sha256'] != digest(payload['gpu_profile'])
            or payload['scope']['identity'] != payload['gpu_profile']['identity']
            or payload['gpu_profile'].get('identity_unchanged') is not True):
        raise ValueError('GPU evidence mismatch')
    if payload['gpu_profile'].get('suggested_tuning_dimension') != 'prefill_step_size':
        raise ValueError('GPU profile does not support this tuning dimension')
    if ('operating_conditions' in payload['scope']
            and payload['gpu_profile'].get('operating_conditions') != payload['scope']['operating_conditions']):
        raise ValueError('GPU and HTTP operating conditions differ')
    return select_candidate(policy,candidate(payload['baseline']),
        tuple(candidate(v) for v in payload['candidates']),
        independent_acquisition_verified=payload.get('independent_acquisition_verified',False),
        prerequisites_verified=False)


def selected_settings(payload, current_identity, current_conditions=None):
    from .conditions import require_matching_conditions

    if payload['scope']['identity'] != current_identity:
        raise ValueError('measured hardware/runtime/model identity is stale')
    cores = current_identity.get('hardware', {}).get('gpu_core_count')
    if type(cores) is not int or not 0 < cores <= 512:
        raise ValueError('GPU core count missing or invalid')
    require_matching_conditions(payload['scope'].get('operating_conditions'), current_conditions)
    decision = evaluate(payload)
    selected = decision['selected_candidate_id']
    return decision, ['--compute-dtype','float32','--scheduler-budget-ms','50',
        '--decode-concurrency','4','--prompt-concurrency','2',
        '--prefill-step-size',str(SETTINGS[selected]),'--prompt-cache-size','4', '--disable-prefix-cache']
