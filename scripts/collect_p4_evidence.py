"""Connect real collector outputs to P4; preserve missing/failed evidence as rejection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vllm_apple.p4_certification import (
    IDENTITY_FIELDS,
    ROLES,
    _hash,
    _load,
    digest,
    verify_certification,
)


def normalize(role, collector, identity, source_commit, artifact_sha256):
    """Only exact scoped P4 collector reports can pass; legacy results stay diagnostic."""
    if (collector.get('p4_identity') == identity and collector.get('role') == role
            and collector.get('source_commit') == source_commit
            and collector.get('artifact_sha256') == artifact_sha256):
        return dict(collector)
    raw = dict(passed=False, identity_verified=False,
               reason='collector lacks exact P4 identity and role', collector=collector)
    if role == 'soak':
        raw.update(duration_seconds=collector.get('elapsed_seconds'),
                   clean_shutdown=collector.get('shutdown_clean') is True)
    if role == 'performance':
        # Retain the actual P3 decision and its original scope; never rewrite its policy.
        raw.update(collector)
        raw['passed'] = False
    if role == 'quality':
        raw['slices'] = {name: False for name in
                         ('en', 'ja', 'zh', 'coding', 'tool', 'long_context')}
        raw['limited_workload_gate_passed'] = collector.get('roadmap_workload_gate_passed') is True
    return raw


def collect(output, identity, source_commit, artifact_sha256, inputs):
    if (set(identity) != set(IDENTITY_FIELDS) or not all(_hash(v) for v in identity.values())
            or not _hash(source_commit, 40) or not _hash(artifact_sha256)
            or set(inputs) - set(ROLES)):
        raise ValueError('invalid identity, source, artifact or role')
    # Validate and load every source before creating any bundle; never overwrite a run.
    sources = {role: _load(path) for role, path in inputs.items()}
    output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        path = output / name
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    refs, inventory = {}, {}
    for role in ROLES:
        if role in sources:
            payload, original_sha = sources[role]
            original = write(role + '-collector.json', payload)
            inventory[role] = dict(input_path=str(inputs[role].absolute()),
                                   input_sha256=original_sha, stored=original)
            raw = normalize(role, payload, identity, source_commit, artifact_sha256)
            # P3 report_id binds every field: do not inject provenance into its payload.
        else:
            raw = dict(passed=False, reason='collector evidence missing')
        raw_ref = write(role + '-raw.json', raw)
        refs[role] = write(role + '.json', dict(
            schema_version=1, role=role, scope_sha256=digest(identity),
            source_commit=source_commit, artifact_sha256=artifact_sha256,
            passed=raw.get('passed') is True, raw_evidence=raw_ref))
    bundle = dict(schema_version=1, source_commit=source_commit,
                  artifact_sha256=artifact_sha256,
                  cells=[dict(identity=identity, evidence=refs)])
    write('bundle.json', bundle)
    write('collector-inventory.json', inventory)
    result = verify_certification(output / 'bundle.json', source_commit=source_commit,
                                  artifact_sha256=artifact_sha256)
    write('certification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--identity', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--artifact-sha256', required=True)
    parser.add_argument('--evidence', action='append', default=[], metavar='ROLE=PATH')
    args = parser.parse_args()
    inputs = {}
    for item in args.evidence:
        role, separator, path = item.partition('=')
        if not separator or not path or role in inputs:
            parser.error('each evidence role must occur once as ROLE=PATH')
        inputs[role] = Path(path)
    identity, _ = _load(args.identity)
    result = collect(args.output_directory, identity, args.source_commit,
                     args.artifact_sha256, inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
