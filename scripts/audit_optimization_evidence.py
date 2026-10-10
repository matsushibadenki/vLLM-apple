"""Recheck optimization receipts without running inference or granting qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def check_files(root: Path, files: dict[str, str]) -> list[dict]:
    results = []
    for name, expected in sorted(files.items()):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('receipt path escapes repository')
        try:
            hasher = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        except FileNotFoundError:
            digest = None
        results.append(dict(path=name, matches=digest == expected,
                            expected=expected, actual=digest))
    return results


def audit(root: Path) -> dict:
    directory = root / 'docs/evaluation'
    p1 = json.loads((directory / 'p1-spm-final-runtime-2026-10-10.json').read_text())
    p2 = json.loads((directory / 'p2-optimization-validation-2026-10-10.json').read_text())
    p3 = json.loads((directory / 'p3-integration-validation-2026-10-10-r2.json').read_text())
    groups = {
        'p1_runtime': p1['runtime_sources'],
        'p1_runners': p1['runner_sources'],
        'p1_model': {str(Path(p1['model_directory']) / name): value
                     for name, value in p1['model_files'].items()},
        'p2_artifacts': p2['artifacts'],
        'p2_collectors': p2['validation_tools'],
        'p3_artifacts': p3['artifacts'],
        'p3_sources': p3['source_hashes'],
    }
    checks = {key: check_files(root, value) for key, value in groups.items()}
    from experiments.p3_mlx.config import evaluate, local_identity
    evidence = json.loads((directory / 'p3-real-selection-m4-2026-10-10-r3/evidence.json').read_text())
    selection = json.loads((directory / 'p3-real-selection-m4-2026-10-10-r3/selection.json').read_text())
    recomputed = json.loads(json.dumps(evaluate(evidence)))
    selection_matches = recomputed == selection
    current_identity = local_identity(root / p1['model_directory'], Path(p1['native_python']))
    identity_matches = current_identity == evidence['scope']['identity']
    return dict(recorded_at=datetime.now(timezone.utc).isoformat(),
                scope='offline source/artifact and selector consistency only; no new hardware qualification',
                passed=all(row['matches'] for rows in checks.values() for row in rows)
                and selection_matches and identity_matches,
                checks=checks, selection_recomputed_matches=selection_matches,
                current_hardware_runtime_model_identity_matches=identity_matches,
                selected_candidate=recomputed['selected_candidate_id'],
                qualification=False, long_run_qualification=False,
                remaining=['P1 scheduled long-run gates', 'P2 broad workload and cached c1 tails',
                           'P3 hardware counters, quality, energy and standard adoption',
                           'P4 full matrix and 24-hour qualification'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1])
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('passed', 'qualification', 'selected_candidate')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
