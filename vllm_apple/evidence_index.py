"""Hash-index heterogeneous reports without interpreting them as qualifications."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .text_benchmark_comparison import _read_report


def build_evidence_index(paths: list[Path]) -> dict:
    if not 1 <= len(paths) <= 256:
        raise ValueError("evidence index requires 1 to 256 reports")
    entries = []
    for path in paths:
        report, digest = _read_report(path)
        profile = report.get('phase_profile')
        entries.append(dict(path=str(path), sha256=digest,
            report_kind=report.get('report_kind', 'legacy_untyped'),
            recorded_passed=report.get('passed'), recorded_qualification=report.get('qualification'),
            recorded_scope=report.get('qualification_scope'),
            phase_profile=profile if isinstance(profile, dict) else None,
            availability=dict(queue=None, tokenize=None, backend_prefill=None,
                              backend_decode=None, model_load=None, serialization=None),
            timing_provenance='client arrival times when phase_profile exists; other phases unavailable'))
    return dict(schema_version=1, report_kind='common_evidence_index', evidence=entries,
                qualification=False, transfers_qualification=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(build_evidence_index(args.reports), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
