"""Evidence-bound P0 baseline aggregation; never promotes backend capabilities."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .text_benchmark_comparison import _read_report


def build_p0_audit(reports: list[tuple[dict, str]]) -> dict:
    if not 1 <= len(reports) <= 64:
        raise ValueError("P0 audit requires 1 to 64 bounded reports")
    issues = []
    groups: dict[str, list[dict]] = {}
    identities = set()
    seen = set()
    for report, digest in reports:
        if not isinstance(report, dict):
            raise ValueError("baseline report must be an object")
        route = report.get("route")
        if not isinstance(route, str) or not route:
            raise ValueError("baseline route is required")
        if report.get("report_kind") != "text_http_benchmark" or report.get("schema_version") != 1:
            raise ValueError("unsupported baseline report")
        for key in ("workload_sha256", "artifact_identity_sha256", "backend_build_sha256"):
            value = report.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                issues.append(f"{route}: missing identity {key}")
        if report.get("artifact_identity_verified") is not True:
            issues.append(f"{route}: artifact identity not verified")
        profile = report.get("phase_profile", {})
        if not isinstance(profile, dict):
            raise ValueError("phase profile must be an object")
        if not isinstance(profile.get("hardware_fingerprint"), str) or not profile['hardware_fingerprint']:
            issues.append(f"{route}: missing hardware identity")
        identities.add((report.get("workload_sha256"), report.get("artifact_identity_sha256"),
                        profile.get("hardware_fingerprint")))
        reproduction = report.get("reproduction", {})
        if not isinstance(reproduction, dict):
            raise ValueError("reproduction must be an object")
        if reproduction.get("independent_process") is not True:
            issues.append(f"{route}: independent process not documented")
        key = (route, report.get("started_at"))
        if not isinstance(report.get("started_at"), str) or not report['started_at']:
            issues.append(f"{route}: missing run timestamp")
        if digest in seen or key in seen:
            issues.append(f"{route}: duplicate run evidence")
        seen.update((digest, key))
        attempts = report.get("requests")
        if type(attempts) is not int or attempts < 30:
            issues.append(f"{route}: insufficient attempted requests")
        for field in ("completed", "quality_passed", "slo_quality_passed"):
            if type(report.get(field)) is not int or report[field] != attempts:
                issues.append(f"{route}: incomplete {field}")
        warmup = report.get("warmup", {})
        if not isinstance(warmup, dict):
            raise ValueError("warmup must be an object")
        if (type(warmup.get("attempted")) is not int or warmup['attempted'] < 0
                or warmup.get("attempted") != report.get("warmup_requests")
                or warmup.get("quality_passed") != warmup.get("attempted")
                or warmup.get("completed") != warmup.get("attempted")):
            issues.append(f"{route}: incomplete warmup")
        languages = report.get("languages", {})
        if not isinstance(languages, dict):
            raise ValueError("language results must be an object")
        language_total = 0
        for language in ('en', 'ja', 'zh'):
            counts = languages.get(language, {})
            if not isinstance(counts, dict):
                raise ValueError("language counts must be an object")
            count = counts.get('attempted')
            if (type(count) is not int or count < 1 or any(
                    type(counts.get(field)) is not int or counts[field] != count
                    for field in ('completed', 'quality_passed', 'slo_passed'))):
                issues.append(f"{route}: incomplete {language} results")
            if type(count) is int:
                language_total += count
        if language_total != attempts:
            issues.append(f"{route}: language count mismatch")
        groups.setdefault(route, []).append(dict(
            report_sha256=digest, started_at=report.get("started_at"),
            backend_build_sha256=report.get("backend_build_sha256"),
            completed=report.get("completed"), languages=languages,
            goodput=report.get("goodput_tokens_per_second"),
            phase_profile=profile, operating_context=report.get("operating_context"),
            p99_reference_only=report.get("e2e_p99_reference_only")))
    if len(identities) != 1:
        issues.append("workload, artifact or hardware mismatch")
    if len(groups) < 2:
        issues.append("fewer than two backends")
    backends = {}
    for route, runs in groups.items():
        if len(runs) < 3:
            issues.append(f"{route}: fewer than three independent runs")
        if len({r['backend_build_sha256'] for r in runs}) != 1:
            issues.append(f"{route}: backend build changed")
        complete = sum(r['completed'] for r in runs if type(r['completed']) is int)
        if complete < 100:
            issues.append(f"{route}: fewer than 100 completed requests")
        language_counts = {}
        for language in ('en', 'ja', 'zh'):
            counts = [r['languages'].get(language, {}).get('completed') for r in runs]
            language_counts[language] = sum(count for count in counts if type(count) is int and count >= 0)
            if language_counts[language] < 100:
                issues.append(f"{route}: fewer than 100 {language} completions")
        values = [r['goodput'] for r in runs if type(r['goodput']) in (int, float)
                  and math.isfinite(r['goodput']) and r['goodput'] > 0]
        if len(values) != len(runs):
            issues.append(f"{route}: missing goodput")
        median = statistics.median(values) if values else None
        backends[route] = dict(runs=runs, completed=complete, language_completions=language_counts,
            median_goodput=median, relative_spread=(max(values)-min(values))/median if values else None)
    return dict(schema_version=1, report_kind='p0_baseline_audit',
                baseline_gate_passed=not issues, issues=issues, backends=backends,
                performance_qualification=False, capability_promotion=False,
                unavailable_internal_phases=['queue', 'tokenize', 'prefill', 'decode', 'serialization', 'model_load'],
                client_phase_note='TTFT includes queue/tokenize/prefill; TPOT is a client-derived estimate')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', type=Path)
    args = parser.parse_args()
    result = build_p0_audit([_read_report(path) for path in args.reports])
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['baseline_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
