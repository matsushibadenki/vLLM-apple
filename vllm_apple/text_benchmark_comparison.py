"""Compare two bounded text benchmark reports without overstating qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_REPORT_BYTES = 4 * 1024 * 1024
MINIMUM_COMPARISON_REQUESTS = 30
_BOUND_FIELDS = (
    "workload_sha256", "requests", "concurrency", "maximum_output_tokens",
    "timeout_seconds", "slo", "load_policy", "quality_policy",
    "artifact_identity_sha256", "backend_build_sha256",
    "warmup_requests",
)


def _read_report(path: Path) -> tuple[dict[str, Any], str]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= MAX_REPORT_BYTES:
            raise ValueError("benchmark report must be a bounded regular file")
        raw = path.read_bytes()
        if len(raw) != info.st_size:
            raise ValueError("benchmark report changed while reading")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("benchmark report is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("benchmark report must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _count(report: dict[str, Any], name: str) -> int:
    value = report.get(name)
    requests = report.get("requests")
    if (
        type(value) is not int or type(requests) is not int
        or not 0 <= value <= requests <= 100_000
    ):
        raise ValueError(f"benchmark report has invalid {name}")
    return value


def _metric(report: dict[str, Any], name: str) -> float | None:
    value = report.get(name)
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"benchmark report has invalid {name}")
    return float(value)


def _sha256(value: object, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"benchmark report has invalid {name}")
    return value


def _summary(report: dict[str, Any], digest: str) -> dict[str, object]:
    if report.get("schema_version") != 1 or report.get("report_kind") != "text_http_benchmark":
        raise ValueError("unsupported benchmark report")
    route = report.get("route")
    if not isinstance(route, str) or not route or len(route.encode("utf-8")) > 128:
        raise ValueError("benchmark route is invalid")
    started_at = report.get("started_at")
    if not isinstance(started_at, str) or len(started_at) > 64:
        raise ValueError("benchmark start time is invalid")
    try:
        parsed_started_at = datetime.fromisoformat(started_at)
    except ValueError as error:
        raise ValueError("benchmark start time is invalid") from error
    if parsed_started_at.tzinfo is None:
        raise ValueError("benchmark start time must include a timezone")
    requests = _count(report, "completed") + _count(report, "failed")
    if requests != report["requests"]:
        raise ValueError("benchmark completion counts differ")
    quality = _count(report, "quality_passed")
    slo_quality = _count(report, "slo_quality_passed")
    if slo_quality > quality:
        raise ValueError("benchmark SLO count exceeds quality count")
    errors = report.get("errors")
    if not isinstance(errors, dict) or len(errors) > 64 or any(
        not isinstance(key, str) or not key or type(value) is not int or value < 0
        for key, value in errors.items()
    ):
        raise ValueError("benchmark errors are invalid")
    artifact_digest = _sha256(
        report.get("artifact_identity_sha256"), "artifact identity", optional=True
    )
    backend_digest = _sha256(
        report.get("backend_build_sha256"), "backend build", optional=True
    )
    declared_verified = report.get("artifact_identity_verified") is True
    if declared_verified != (artifact_digest is not None and backend_digest is not None):
        raise ValueError("benchmark identity verification fields disagree")
    p99_reference_only = report.get("e2e_p99_reference_only")
    if type(p99_reference_only) is not bool:
        raise ValueError("benchmark p99 reference flag is invalid")
    warmup = report.get("warmup", {
        "attempted": 0, "completed": 0, "quality_passed": 0, "errors": {},
    })
    if not isinstance(warmup, dict) or set(warmup) != {
        "attempted", "completed", "quality_passed", "errors",
    }:
        raise ValueError("benchmark warmup summary is invalid")
    attempted = warmup["attempted"]
    completed = warmup["completed"]
    warmup_quality = warmup["quality_passed"]
    warmup_errors = warmup["errors"]
    configured_warmup = report.get("warmup_requests", 0)
    if (
        type(configured_warmup) is not int or not 0 <= configured_warmup <= 100
        or type(attempted) is not int or attempted != configured_warmup
        or type(completed) is not int or not 0 <= completed <= attempted
        or type(warmup_quality) is not int or not 0 <= warmup_quality <= completed
        or not isinstance(warmup_errors, dict)
        or any(not isinstance(key, str) or type(value) is not int or value < 0
               for key, value in warmup_errors.items())
        or sum(warmup_errors.values()) != attempted - completed
    ):
        raise ValueError("benchmark warmup summary is invalid")
    return {
        "report_sha256": _sha256(digest, "report digest"),
        "route": route,
        "started_at": started_at,
        "artifact_identity_verified": declared_verified,
        "artifact_identity_sha256": artifact_digest,
        "backend_build_sha256": backend_digest,
        "requests": report["requests"],
        "completed": report["completed"],
        "failed": report["failed"],
        "quality_passed": quality,
        "slo_quality_passed": slo_quality,
        "errors": errors,
        "elapsed_seconds": _metric(report, "elapsed_seconds"),
        "output_tokens_per_second": _metric(report, "output_tokens_per_second"),
        "goodput_tokens_per_second": _metric(report, "goodput_tokens_per_second"),
        "e2e_p99_upper_bound_ms": _metric(report, "e2e_p99_upper_bound_ms"),
        "e2e_p99_reference_only": p99_reference_only,
        "warmup": warmup,
        "warmup_passed": completed == attempted and warmup_quality == attempted,
    }


def _ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return round(candidate / baseline, 6)


def _operating_context_issues(direct: dict, proxy: dict) -> list[str]:
    issues: list[str] = []
    starts = []
    power = set()
    dates = set()
    for route, report in (("direct", direct), ("proxy", proxy)):
        context = report.get("operating_context")
        if not isinstance(context, dict):
            issues.append(f"{route}:context_missing")
            continue
        previous_time = None
        previous_age = None
        for phase in ("before_warmup", "before_measurement", "after_measurement"):
            sample = context.get(phase)
            if not isinstance(sample, dict):
                issues.append(f"{route}:{phase}:missing")
                continue
            if sample.get("thermal_state") != "nominal":
                issues.append(f"{route}:{phase}:thermal_not_nominal")
            source, mode = sample.get("power_source"), sample.get("power_mode")
            if source not in ("AC Power", "Battery Power") or mode not in (
                "automatic", "low_power", "high_power",
            ):
                issues.append(f"{route}:{phase}:power_unknown")
            else:
                power.add((source, mode))
            try:
                observed = datetime.fromisoformat(sample["observed_at"])
                if observed.utcoffset() is None or observed.utcoffset().total_seconds() != 0:
                    raise ValueError("UTC required")
                if previous_time is not None and observed < previous_time:
                    raise ValueError("time reversal")
                previous_time = observed
                if phase == "before_measurement":
                    dates.add(observed.date())
            except (KeyError, TypeError, ValueError):
                issues.append(f"{route}:{phase}:timestamp_invalid")
            age = sample.get("target_process_age_seconds")
            if type(age) is not int or age < 0:
                issues.append(f"{route}:{phase}:process_age_missing")
            else:
                if previous_age is not None and age < previous_age:
                    issues.append(f"{route}:{phase}:process_age_reversed")
                previous_age = age
                if phase == "before_measurement":
                    starts.append(age)
    if len(power) > 1:
        issues.append("power_conditions_changed")
    if len(dates) > 1:
        issues.append("measurement_dates_differ")
    if len(starts) == 2 and abs(starts[0] - starts[1]) > 5:
        issues.append("process_age_difference_exceeds_5_seconds")
    return issues


def compare_text_benchmarks(
    direct: dict[str, Any], proxy: dict[str, Any], *,
    direct_sha256: str, proxy_sha256: str,
    require_operating_context: bool = False,
) -> dict[str, object]:
    for field in _BOUND_FIELDS:
        if direct.get(field) != proxy.get(field):
            raise ValueError(f"benchmark comparison identity mismatch:{field}")
    direct_summary = _summary(direct, direct_sha256)
    proxy_summary = _summary(proxy, proxy_sha256)
    artifact_verified = bool(
        direct_summary["artifact_identity_verified"]
        and proxy_summary["artifact_identity_verified"]
    )
    quality_comparable = bool(
        direct_summary["quality_passed"] == direct_summary["requests"]
        and proxy_summary["quality_passed"] == proxy_summary["requests"]
    )
    direct_goodput = direct_summary["goodput_tokens_per_second"]
    proxy_goodput = proxy_summary["goodput_tokens_per_second"]
    direct_p99 = direct_summary["e2e_p99_upper_bound_ms"]
    proxy_p99 = proxy_summary["e2e_p99_upper_bound_ms"]
    sample_gate_passed = bool(
        direct_summary["requests"] >= MINIMUM_COMPARISON_REQUESTS
        and proxy_summary["requests"] >= MINIMUM_COMPARISON_REQUESTS
    )
    warmup_comparable = bool(
        direct_summary["warmup_passed"] and proxy_summary["warmup_passed"]
    )
    context_issues = _operating_context_issues(direct, proxy) if require_operating_context else []
    if not warmup_comparable:
        conclusion = "blocked_warmup_failure"
    elif not quality_comparable:
        conclusion = "blocked_quality_failure"
    elif not artifact_verified:
        conclusion = "blocked_artifact_identity_unverified"
    elif direct_goodput is None or proxy_goodput is None:
        conclusion = "blocked_metric_unavailable"
    elif not sample_gate_passed:
        conclusion = "blocked_insufficient_samples"
    elif context_issues:
        conclusion = "blocked_operating_context"
    else:
        conclusion = "comparable"
    return {
        "schema_version": 1,
        "report_kind": "text_route_comparison",
        "workload_sha256": direct["workload_sha256"],
        "identity_fields_matched": True,
        "artifact_identity_verified": artifact_verified,
        "quality_comparable": quality_comparable,
        "warmup_comparable": warmup_comparable,
        **({"operating_context_gate": {
            "passed": not context_issues, "issues": context_issues,
            "maximum_process_age_difference_seconds": 5,
        }} if require_operating_context else {}),
        "first_route": (
            "direct"
            if datetime.fromisoformat(str(direct_summary["started_at"]))
            <= datetime.fromisoformat(str(proxy_summary["started_at"]))
            else "proxy"
        ),
        "minimum_requests_per_route": MINIMUM_COMPARISON_REQUESTS,
        "sample_gate_passed": sample_gate_passed,
        "p99_reference_only": bool(
            direct_summary["e2e_p99_reference_only"]
            or proxy_summary["e2e_p99_reference_only"]
        ),
        "direct": direct_summary,
        "proxy": proxy_summary,
        "ratios": {
            "proxy_to_direct_goodput": _ratio(proxy_goodput, direct_goodput),
            "proxy_to_direct_output_throughput": _ratio(
                proxy_summary["output_tokens_per_second"],
                direct_summary["output_tokens_per_second"],
            ),
            "proxy_to_direct_e2e_p99": _ratio(proxy_p99, direct_p99),
        },
        "conclusion": conclusion,
        "qualification": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", required=True, type=Path)
    parser.add_argument("--proxy", required=True, type=Path)
    parser.add_argument("--require-operating-context", action="store_true")
    arguments = parser.parse_args()
    try:
        direct, direct_digest = _read_report(arguments.direct)
        proxy, proxy_digest = _read_report(arguments.proxy)
        result = compare_text_benchmarks(
            direct, proxy, direct_sha256=direct_digest, proxy_sha256=proxy_digest,
            require_operating_context=arguments.require_operating_context,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["conclusion"] == "comparable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
