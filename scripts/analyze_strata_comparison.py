"""Summarize isolated Strata trials without treating noisy CPU timing as qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(report, *, maximum_spread_percent=5.0):
    groups = defaultdict(list)
    for trial in report["trials"]:
        groups[(trial["variant"], trial["policy"])].append(trial)
    results = []
    for (variant, policy), trials in groups.items():
        totals = [sum(t["request_nanoseconds"]) for t in trials]
        spread = 100 * (max(totals) - min(totals)) / statistics.median(totals)
        residents = [t for t in trials if t["residency"] is not None]
        deltas = [{key: t["residency"][key] - t["initial_residency"][key]
                   for key in ("hits", "misses", "evictions", "rejections")}
                  for t in residents]
        contracts = all(t["residency"]["active_leases"] == 0
                        and t["residency"]["peak_resident_bytes"] <= t["resident_budget_bytes"]
                        and t["residency"].get("peak_reserved_bytes", 0) <= t["resident_budget_bytes"]
                        and t["closed_resident_bytes"] == 0 for t in residents)
        gathers = [t for t in trials if t.get("gather", False)]
        contracts = contracts and all(t["gather_resident_bytes"] <= t["resident_budget_bytes"]
                                      and t["closed_gather_bytes"] == 0 for t in gathers)
        parity = all(t["parity_passed"] for t in trials)
        results.append(dict(
            variant=variant, policy=policy, runs=len(trials),
            timed_requests=sum(len(t["request_nanoseconds"]) for t in trials),
            median_of_run_medians_us=statistics.median([
                statistics.median(t["request_nanoseconds"]) / 1000 for t in trials]),
            run_total_spread_percent=spread,
            timing_stable=len(trials) >= 3 and spread <= maximum_spread_percent,
            host_load_passed=bool(report.get("quiet_host_requested")) and all(
                t.get("request_host_load" if report.get("host_load_window") == "requests_only"
                      else "host_load", {}).get("passed", False) for t in trials),
            startup_host_passed_runs=sum(t.get("startup_host_load", {}).get("passed", False)
                                         for t in trials),
            request_host_passed_runs=sum(t.get("request_host_load", {}).get("passed", False)
                                         for t in trials),
            parity_passed=parity, residency_contract_passed=contracts,
            peak_retained_bytes=max((t["residency"]["peak_resident_bytes"]
                                     for t in residents), default=None),
            peak_reserved_bytes=max((t["residency"].get("peak_reserved_bytes")
                                     for t in residents
                                     if "peak_reserved_bytes" in t["residency"]), default=None),
            gather_retained_bytes=max((t["gather_resident_bytes"] for t in gathers), default=None),
            request_residency_deltas=deltas,
        ))
    return dict(scope=report["scope"], maximum_spread_percent=maximum_spread_percent,
                spread_definition="100 * (max - min) / median of run request totals",
                campaign_completed=report.get("campaign_completed", True),
                timing_stable=bool(results) and report.get("campaign_completed", True)
                and all(g["timing_stable"] for g in results),
                uncontended_comparison_passed=bool(results)
                and report.get("campaign_completed", True)
                and all(g["timing_stable"] and g["host_load_passed"] for g in results),
                standard_adoption_eligible=False, groups=results,
                limitations=["Synthetic CPU block, not model/GPU/E2E qualification",
                             "Retained tensor peak excludes transient loading and RSS",
                             "Baseline retains all source weights even in memory suite",
                             "Warmup/export/startup/parity checks excluded from timed spans"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.report.read_bytes()
    result = summarize(json.loads(source))
    result["report"] = str(args.report)
    result["report_sha256"] = hashlib.sha256(source).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
