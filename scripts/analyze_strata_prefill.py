"""Summarize diverse prefill results without qualifying speed or real OOM safety."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(report):
    speed, memory = defaultdict(list), defaultdict(list)
    errors = []
    for trial in report["trials"]:
        if trial["status"] != "passed":
            continue
        for result in trial["results"]:
            if (result["residency"]["active_leases"] or result["closed_resident_bytes"]
                    or result["residency"]["peak_reserved_bytes"] > result["budget_bytes"]):
                raise ValueError("failed retention contract")
            errors.append(result["maximum_absolute_error"])
            if not math.isfinite(errors[-1]):
                raise ValueError("nonfinite parity result")
            key = (trial["device"], result["case"], trial["maximum_group_rows"])
            if trial["measure"] == "speed":
                speed[key].append(result)
            for observation in result["memory"]:
                memory[(*key, observation["iteration"])].append(observation)
    speed_groups = []
    for (device, case, group_rows), results in speed.items():
        totals = [sum(r["latencies_ns"]) for r in results]
        speed_groups.append(dict(device=device, case=case, maximum_group_rows=group_rows,
                                 runs=len(results), median_of_run_medians_us=statistics.median(
                                     statistics.median(r["latencies_ns"]) / 1000 for r in results),
                                 run_total_spread_percent=100 * (max(totals) - min(totals))
                                 / statistics.median(totals)))
    memory_groups = []
    for (device, case, group_rows, iteration), observations in memory.items():
        def extent(values):
            return [min(values), max(values)]
        memory_groups.append(dict(
            device=device, case=case, maximum_group_rows=group_rows,
            iteration=iteration, runs=len(observations),
            mlx_active_peak_range_bytes=extent([o["mlx_peak_active_bytes"] for o in observations]),
            mlx_peak_increment_range_bytes=extent([
                o["mlx_peak_active_bytes"] - o["before"]["mlx_active_bytes"] for o in observations]),
            sampled_rss_peak_range_bytes=extent([o["sampled_rss_peak_bytes"] for o in observations]),
            mlx_cache_range_bytes=extent([o["mlx_cache_bytes"] for o in observations]),
            phase_count_ranges={phase: extent([o["counts"][phase] for o in observations])
                                for phase in ("expert_eval", "expert_build", "acquire", "mlx_load")},
        ))
    return dict(scope=report["scope"], maximum_absolute_error=max(errors) if errors else None,
                all_devices_available=all(t["status"] == "passed" for t in report["trials"]),
                all_parity_passed=bool(errors) and max(errors) < 1e-5,
                speed_qualified=False, real_oom_qualified=False,
                speed_groups=speed_groups, memory_groups=memory_groups, limitations=report["notes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = args.report.read_bytes()
    result = summarize(json.loads(data))
    result.update(report=str(args.report), report_sha256=hashlib.sha256(data).hexdigest())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(report=str(args.output), all_parity_passed=result["all_parity_passed"])))


if __name__ == "__main__":
    main()
