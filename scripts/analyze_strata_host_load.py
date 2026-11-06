"""Attribute saved Strata host-gate failures without requalifying any trial."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.strata_measurement import quiet


def analyze(paths):
    phases = {}
    sources = {}
    for path in sorted(set(map(Path, paths))):
        content = path.read_bytes()
        report = json.loads(content)
        sources[str(path)] = hashlib.sha256(content).hexdigest()
        windows = []
        if "trials" in report:
            for trial in report["trials"]:
                for phase, key in (("startup", "startup_host_load"),
                                   ("ready_wait", "request_start_gate"),
                                   ("requests", "request_host_load")):
                    window = trial.get(key)
                    if window is not None:
                        windows.append((phase, window))
            windows.extend((gate.get("phase", "pre_worker"), gate)
                           for gate in report.get("host_gates", []))
        elif "observations" in report:
            windows.append(("preflight", report))
        for phase, window in windows:
            counts, reasons, offenders = phases.setdefault(phase, (Counter(), Counter(), Counter()))
            counts["windows"] += 1
            counts["sampling_errors"] += bool(window.get("sampling_error"))
            for observation in window.get("observations", []):
                counts["samples"] += 1
                failed = []
                for key, limit in (("total_percent", 200), ("maximum_process_percent", 75),
                                   ("new_or_exited_processes", 0)):
                    if observation[key] > limit:
                        counts[key] += 1
                        failed.append(key)
                if not quiet(observation):
                    counts["failed_samples"] += 1
                    reasons["+".join(failed)] += 1
                for process in observation.get("top_processes", []):
                    if process["cpu_percent"] > 75:
                        offenders[process["name"]] += 1
                counts["attributed_samples"] += "observer_cpu_percent" in observation
    return dict(
        source_sha256=sources,
        thresholds=dict(total_percent=200, maximum_process_percent=75,
                        new_or_exited_processes=0),
        phases={phase: dict(counts=dict(counts), failure_combinations=dict(reasons),
                            over_75_process_samples=dict(offenders.most_common()))
                for phase, (counts, reasons, offenders) in phases.items()},
        qualification_changed=False,
        limitations=["Reasons overlap; process names are observations, not proof of causation",
                     "Legacy records omit observer/worker CPU; aggregate excess is not attributable",
                     "Top five processes omit other/inaccessible activity; do not subtract them",
                     "Startup, READY wait, preflight and requests remain separate",
                     "No latency-based filtering, thresholds changed or CPU/GPU/E2E qualification"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.reports)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report["phases"], ensure_ascii=False))


if __name__ == "__main__":
    main()
