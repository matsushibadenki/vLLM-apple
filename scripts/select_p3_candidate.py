#!/usr/bin/env python3
"""Evaluate bounded collector JSON without running or applying its configurations."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm_apple.p3_selection import Candidate, SelectionPolicy, Trial, select_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    if args.evidence.stat().st_size > 4 * 1024 * 1024:
        parser.error("evidence exceeds 4 MiB")
    payload = json.loads(args.evidence.read_text())
    policy_values = dict(payload["policy"])
    policy_values["quality_slices"] = tuple(policy_values["quality_slices"])
    policy = SelectionPolicy(**policy_values)

    def candidate(value):
        trials = []
        for raw in value["trials"]:
            raw = dict(raw)
            raw["quality"] = tuple(tuple(item) for item in raw["quality"])
            trials.append(Trial(**raw))
        return Candidate(value["candidate_id"], value["kind"], tuple(trials))

    report = select_candidate(
        policy, candidate(payload["baseline"]),
        tuple(candidate(c) for c in payload["candidates"]),
        independent_acquisition_verified=payload.get("independent_acquisition_verified", False),
        prerequisites_verified=payload.get("prerequisites_verified", False))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
