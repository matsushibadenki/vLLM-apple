"""Aggregate repeated, order-balanced direct-versus-proxy comparisons."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .text_benchmark_comparison import _read_report

MINIMUM_PAIRS = 3
MAXIMUM_PAIRS = 21
MAXIMUM_RELATIVE_SPREAD = 0.05


def compare_text_benchmark_series(
    comparisons: list[tuple[dict[str, object], str]],
) -> dict[str, object]:
    if not MINIMUM_PAIRS <= len(comparisons) <= MAXIMUM_PAIRS:
        raise ValueError("comparison series requires between 3 and 21 pairs")
    first = comparisons[0][0]
    bound = {
        "workload_sha256": first.get("workload_sha256"),
        "artifact_identity_sha256": first.get("direct", {}).get("artifact_identity_sha256")
        if isinstance(first.get("direct"), dict) else None,
        "backend_build_sha256": first.get("direct", {}).get("backend_build_sha256")
        if isinstance(first.get("direct"), dict) else None,
    }
    pairs: list[dict[str, object]] = []
    ratios: list[float] = []
    order_counts = {"direct": 0, "proxy": 0}
    p99_reference_only = False
    for report, digest in comparisons:
        direct = report.get("direct")
        proxy = report.get("proxy")
        ratio_set = report.get("ratios")
        if (
            report.get("schema_version") != 1
            or report.get("report_kind") != "text_route_comparison"
            or report.get("conclusion") != "comparable"
            or report.get("qualification") is not False
            or not isinstance(direct, dict) or not isinstance(proxy, dict)
            or not isinstance(ratio_set, dict)
        ):
            raise ValueError("series input is not a comparable route report")
        current = {
            "workload_sha256": report.get("workload_sha256"),
            "artifact_identity_sha256": direct.get("artifact_identity_sha256"),
            "backend_build_sha256": direct.get("backend_build_sha256"),
        }
        if current != bound or (
            proxy.get("artifact_identity_sha256") != bound["artifact_identity_sha256"]
            or proxy.get("backend_build_sha256") != bound["backend_build_sha256"]
        ):
            raise ValueError("comparison series identity mismatch")
        first_route = report.get("first_route")
        if first_route not in order_counts:
            raise ValueError("comparison series order is invalid")
        ratio = ratio_set.get("proxy_to_direct_goodput")
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("comparison series ratio is invalid")
        ratio = float(ratio)
        ratios.append(ratio)
        order_counts[first_route] += 1
        p99_reference_only = p99_reference_only or report.get("p99_reference_only") is True
        pairs.append({
            "report_sha256": digest,
            "first_route": first_route,
            "proxy_to_direct_goodput": ratio,
        })
    median = statistics.median(ratios)
    minimum = min(ratios)
    maximum = max(ratios)
    relative_spread = (maximum - minimum) / median
    order_balanced = (
        min(order_counts.values()) > 0
        and abs(order_counts["direct"] - order_counts["proxy"]) <= 1
    )
    stable = relative_spread <= MAXIMUM_RELATIVE_SPREAD
    conclusion = (
        "blocked_order_unbalanced" if not order_balanced
        else "blocked_variance" if not stable
        else "comparable_stable_reference"
    )
    return {
        "schema_version": 1,
        "report_kind": "text_route_comparison_series",
        **bound,
        "pair_count": len(pairs),
        "order": {**order_counts, "balanced": order_balanced},
        "pairs": pairs,
        "proxy_to_direct_goodput": {
            "median": round(median, 6),
            "minimum": round(minimum, 6),
            "maximum": round(maximum, 6),
            "relative_spread": round(relative_spread, 6),
            "maximum_allowed_relative_spread": MAXIMUM_RELATIVE_SPREAD,
        },
        "p99_reference_only": p99_reference_only,
        "conclusion": conclusion,
        "qualification": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", nargs="+", type=Path)
    arguments = parser.parse_args()
    try:
        inputs = []
        for path in arguments.comparison:
            report, digest = _read_report(path)
            inputs.append((report, digest))
        result = compare_text_benchmark_series(inputs)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["conclusion"] == "comparable_stable_reference" else 1


if __name__ == "__main__":
    raise SystemExit(main())
