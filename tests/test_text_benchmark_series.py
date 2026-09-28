import json
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.text_benchmark_series import compare_text_benchmark_series


def comparison(order: str, ratio: float, **changes):
    value = {
        "schema_version": 1, "report_kind": "text_route_comparison",
        "workload_sha256": "a" * 64, "conclusion": "comparable",
        "qualification": False, "first_route": order, "p99_reference_only": True,
        "direct": {"artifact_identity_sha256": "b" * 64,
                   "backend_build_sha256": "c" * 64},
        "proxy": {"artifact_identity_sha256": "b" * 64,
                  "backend_build_sha256": "c" * 64},
        "ratios": {"proxy_to_direct_goodput": ratio},
    }
    value.update(changes)
    return value


class TextBenchmarkSeriesTests(unittest.TestCase):
    def test_balanced_stable_series_uses_median_and_stays_unqualified(self):
        result = compare_text_benchmark_series([
            (comparison("direct", 1.01), "d" * 64),
            (comparison("proxy", 0.99), "e" * 64),
            (comparison("direct", 1.00), "f" * 64),
        ])
        self.assertEqual(result["conclusion"], "comparable_stable_reference")
        self.assertEqual(result["proxy_to_direct_goodput"]["median"], 1.0)
        self.assertEqual(result["order"], {"direct": 2, "proxy": 1, "balanced": True})
        self.assertFalse(result["qualification"])
        schema = json.loads(Path(
            "schemas/runtime/text-route-comparison-series-v1.schema.json"
        ).read_text())
        validate_instance(result, schema)

    def test_variance_and_order_are_independent_gates(self):
        varied = compare_text_benchmark_series([
            (comparison("direct", 0.9), "d" * 64),
            (comparison("proxy", 1.1), "e" * 64),
            (comparison("direct", 1.0), "f" * 64),
        ])
        self.assertEqual(varied["conclusion"], "blocked_variance")
        unbalanced = compare_text_benchmark_series([
            (comparison("direct", 1.0), "d" * 64),
            (comparison("direct", 1.0), "e" * 64),
            (comparison("direct", 1.0), "f" * 64),
        ])
        self.assertEqual(unbalanced["conclusion"], "blocked_order_unbalanced")

    def test_identity_noncomparable_and_pair_bounds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 3 and 21"):
            compare_text_benchmark_series([(comparison("direct", 1.0), "d" * 64)])
        bad = comparison("proxy", 1.0)
        bad["direct"]["artifact_identity_sha256"] = "x" * 64
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            compare_text_benchmark_series([
                (comparison("direct", 1.0), "d" * 64),
                (bad, "e" * 64),
                (comparison("direct", 1.0), "f" * 64),
            ])
        with self.assertRaisesRegex(ValueError, "not a comparable"):
            compare_text_benchmark_series([
                (comparison("direct", 1.0), "d" * 64),
                (comparison("proxy", 1.0, conclusion="blocked_variance"), "e" * 64),
                (comparison("direct", 1.0), "f" * 64),
            ])


if __name__ == "__main__":
    unittest.main()
