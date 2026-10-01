import json
import tempfile
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.text_benchmark_comparison import _read_report, compare_text_benchmarks


def report(**changes):
    value = {
        "schema_version": 1, "report_kind": "text_http_benchmark",
        "route": "direct",
        "started_at": "2026-09-28T00:00:00+00:00",
        "workload_sha256": "a" * 64, "requests": 30, "concurrency": 2,
        "maximum_output_tokens": 16, "timeout_seconds": 30,
        "slo": {"ttft_ms": 1000, "e2e_ms": 5000},
        "load_policy": "closed_loop", "quality_policy": "arithmetic_trimmed_exact",
        "artifact_identity_verified": True, "completed": 30, "failed": 0,
        "artifact_identity_sha256": "d" * 64, "backend_build_sha256": "e" * 64,
        "warmup_requests": 3,
        "warmup": {"attempted": 3, "completed": 3, "quality_passed": 3, "errors": {}},
        "quality_passed": 30, "slo_quality_passed": 30, "errors": {},
        "elapsed_seconds": 3.0, "output_tokens_per_second": 20.0,
        "goodput_tokens_per_second": 18.0, "e2e_p99_upper_bound_ms": 100.0,
        "e2e_p99_reference_only": True,
    }
    value.update(changes)
    return value


class TextBenchmarkComparisonTests(unittest.TestCase):
    def test_operating_context_gate_blocks_missing_and_accepts_matched_conditions(self):
        def compare(a, b):
            return compare_text_benchmarks(
                a, b, direct_sha256="b" * 64, proxy_sha256="c" * 64,
                require_operating_context=True,
            )
        self.assertEqual(compare(report(), report())["conclusion"], "blocked_operating_context")
        context = {phase: {
            "observed_at": "2026-10-01T00:00:00+00:00", "thermal_state": "nominal",
            "power_source": "Battery Power", "power_mode": "automatic",
            "target_process_age_seconds": 20,
        } for phase in ("before_warmup", "before_measurement", "after_measurement")}
        a = report(operating_context=context)
        matched = compare(a, a)
        self.assertEqual(matched["conclusion"], "comparable")
        schema = json.loads(Path("schemas/runtime/text-route-comparison-v1.schema.json").read_text())
        validate_instance(matched, schema)
        for field, value in (("thermal_state", "unknown"), ("power_source", "AC Power"),
                             ("target_process_age_seconds", 30),
                             ("observed_at", "2026-10-02T00:00:00+00:00")):
            changed = json.loads(json.dumps(context))
            changed["before_measurement"][field] = value
            with self.subTest(field=field):
                self.assertEqual(compare(a, report(operating_context=changed))["conclusion"],
                                 "blocked_operating_context")

    def test_matching_quality_reports_are_compared_and_schema_valid(self):
        comparison = compare_text_benchmarks(
            report(), report(
                route="proxy", goodput_tokens_per_second=19.8,
                started_at="2026-09-28T00:01:00+00:00",
                e2e_p99_upper_bound_ms=110,
            ),
            direct_sha256="b" * 64, proxy_sha256="c" * 64,
        )
        self.assertEqual(comparison["conclusion"], "comparable")
        self.assertEqual(comparison["ratios"]["proxy_to_direct_goodput"], 1.1)
        self.assertEqual(comparison["ratios"]["proxy_to_direct_e2e_p99"], 1.1)
        self.assertTrue(comparison["sample_gate_passed"])
        self.assertTrue(comparison["p99_reference_only"])
        self.assertEqual(comparison["first_route"], "direct")
        schema = json.loads(Path(
            "schemas/runtime/text-route-comparison-v1.schema.json"
        ).read_text())
        validate_instance(comparison, schema)

    def test_unverified_artifact_blocks_performance_conclusion(self):
        comparison = compare_text_benchmarks(
            report(
                artifact_identity_verified=False,
                artifact_identity_sha256=None,
                backend_build_sha256=None,
            ),
            report(
                artifact_identity_verified=False,
                artifact_identity_sha256=None,
                backend_build_sha256=None,
            ),
            direct_sha256="b" * 64, proxy_sha256="c" * 64,
        )
        self.assertEqual(
            comparison["conclusion"], "blocked_artifact_identity_unverified"
        )
        self.assertFalse(comparison["qualification"])

    def test_quality_failure_has_priority_and_remains_in_denominator(self):
        comparison = compare_text_benchmarks(
            report(), report(quality_passed=29, slo_quality_passed=29),
            direct_sha256="b" * 64, proxy_sha256="c" * 64,
        )
        self.assertEqual(comparison["conclusion"], "blocked_quality_failure")
        self.assertEqual(comparison["proxy"]["quality_passed"], 29)

    def test_warmup_failure_blocks_comparison(self):
        failed_warmup = {
            "attempted": 3, "completed": 2, "quality_passed": 2,
            "errors": {"backend_unavailable": 1},
        }
        comparison = compare_text_benchmarks(
            report(), report(warmup=failed_warmup),
            direct_sha256="b" * 64, proxy_sha256="c" * 64,
        )
        self.assertEqual(comparison["conclusion"], "blocked_warmup_failure")
        self.assertFalse(comparison["warmup_comparable"])

    def test_warmup_summary_must_match_configured_count(self):
        malformed = report(warmup={
            "attempted": 2, "completed": 2, "quality_passed": 2, "errors": {},
        })
        with self.assertRaisesRegex(ValueError, "warmup summary"):
            compare_text_benchmarks(
                report(), malformed,
                direct_sha256="b" * 64, proxy_sha256="c" * 64,
            )

    def test_short_smoke_retains_ratios_but_blocks_conclusion(self):
        comparison = compare_text_benchmarks(
            report(requests=9, completed=9, quality_passed=9, slo_quality_passed=9),
            report(requests=9, completed=9, quality_passed=9, slo_quality_passed=9),
            direct_sha256="b" * 64, proxy_sha256="c" * 64,
        )
        self.assertEqual(comparison["conclusion"], "blocked_insufficient_samples")
        self.assertFalse(comparison["sample_gate_passed"])
        self.assertIsNotNone(comparison["ratios"]["proxy_to_direct_goodput"])

    def test_identity_and_count_mismatches_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity mismatch:concurrency"):
            compare_text_benchmarks(
                report(), report(concurrency=1),
                direct_sha256="b" * 64, proxy_sha256="c" * 64,
            )

    def test_report_reader_rejects_symlink_and_hashes_exact_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "report.json"
            source.write_text(json.dumps(report()))
            payload, digest = _read_report(source)
            self.assertEqual(payload["requests"], 30)
            self.assertEqual(len(digest), 64)
            link = root / "link.json"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "bounded regular file"):
                _read_report(link)
        with self.assertRaisesRegex(ValueError, "identity mismatch:artifact_identity_sha256"):
            compare_text_benchmarks(
                report(), report(artifact_identity_sha256="f" * 64),
                direct_sha256="b" * 64, proxy_sha256="c" * 64,
            )
        with self.assertRaisesRegex(ValueError, "completion counts differ"):
            compare_text_benchmarks(
                report(), report(completed=29),
                direct_sha256="b" * 64, proxy_sha256="c" * 64,
            )


if __name__ == "__main__":
    unittest.main()
