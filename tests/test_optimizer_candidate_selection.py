import json
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.optimizer import CandidateEvidence, compare_candidates


def evidence(
    candidate_id: str,
    *,
    approved: bool = True,
    score: float = 0.9,
    size: int = 1_000,
    throughput: float = 20.0,
    rss: int = 2_000,
) -> CandidateEvidence:
    return CandidateEvidence(candidate_id, approved, score, size, throughput, rss)


class CandidateSelectionTests(unittest.TestCase):
    def test_quality_gate_is_hard_requirement_and_report_is_schema_valid(self) -> None:
        report = compare_candidates(
            (
                evidence("rejected", approved=False, score=1.0, size=1),
                evidence("winner", score=0.95, size=900),
                evidence("runner-up", score=0.90, size=800),
            ),
            created_at="2026-08-30T00:00:00+00:00",
        )
        self.assertEqual(report.winner_candidate_id, "winner")
        self.assertEqual([item.rank for item in report.candidates], [1, 2, None])
        self.assertEqual(report.candidates[-1].rejection_reasons, ("quality_gate_failed",))
        schema = json.loads(
            Path("schemas/optimizer/candidate-comparison-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        validate_instance(report.to_dict(), schema)

    def test_ties_use_size_throughput_rss_and_identifier_in_order(self) -> None:
        candidates = (
            evidence("identifier-z", size=800, throughput=30, rss=1_000),
            evidence("slow", size=700, throughput=20, rss=1_000),
            evidence("large", size=900, throughput=100, rss=100),
            evidence("identifier-a", size=800, throughput=30, rss=1_000),
            evidence("high-rss", size=800, throughput=30, rss=2_000),
            evidence("fast", size=700, throughput=30, rss=5_000),
        )
        report = compare_candidates(candidates, created_at="2026-08-30T00:00:00+00:00")
        self.assertEqual(
            [item.evidence.candidate_id for item in report.candidates],
            ["fast", "slow", "identifier-a", "identifier-z", "high-rss", "large"],
        )

    def test_order_is_deterministic_and_invalid_sets_fail_closed(self) -> None:
        left = evidence("left", score=0.8)
        right = evidence("right", score=0.9)
        first = compare_candidates((left, right), created_at="2026-08-30T00:00:00+00:00")
        second = compare_candidates((right, left), created_at="2026-08-30T00:00:00+00:00")
        self.assertEqual(first.to_dict(), second.to_dict())
        with self.assertRaisesRegex(ValueError, "unique"):
            compare_candidates((left, left))
        with self.assertRaisesRegex(ValueError, "quality"):
            compare_candidates((evidence("bad", approved=False),))
        with self.assertRaisesRegex(ValueError, "bounded"):
            compare_candidates(tuple(evidence(str(index)) for index in range(65)))


if __name__ == "__main__":
    unittest.main()
