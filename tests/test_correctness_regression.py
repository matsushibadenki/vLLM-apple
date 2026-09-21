from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.optimizer import (
    GenerationEvaluationReport,
    GenerationSampleResult,
    evaluate_real_model_regression,
    generation_token_fingerprint,
)
from vllm_apple.optimizer.cli import main as optimizer_main


def _sample(sample_id: str, language: str, tokens: tuple[int, ...]) -> GenerationSampleResult:
    return GenerationSampleResult(
        sample_id=sample_id,
        domain="chat",
        language=language,
        prompt_token_count=4,
        token_ids=tokens,
        output_fingerprint=generation_token_fingerprint(tokens),
        expectation_score=1.0,
    )


def _report(
    model_hash: str, *, tokens: tuple[int, ...] = (1, 2), elapsed: int = 100
) -> GenerationEvaluationReport:
    return GenerationEvaluationReport(
        model_path="/tmp/model",
        model_hash=model_hash,
        dataset_path="/tmp/dataset.jsonl",
        dataset_fingerprint="d" * 64,
        prompt_format="chat_template",
        maximum_prompt_tokens=32,
        maximum_new_tokens=8,
        elapsed_milliseconds=elapsed,
        peak_rss_bytes=1000,
        samples=tuple(
            _sample(name, language, tokens)
            for name, language in (("en", "en"), ("ja", "ja"), ("zh", "zh-Hans"))
        ),
    )


class CorrectnessRegressionTests(unittest.TestCase):
    def test_approves_repeatable_multilingual_real_model_runs(self) -> None:
        report = evaluate_real_model_regression(
            _report("a" * 64),
            (_report("b" * 64), _report("b" * 64)),
            minimum_token_agreement=1.0,
            maximum_expectation_regression=0.0,
            maximum_latency_regression=0.05,
            maximum_rss_regression=0.05,
        )
        self.assertTrue(report.approved)
        self.assertEqual(report.languages, ("en", "ja", "zh-Hans"))
        self.assertEqual(len(report.report_id), 64)
        self.assertNotIn("token_ids", report.to_dict())
        schema = json.loads(
            Path("schemas/optimizer/real-model-correctness-regression-v1.schema.json").read_text()
        )
        validate_instance(report.to_dict(), schema)

    def test_rejects_nondeterministic_repeated_candidate(self) -> None:
        report = evaluate_real_model_regression(
            _report("a" * 64),
            (_report("b" * 64), _report("b" * 64, tokens=(1, 3))),
            minimum_token_agreement=0.0,
            maximum_expectation_regression=1.0,
            maximum_latency_regression=1.0,
            maximum_rss_regression=1.0,
        )
        self.assertFalse(report.approved)
        self.assertFalse(report.runs[1].deterministic_with_prior_runs)

    def test_rejects_missing_language_and_mixed_candidate_hash(self) -> None:
        baseline = _report("a" * 64)
        missing = replace(baseline, samples=baseline.samples[:2])
        with self.assertRaisesRegex(ValueError, "requires en, ja, and zh-Hans"):
            evaluate_real_model_regression(
                missing,
                (_report("b" * 64), _report("b" * 64)),
                minimum_token_agreement=1,
                maximum_expectation_regression=0,
                maximum_latency_regression=1,
                maximum_rss_regression=1,
            )
        with self.assertRaisesRegex(ValueError, "one model hash"):
            evaluate_real_model_regression(
                baseline,
                (_report("b" * 64), _report("c" * 64)),
                minimum_token_agreement=1,
                maximum_expectation_regression=0,
                maximum_latency_regression=1,
                maximum_rss_regression=1,
            )

    def test_performance_regression_fails_gate(self) -> None:
        report = evaluate_real_model_regression(
            _report("a" * 64),
            (_report("b" * 64, elapsed=100), _report("b" * 64, elapsed=130)),
            minimum_token_agreement=1,
            maximum_expectation_regression=0,
            maximum_latency_regression=0.1,
            maximum_rss_regression=0,
        )
        self.assertFalse(report.approved)
        self.assertAlmostEqual(report.latency_regression, 0.3)

    def test_cli_loads_repeated_reports_and_persists_schema_output(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidates = (root / "candidate-1.json", root / "candidate-2.json")
            baseline.write_text(json.dumps(_report("a" * 64).to_dict()))
            for candidate in candidates:
                candidate.write_text(json.dumps(_report("b" * 64).to_dict()))
            output = root / "regression.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = optimizer_main(
                    [
                        "correctness-regression",
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidates[0]),
                        "--candidate",
                        str(candidates[1]),
                        "--min-token-agreement",
                        "1",
                        "--max-expectation-regression",
                        "0",
                        "--max-latency-regression",
                        "0.1",
                        "--max-rss-regression",
                        "0.1",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["approved"])
            self.assertEqual(
                json.loads(output.read_text())["report_id"],
                json.loads(stdout.getvalue())["report_id"],
            )


if __name__ == "__main__":
    unittest.main()
