import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from vllm_apple.p3_selection import Candidate, SelectionPolicy, Trial, select_candidate


class P3SelectionTests(unittest.TestCase):
    def setUp(self):
        self.policy = SelectionPolicy("a" * 64, ("en", "ja", "zh", "coding", "long", "tool"),
                                      1024)
        self.baseline = self.candidate("base", "baseline", 10)

    def candidate(self, name, kind, time):
        return Candidate(name, kind, tuple(
            Trial(f"{name}-fresh-process-{i}", self.policy.scope_sha256, self.policy.digest,
                  time, 1, 1, 100, tuple((s, True) for s in self.policy.quality_slices))
            for i in range(3)))

    def select(self, candidate, **kwargs):
        return select_candidate(self.policy, self.baseline, (candidate,),
                                independent_acquisition_verified=True, **kwargs)

    def test_all_three_kinds_require_e2e_and_prerequisites(self):
        for kind in ("kernel", "quantization", "speculative"):
            candidate = self.candidate(kind, kind, 9.5)
            report = self.select(candidate)
            self.assertEqual(report["selected_candidate_id"], kind)
            self.assertFalse(report["standard_adoption_eligible"])
            self.assertFalse(report["automatic_application"])
            self.assertTrue(self.select(candidate, prerequisites_verified=True)
                            ["standard_adoption_eligible"])

    def test_failure_never_hidden_by_fast_median(self):
        candidate = self.candidate("fast", "quantization", 8)
        for bad in (
            replace(candidate.trials[0], peak_memory_bytes=1025),
            replace(candidate.trials[0], quality=(("en", False),)),
            replace(candidate.trials[0], policy_sha256="b" * 64),
            replace(candidate.trials[0], ttft_p95_seconds=1.051),
            replace(candidate.trials[0], tpot_p95_seconds=1.051),
            replace(candidate.trials[0], e2e_seconds=10),
            replace(candidate.trials[0], acquisition_id=self.baseline.trials[0].acquisition_id),
        ):
            with self.subTest(bad=bad):
                report = self.select(replace(candidate, trials=(bad, *candidate.trials[1:])))
                self.assertTrue(report["baseline_retained"])
                self.assertFalse(report["standard_adoption_eligible"])

    def test_insufficient_and_unverified_evidence(self):
        candidate = self.candidate("fast", "kernel", 8)
        self.assertTrue(self.select(replace(candidate, trials=candidate.trials[:2]))
                        ["baseline_retained"])
        self.assertTrue(select_candidate(self.policy, self.baseline, (candidate,))
                        ["baseline_retained"])

    def test_slow_candidate_and_failed_baseline(self):
        self.assertTrue(self.select(self.candidate("slow", "speculative", 115.9))
                        ["baseline_retained"])
        self.baseline = replace(self.baseline, trials=self.baseline.trials[:2])
        self.assertTrue(self.select(self.candidate("fast", "kernel", 8))["baseline_retained"])

    def test_report_binds_every_trial(self):
        candidate = self.candidate("fast", "kernel", 8)
        first = self.select(candidate)
        changed = replace(candidate, trials=(replace(candidate.trials[0], peak_memory_bytes=101),
                                             *candidate.trials[1:]))
        self.assertNotEqual(first["report_id"], self.select(changed)["report_id"])
        self.assertEqual(first, self.select(candidate))

    def test_invalid_policy_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.policy, minimum_improvement=0.01)
        with self.assertRaises(ValueError):
            replace(self.baseline.trials[0], e2e_seconds=float("nan"))

    def test_cli_preserves_safe_default_and_rejects_invalid_boolean(self):
        payload = dict(policy=asdict(self.policy), baseline=asdict(self.baseline),
                       candidates=[asdict(self.candidate("fast", "kernel", 8))])
        script = Path(__file__).resolve().parents[1] / "scripts/select_p3_candidate.py"
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence.json"
            evidence.write_text(json.dumps(payload))
            result = subprocess.run([sys.executable, str(script), str(evidence)],
                                    capture_output=True, text=True, check=True)
            self.assertTrue(json.loads(result.stdout)["baseline_retained"])
            payload["independent_acquisition_verified"] = "false"
            evidence.write_text(json.dumps(payload))
            result = subprocess.run([sys.executable, str(script), str(evidence)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
