import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_strata_host_load import analyze


class HostDiagnosisTests(unittest.TestCase):
    def test_overlapping_causes_phase_separation_and_no_requalification(self):
        bad = dict(total_percent=201, maximum_process_percent=76,
                   new_or_exited_processes=1,
                   top_processes=[dict(name="external", cpu_percent=76)])
        good = dict(total_percent=200, maximum_process_percent=75,
                    new_or_exited_processes=0, observer_cpu_percent=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(dict(trials=[dict(
                startup_host_load=dict(observations=[good]),
                request_start_gate=dict(observations=[good]),
                request_host_load=dict(observations=[bad, good]))])))
            original = path.read_bytes()
            report = analyze([path, path])
            self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(report["source_sha256"]), 1)
        self.assertFalse(report["qualification_changed"])
        requests = report["phases"]["requests"]
        self.assertEqual(requests["counts"]["samples"], 2)
        self.assertEqual(requests["counts"]["failed_samples"], 1)
        for key in ("total_percent", "maximum_process_percent", "new_or_exited_processes"):
            self.assertEqual(requests["counts"][key], 1)
        self.assertEqual(requests["over_75_process_samples"], {"external": 1})
        self.assertEqual(requests["counts"]["attributed_samples"], 1)
        self.assertEqual(report["phases"]["startup"]["counts"]["samples"], 1)
        self.assertEqual(report["phases"]["ready_wait"]["counts"]["samples"], 1)
