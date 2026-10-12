import sys
import unittest
from unittest.mock import patch

from scripts.compare_strata_experts import phased_trial


class Monitor:
    def __init__(self, *_args, **_kwargs):
        self.observations = [dict(total_percent=0, maximum_process_percent=0,
                                  new_or_exited_processes=0)]
        self.error = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


class HandshakeTests(unittest.TestCase):
    @patch("scripts.strata_measurement.HostLoadMonitor", Monitor)
    @patch("scripts.strata_measurement.wait_for_quiet", return_value=dict(passed=True))
    def test_ready_go_done_ack_separates_windows(self, _gate):
        code = ('import sys,json; print("READY",flush=True); '
                'assert input()=="GO"; print("DONE",flush=True); '
                'assert input()=="ACK"; print(json.dumps({"parity_passed":True}))')
        trial, gate = phased_trial([sys.executable, "-c", code])
        self.assertTrue(gate["passed"])
        self.assertTrue(trial["startup_host_load"]["passed"])
        self.assertTrue(trial["request_host_load"]["passed"])

    @patch("scripts.strata_measurement.HostLoadMonitor", Monitor)
    @patch("scripts.strata_measurement.wait_for_quiet", return_value=dict(passed=False))
    def test_failed_ready_gate_does_not_start_requests(self, _gate):
        code = 'print("READY",flush=True); input(); raise RuntimeError("must not run")'
        trial, gate = phased_trial([sys.executable, "-c", code])
        self.assertIsNone(trial)
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
