import json
import tempfile
import unittest
from pathlib import Path

from scripts.p1_stability_status import inspect_status


class P1StabilityStatusTests(unittest.TestCase):
    def test_dead_runner_and_stale_checkpoint_are_not_healthy(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            state = directory/'state.json'
            state.write_text(json.dumps(dict(status='running', pid=1, stages=[], passed=False)))
            now = state.stat().st_mtime
            dead = inspect_status(directory, now=now, runner_alive=lambda pid: False)
            self.assertEqual(dead['observed_status'], 'interrupted')
            stale = inspect_status(directory, now=now+301, runner_alive=lambda pid: True)
            self.assertEqual(stale['observed_status'], 'stale_checkpoint')
            healthy = inspect_status(directory, now=now+10, runner_alive=lambda pid: True)
            self.assertEqual(healthy['observed_status'], 'running')
            self.assertFalse(healthy['qualification'])
