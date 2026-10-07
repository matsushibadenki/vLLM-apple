import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import run_p1_stability


class P1RunnerTests(unittest.TestCase):
    def test_prefill_candidate_stays_fixed_across_both_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = []
            policies = []
            def spawn(command, **kwargs):
                child = Mock()
                child.wait.return_value = 0
                child.poll.return_value = 0
                if '--output' in command:
                    commands.append(command)
                    policies.append(kwargs['env']['VLLM_APPLE_P1_EFFICIENCY'])
                    Path(command[command.index('--output') + 1]).write_text(json.dumps({'passed': True}))
                return child
            argv = ['runner', '--python', sys.executable, '--model', str(root),
                    '--output-directory', str(root/'result'), '--prefill-step-size', '256', '--efficiency', 'compact']
            with patch.object(sys, 'argv', argv), patch.object(run_p1_stability.subprocess, 'Popen',
                                                             side_effect=spawn), \
                 patch.object(run_p1_stability.signal, 'signal'):
                self.assertEqual(run_p1_stability.main(), 0)
            self.assertEqual(len(commands), 2)
            self.assertEqual(policies, ['compact', 'compact'])
            self.assertEqual([c[c.index('--prefill-step-size')+1] for c in commands], ['256', '256'])
            self.assertIn('--require-30-minute-window', commands[0])
            self.assertIn('--require-8-hour-window', commands[1])
            state = json.loads((root/'result/state.json').read_text())
            self.assertTrue(state['passed'])
            self.assertFalse(state['automatic_promotion'])
