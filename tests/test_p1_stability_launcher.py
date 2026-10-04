import io
import json
import plistlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.launch_p1_stability import main


class P1StabilityLauncherTests(unittest.TestCase):
    def test_failed_bootstrap_retains_receipt_and_job_is_not_kept_alive(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            receipt = base/'launch.json'
            argv = ['launch', '--python', sys.executable, '--model', str(base),
                    '--output-directory', str(base/'output'), '--receipt', str(receipt)]
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()), patch(
                'scripts.launch_p1_stability.subprocess.run',
                return_value=Mock(returncode=5, stderr='bootstrap failed')) as run:
                self.assertEqual(main(), 5)
            self.assertEqual(run.call_args.args[0][1], 'bootstrap')
            job = plistlib.loads(receipt.with_suffix('.plist').read_bytes())
            self.assertFalse(job['KeepAlive'])
            self.assertTrue(job['RunAtLoad'])
            record = json.loads(receipt.read_text())
            self.assertFalse(record['submitted'])
            self.assertFalse(record['qualification'])
