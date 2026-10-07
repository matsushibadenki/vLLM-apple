import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm_apple import local_text


class LocalTextTests(unittest.TestCase):
    def test_pinned_artifacts_and_hardware_are_required_before_launch(self):
        digest = hashlib.sha256(b'fixture').hexdigest()
        hardware = Mock(is_apple_silicon=True, soc='Apple M4')
        hardware.memory = SimpleNamespace(total_bytes=32*1024**3, available_bytes=16*1024**3)
        hardware.to_dict.return_value = {'soc': 'Apple M4'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'model'
            model.mkdir()
            for name in local_text.MODEL_HASHES:
                (model/name).write_bytes(b'fixture')
            for name in local_text.SOURCE_HASHES:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            distribution = Mock()
            distribution.locate_file.side_effect = lambda name: root/name
            with patch.object(local_text, 'detect_hardware', return_value=hardware), \
                 patch.object(local_text, 'MODEL_HASHES', {name: digest for name in local_text.MODEL_HASHES}), \
                 patch.object(local_text, 'SOURCE_HASHES', {name: digest for name in local_text.SOURCE_HASHES}), \
                 patch.object(local_text.importlib.metadata, 'version', side_effect=lambda n: {'mlx': '0.32.1', 'mlx-lm': '0.32.0', 'transformers': '5.17.0'}[n]), \
                 patch.object(local_text.importlib.metadata, 'distribution', return_value=distribution):
                plan = local_text.preview_plan(model, 8000)
                self.assertTrue(plan['eligible_for_preview'])
                self.assertFalse(plan['production_qualified'])
                self.assertFalse(plan['gpu_verified'])
                (model/'tokenizer.json').write_bytes(b'changed')
                self.assertIn('model_artifact_mismatch:tokenizer.json', local_text.preview_plan(model, 8000)['rejection_reasons'])
                (model/'tokenizer.json').write_bytes(b'fixture')
                (root/'mlx_lm/server.py').write_bytes(b'changed')
                self.assertIn('dependency_source_mismatch:mlx_lm/server.py', local_text.preview_plan(model, 8000)['rejection_reasons'])
                (model/'extra.safetensors').write_bytes(b'extra')
                self.assertIn('unexpected_weight_files', local_text.preview_plan(model, 8000)['rejection_reasons'])
                hardware.soc = 'Apple M5'
                self.assertIn('hardware_outside_measured_preview', local_text.preview_plan(model, 8000)['rejection_reasons'])
                hardware.memory.available_bytes = 1
                self.assertIn('insufficient_available_memory', local_text.preview_plan(model, 8000)['rejection_reasons'])

    def test_check_never_starts_worker_and_launch_replaces_process(self):
        plan = dict(eligible_for_preview=True, launch_command=['python', '-m', 'worker'])
        with patch.object(local_text, 'preview_plan', return_value=plan), \
             patch.object(local_text.os, 'execve') as execute, patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(local_text.main(['--model', '/unused', '--check']), 0)
            execute.assert_not_called()
            with patch('sys.stderr', new=io.StringIO()):
                local_text.main(['--model', '/unused', '--language', 'ja'])
            self.assertEqual(execute.call_args.args[2]['VLLM_APPLE_P1_PROFILE'], '1')
        with patch.object(local_text, 'preview_plan', return_value=dict(eligible_for_preview=False)), \
             patch.object(local_text.os, 'execve') as execute, patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(local_text.main(['--model', '/unused']), 1)
            execute.assert_not_called()
