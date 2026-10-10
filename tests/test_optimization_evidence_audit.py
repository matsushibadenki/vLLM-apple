import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_optimization_evidence import audit, check_files


class EvidenceAuditTests(unittest.TestCase):
    def test_content_change_and_missing_file_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'proof'
            path.write_bytes(b'original')
            expected = hashlib.sha256(b'original').hexdigest()
            self.assertTrue(check_files(root, {'proof': expected})[0]['matches'])
            path.write_bytes(b'changed')
            self.assertFalse(check_files(root, {'proof': expected})[0]['matches'])
            path.unlink()
            self.assertFalse(check_files(root, {'proof': expected})[0]['matches'])

    def test_paths_outside_repository_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                check_files(Path(directory), {'../proof': 'invalid'})

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'escape').symlink_to('/etc/hosts')
            with self.assertRaises(ValueError):
                check_files(root, {'escape': 'invalid'})

    def test_changed_native_identity_prevents_audit_success(self):
        root = Path(__file__).resolve().parents[1]
        with patch('scripts.audit_optimization_evidence.check_files', return_value=[]), \
                patch('experiments.p3_mlx.config.local_identity', return_value={'changed': True}):
            result = audit(root)
        self.assertFalse(result['passed'])
        self.assertFalse(result['current_hardware_runtime_model_identity_matches'])
        self.assertFalse(result['qualification'])
        self.assertFalse(result['long_run_qualification'])
