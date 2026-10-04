import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.evidence_index import build_evidence_index


class EvidenceIndexTests(unittest.TestCase):
    def test_legacy_pass_does_not_transfer_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            raw = json.dumps(dict(passed=True, qualification_scope='legacy model only',
                                  phase_profile={'sample_count': 3})).encode()
            path.write_bytes(raw)
            index = build_evidence_index([path])
        self.assertFalse(index['qualification'])
        self.assertFalse(index['transfers_qualification'])
        self.assertTrue(index['evidence'][0]['recorded_passed'])
        self.assertEqual(index['evidence'][0]['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertIsNone(index['evidence'][0]['availability']['backend_prefill'])
        with self.assertRaises(ValueError):
            build_evidence_index([])
