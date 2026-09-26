import hashlib
import os
import types
import unittest
from unittest.mock import patch

from vllm_apple.mlx_gemma2_compat import (
    grouped_query_mask,
    install_gemma2_batch_mask_fix,
)


class Mask:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)

    def reshape(self, *shape):
        return Mask(shape)


class Gemma2CompatTests(unittest.TestCase):
    def test_only_grouped_batch_shared_head_masks_are_changed(self):
        mask = Mask((3, 1, 4, 7))
        self.assertEqual(grouped_query_mask(mask, 2).shape, (3, 1, 1, 4, 7))
        self.assertIs(grouped_query_mask(mask, 1), mask)
        for unchanged in (None, Mask((4, 7)), Mask((3, 1, 1, 4, 7))):
            self.assertIs(grouped_query_mask(unchanged, 2), unchanged)
        with self.assertRaises(ValueError):
            grouped_query_mask(Mask((3, 2, 4, 7)), 2)

    def test_unreviewed_version_rejected_before_import(self):
        with patch('importlib.metadata.version', return_value='0.33.0'):
            with self.assertRaises(ValueError):
                install_gemma2_batch_mask_fix()

    def test_source_guard_idempotence_and_forwarding(self):
        class Attention:
            repeats = 2

            def __call__(self, x, mask=None, cache=None):
                return x, mask.shape, cache

        gemma = types.SimpleNamespace(Attention=Attention, __file__='/fake/gemma2.py')
        models = types.ModuleType('mlx_lm.models')
        models.gemma2 = gemma
        with patch.dict('sys.modules', {'mlx_lm.models': models}), patch(
            'importlib.metadata.version', return_value='0.32.0'
        ), patch('pathlib.Path.read_bytes', return_value=b'fixture'):
            with self.assertRaises(ValueError):
                install_gemma2_batch_mask_fix()
            with patch('vllm_apple.mlx_gemma2_compat.GEMMA2_SOURCE_SHA256',
                       hashlib.sha256(b'fixture').hexdigest()):
                self.assertTrue(install_gemma2_batch_mask_fix())
                self.assertFalse(install_gemma2_batch_mask_fix())
                self.assertEqual(Attention()('input', Mask((3, 1, 4, 7)), 'cache'),
                                 ('input', (3, 1, 1, 4, 7), 'cache'))


@unittest.skipUnless(os.environ.get('VLLM_APPLE_TEST_GEMMA2_MASK') == '1',
                     'requires explicit MLX GPU environment')
class Gemma2MaskDeviceTests(unittest.TestCase):
    def test_batched_attention_matches_individual_unpatched_attention(self):
        import mlx.core as mx
        from mlx_lm.models.gemma2 import Attention, ModelArgs

        original = Attention.__call__
        try:
            install_gemma2_batch_mask_fix()
            mx.random.seed(42)
            for batch in (2, 3):
                for length in (1, 4):
                    for boolean in (True, False):
                        with self.subTest(batch=batch, length=length, boolean=boolean):
                            args = ModelArgs('gemma2', 16, 1, 32, 4, 4, 1e-6, 32, 2)
                            attention = Attention(args)
                            x = mx.random.normal((batch, length, 16))
                            # Distinct per-row key masks catch silent batch/head alignment errors.
                            valid = mx.arange(length)[None, :] >= (
                                mx.arange(batch)[:, None] % length)
                            valid = mx.broadcast_to(valid[:, None, None, :],
                                                    (batch, 1, length, length))
                            mask = valid if boolean else mx.where(valid, 0., -1e9)
                            actual = attention(x, mask=mask)
                            expected = mx.concatenate([
                                original(attention, x[i:i+1], mask=mask[i:i+1])
                                for i in range(batch)
                            ], axis=0)
                            mx.eval(actual, expected)
                            self.assertTrue(bool(mx.allclose(actual, expected, atol=1e-5,
                                                            rtol=1e-5).item()))
        finally:
            Attention.__call__ = original


if __name__ == '__main__':
    unittest.main()
