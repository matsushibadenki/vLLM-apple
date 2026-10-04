import hashlib
import os
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm_apple.mlx_gemma2_compat import (
    _activate_pending,
    _register_context,
    _register_pending,
    _release_context,
    _RequestQueue,
    cancel_request,
    grouped_query_mask,
    install_gemma2_batch_mask_fix,
    validate_p1_request,
)


class Mask:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)

    def reshape(self, *shape):
        return Mask(shape)


class P1ProfileTests(unittest.TestCase):
    def test_model_and_output_bounds_precede_generation(self):
        args = SimpleNamespace(model=SimpleNamespace(model='gemma', adapter=None, draft=None), max_tokens=512)
        validate_p1_request(args, 'gemma')
        args.model.draft = 'default_model'
        validate_p1_request(args, 'gemma')
        for value in (0, 513, True):
            args.max_tokens = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_p1_request(args, 'gemma')
        args.max_tokens = 16
        args.model.model = 'other'
        with self.assertRaises(ValueError):
            validate_p1_request(args, 'gemma')


class Gemma2CompatTests(unittest.TestCase):
    def tearDown(self):
        from vllm_apple.mlx_gemma2_compat import _ACTIVE, _PENDING
        _ACTIVE.clear()
        _PENDING.clear()

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

    def test_cancel_registry_is_bounded_to_active_valid_ids(self):
        class Context:
            stopped = 0

            def stop(self):
                self.stopped += 1

        context = Context()
        queue = Mock()
        _register_context("request-1", context, queue)
        self.assertTrue(cancel_request("request-1"))
        self.assertTrue(cancel_request("request-1"))
        self.assertEqual(context.stopped, 1)
        queue.put.assert_called_once_with(None)
        self.assertFalse(cancel_request("../request-1"))
        _release_context("request-1", object())
        self.assertTrue(cancel_request("request-1"))
        _release_context("request-1", context)
        self.assertFalse(cancel_request("request-1"))

    def test_duplicate_request_id_stops_only_the_new_context(self):
        class Context:
            stopped = False

            def stop(self):
                self.stopped = True

        first, duplicate = Context(), Context()
        first_queue, duplicate_queue = Mock(), Mock()
        _register_context("same", first, first_queue)
        with self.assertRaises(ValueError):
            _register_context("same", duplicate, duplicate_queue)
        self.assertFalse(first.stopped)
        self.assertTrue(duplicate.stopped)
        first_queue.put.assert_not_called()
        duplicate_queue.put.assert_called_once_with(None)

    def test_queued_cancel_is_stopped_when_context_activates(self):
        class Context:
            stopped = False

            def stop(self):
                self.stopped = True

        queue = _RequestQueue(__import__("queue").Queue)
        _register_pending("queued-1", queue)
        self.assertTrue(cancel_request("queued-1"))
        context = Context()
        self.assertFalse(_activate_pending("queued-1", context, queue))
        self.assertTrue(context.stopped)
        self.assertIsNone(queue.get())
        self.assertFalse(cancel_request("queued-1"))

    def test_duplicate_pending_request_id_is_rejected(self):
        first = _RequestQueue(__import__("queue").Queue)
        duplicate = _RequestQueue(__import__("queue").Queue)
        _register_pending("same", first)
        with self.assertRaises(ValueError):
            _register_pending("same", duplicate)


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
