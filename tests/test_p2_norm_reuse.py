import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.p2_mlx import precision


class Weight:
    def __init__(self, value):
        self.value = value
        self.additions = 0

    def __radd__(self, value):
        self.additions += 1
        return self.value+value


class P2NormReuseTests(unittest.TestCase):
    def test_layers_keep_separate_weights_and_only_build_once(self):
        class RMSNorm(dict):
            # MLX Module stores attributes in its mapping, not __dict__.
            def __setattr__(self, name, value):
                self[name] = value

            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name) from None

            def __init__(self, weight):
                self.weight = weight
                self.eps = 1e-5

            def __call__(self, value):
                return value*(1+self.weight)

        module = SimpleNamespace(RMSNorm=RMSNorm, __file__=__file__)
        fast = Mock(side_effect=lambda value, weight, eps: value*weight)
        mx = SimpleNamespace(fast=SimpleNamespace(rms_norm=fast))
        with open(__file__, 'rb') as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        with patch.object(precision, 'GEMMA2_SOURCE_SHA256', digest):
            metrics = precision.install_norm_weight_reuse(module, mx)
            self.assertIs(precision.install_norm_weight_reuse(module, mx), metrics)
        a, b = Weight(2), Weight(3)
        first, second = RMSNorm(a), RMSNorm(b)
        self.assertEqual(first(5), 15)
        self.assertEqual(first(7), 21)
        self.assertEqual(second(5), 20)
        self.assertEqual((a.additions, b.additions), (1, 1))
        self.assertEqual(metrics['weights_built'], 2)
        self.assertEqual(fast.call_count, 3)

    def test_unknown_source_is_rejected_before_patch(self):
        with self.assertRaisesRegex(ValueError, 'reviewed'):
            precision.install_norm_weight_reuse(SimpleNamespace(__file__=__file__), None)
