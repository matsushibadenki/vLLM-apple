import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from vllm_apple.spm_tokenmap_reuse import SPMTokenmapReuse, install_spm_tokenmap_reuse


class Detokenizer:
    def reset(self):
        self.offset = 0
        self.tokens = []
        self._unflushed = b''
        self.text = ''


def original(instance, tokenizer, trim_space=True):
    instance.trim_space = trim_space
    instance._sep = b'\xe2\x96\x81'
    instance.tokenmap = list(tokenizer.table)
    instance.reset()


class SPMTokenmapReuseTests(unittest.TestCase):
    def test_table_shared_but_stream_state_and_trim_are_independent(self):
        constructor = Mock(side_effect=original)
        reuse = SPMTokenmapReuse(constructor)
        tokenizer = SimpleNamespace(table=[b'one', b'\xe2\x96\x81two', b'\xe4', b'\xb8', b'\xad'])
        first, second = Detokenizer(), Detokenizer()
        reuse.initialize(first, tokenizer)
        first.tokens.append(0)
        first.text = 'one'
        first._unflushed = b'\xe4'
        reuse.initialize(second, tokenizer, trim_space=False)
        constructor.assert_called_once()
        self.assertIs(first.tokenmap, second.tokenmap)
        self.assertEqual(second.tokenmap, tuple(tokenizer.table))
        self.assertEqual(second.tokens, [])
        self.assertEqual(second.text, '')
        self.assertEqual(second._unflushed, b'')
        self.assertFalse(second.trim_space)
        self.assertTrue(first.trim_space)
        with self.assertRaises(TypeError):
            second.tokenmap[0] = b'changed'
        self.assertEqual(reuse.snapshot()['tables_built'], 1)
        self.assertEqual(reuse.snapshot()['shared_instances'], 1)

    def test_different_tokenizers_never_share_tables(self):
        reuse = SPMTokenmapReuse(original)
        first, second = Detokenizer(), Detokenizer()
        reuse.initialize(first, SimpleNamespace(table=[b'a']))
        reuse.initialize(second, SimpleNamespace(table=[b'b']))
        self.assertIsNot(first.tokenmap, second.tokenmap)
        self.assertEqual(second.tokenmap, (b'b',))
        self.assertEqual(reuse.snapshot()['tables_built'], 2)

    def test_failed_build_is_not_cached(self):
        constructor = Mock(side_effect=[RuntimeError('failed'), None])
        reuse = SPMTokenmapReuse(constructor)
        tokenizer = SimpleNamespace(table=[b'a'])
        instance = Detokenizer()
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            reuse.initialize(instance, tokenizer)
        constructor.side_effect = original
        reuse.initialize(instance, tokenizer)
        self.assertEqual(reuse.snapshot()['tables_built'], 1)
        self.assertEqual(reuse.snapshot()['instances'], 1)

    def test_unreviewed_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reviewed'):
            install_spm_tokenmap_reuse(SimpleNamespace(__file__=__file__))
