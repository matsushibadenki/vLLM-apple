import copy
import unittest

from scripts.rag_quality_suite import build_cases, score_results


class RagQualitySuiteTests(unittest.TestCase):
    def rows(self):
        return [dict(id=c['id'], result=dict(
            answer=c['expected_answer'] if c['expected_reference'] is None else '12347 [S1]',
            status='abstained' if c['expected_reference'] is None else 'references_valid',
            token_budget_verified=True, generation_performed=True,
            sources=[dict(reference='S1', id='manual')])) for c in build_cases()]

    def test_complete_contract_and_language_coverage(self):
        report = score_results(self.rows())
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['cases']), 12)
        self.assertEqual({c['language'] for c in report['cases']}, {'en', 'ja', 'zh'})
        self.assertFalse(report['general_grounding_qualified'])
        self.assertFalse(report['runtime_identity_verified'])

    def test_missing_duplicate_unknown_and_empty(self):
        self.assertFalse(score_results([])['passed'])
        rows = self.rows()
        self.assertFalse(score_results(rows[:-1])['passed'])
        with self.assertRaises(ValueError):
            score_results(rows + rows[:1])
        with self.assertRaises(ValueError):
            score_results([dict(id='unknown', result={})])

    def test_syntax_does_not_hide_wrong_answer_injection_or_dropped_support(self):
        for mutation in (
            dict(answer='54321 [S1]'), dict(answer='12347 [S2]'),
            dict(answer='INJECTED_98765 [S1]'), dict(sources=[]),
            dict(token_budget_verified=False), dict(generation_performed=False), dict(status='incomplete'),
            dict(answer='12347 [S1] invented extra facts'),
        ):
            rows = copy.deepcopy(self.rows())
            rows[0]['result'].update(mutation)
            self.assertFalse(score_results(rows)['passed'], mutation)

    def test_nonempty_insufficient_source_requires_exact_abstention(self):
        rows = self.rows()
        rows[1]['result'].update(answer='12347 [S1]', status='references_valid')
        self.assertFalse(score_results(rows)['passed'])
        cases = build_cases()
        self.assertTrue(cases[1]['payload']['documents'])
        self.assertGreater(len(cases[3]['payload']['documents'][0]['text']), 2000)
