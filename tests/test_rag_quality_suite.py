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

    def test_comparison_rejects_higher_total_with_abstention_regression(self):
        from scripts.rag_quality_suite import compare_reports
        suite = score_results(self.rows())
        baseline = dict(quality_suite=copy.deepcopy(suite), error=None, backend_returncode=0)
        candidate = copy.deepcopy(baseline)
        baseline['quality_suite']['cases'][0]['passed'] = False
        baseline['quality_suite']['cases'][2]['passed'] = False
        candidate['quality_suite']['cases'][1]['passed'] = False
        comparison = compare_reports(baseline, candidate)
        self.assertGreater(comparison['candidate_passes'], comparison['baseline_passes'])
        self.assertFalse(comparison['task_improvement_accepted'])
        self.assertEqual(comparison['lost_cases'], ['en-insufficient'])
        candidate['quality_suite']['cases'][1]['passed'] = True
        self.assertTrue(compare_reports(baseline, candidate)['task_improvement_accepted'])
        candidate['quality_suite']['cases'].pop()
        with self.assertRaises(ValueError):
            compare_reports(baseline, candidate)

    def test_adoption_requires_unchanged_model_known_versions_and_clean_shutdown(self):
        from scripts.rag_quality_suite import compare_reports
        digest = 'a' * 64
        snapshot = dict(model_complete=True,
                        model_files_sha256={name: digest for name in
                                            ('config.json', 'tokenizer_config.json',
                                             'tokenizer.json', 'model.safetensors')},
                        source_sha256={name: digest for name in
                                       ('vllm_apple/rag.py', 'vllm_apple/mlx_server.py',
                                        'scripts/probe_rag_http.py', 'scripts/rag_quality_suite.py')},
                        packages=dict(mlx='0.32.1', **{'mlx-lm': '0.32.0', 'transformers': '5.17.0'}))
        baseline = dict(quality_suite=score_results(self.rows()), error=None, backend_returncode=0,
                        runtime_identity_verified=True, identity_before=copy.deepcopy(snapshot),
                        identity_after=copy.deepcopy(snapshot), shutdown=dict(graceful=True))
        baseline['quality_suite']['cases'][0]['passed'] = False
        candidate = copy.deepcopy(baseline)
        candidate['quality_suite']['cases'][0]['passed'] = True
        self.assertTrue(compare_reports(baseline, candidate)['adoption_evidence_accepted'])
        for mutation in ('model', 'versions', 'during_run', 'shutdown', 'unknown', 'evaluator'):
            modified = copy.deepcopy(candidate)
            if mutation == 'model':
                for key in ('identity_before', 'identity_after'):
                    modified[key]['model_files_sha256']['model.safetensors'] = 'b' * 64
            elif mutation == 'versions':
                for key in ('identity_before', 'identity_after'):
                    modified[key]['packages']['mlx'] = 'different'
            elif mutation == 'during_run':
                modified['identity_after']['model_files_sha256']['tokenizer.json'] = 'b' * 64
            elif mutation == 'shutdown':
                modified['shutdown'] = None
            elif mutation == 'unknown':
                for key in ('identity_before', 'identity_after'):
                    modified[key]['packages']['mlx'] = None
            else:
                for key in ('identity_before', 'identity_after'):
                    modified[key]['source_sha256']['scripts/rag_quality_suite.py'] = 'b' * 64
            result = compare_reports(baseline, modified)
            self.assertTrue(result['task_improvement_accepted'])
            self.assertFalse(result['adoption_evidence_accepted'], mutation)
            self.assertTrue(result['identity_rejections'])
            self.assertFalse(result['automatic_runtime_adoption'])
