import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from experiments.p3_mlx.config import digest, evaluate, selected_settings
from experiments.p3_mlx.trace import add_shader_breakdown, summarize_trace, table, union_ns
from vllm_apple.p3_selection import SelectionPolicy


class TraceTests(unittest.TestCase):
    def test_overlap_union_does_not_double_count(self):
        self.assertEqual(union_ns([(1,10),(2,5),(9,15),(20,25)]),19)
        with self.assertRaises(ValueError):
            union_ns([(2,1)])

    def test_target_filter_clock_refs_and_shader_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def xml(name,fields,rows):
                path = root/name
                schema = ''.join(f'<col><mnemonic>{f}</mnemonic></col>' for f in fields)
                path.write_text(f'<trace-query-result><node><schema>{schema}</schema>{rows}</node></trace-query-result>')
                return path
            clock = xml('clock.xml',['mabs-epoch','timebase-info'],
                '<row><mach>24</mach><base><numer>125</numer><denom>3</denom></base></row>')
            fields = ['start','duration','process','channel-name','state','event-depth','start-latency']
            row1 = '<row><start>5</start><duration>100</duration><process id="p"><pid>7</pid></process><channel id="c">Compute</channel><state id="s">Active</state><depth>0</depth><latency>2</latency></row>'
            nested = '<row><start>15</start><duration>15</duration><process ref="p"/><channel ref="c"/><state ref="s"/><depth>1</depth><latency>2</latency></row>'
            other = '<row><start>10</start><duration>100</duration><process><pid>8</pid></process><channel ref="c"/><state ref="s"/><depth>0</depth><latency>2</latency></row>'
            gpu = xml('gpu.xml',fields,row1+nested+other)
            workload = root/'workload.json'
            workload.write_text(json.dumps(dict(samples=[dict(phase='prefill',start_monotonic_ns=1010,end_monotonic_ns=1110,wall_ns=100,cpu_ns=10,graph_encode_ns=5,evaluation_and_wait_ns=95,finite=True)])))
            report = summarize_trace(gpu,clock,workload,7)
            self.assertEqual(report['phases']['prefill']['gpu_active_ns'],95)
            shader = xml('shader.xml',['start','duration','process','shader-type','name'],
                '<row><start>10</start><duration>50</duration><process id="p"><pid>7</pid></process><type>Compute</type><name>qmm_f32</name></row>'
                '<row><start>60</start><duration>20</duration><process ref="p"/><type>Compute</type><name>rms_norm</name></row>')
            add_shader_breakdown(report,shader,clock,workload,7)
            self.assertEqual(report['suggested_tuning_dimension'],'prefill_step_size')
            self.assertAlmostEqual(report['shader_time_breakdown']['prefill']['normalization']['sampled_time_share'],2/7)
            self.assertIsNone(report['bandwidth_or_compute_bound'])
            with self.assertRaises(ValueError):
                summarize_trace(gpu,clock,workload,999)

    def test_entities_and_invalid_references_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.xml'
            path.write_text('<!DOCTYPE x><trace-query-result/>')
            with self.assertRaises(ValueError):
                table(path)
            path.write_text('<trace-query-result><node><schema><col><mnemonic>x</mnemonic></col></schema><row><value ref="missing"/></row></node></trace-query-result>')
            with self.assertRaises(ValueError):
                table(path)


class SelectionConnectionTests(unittest.TestCase):
    def evidence(self):
        identity = {'files':{'model':'abc'},'hardware':{'soc':'test','gpu_core_count':10}}
        gpu = dict(identity=identity,identity_unchanged=True,suggested_tuning_dimension='prefill_step_size')
        scope = dict(identity=identity,gpu_profile_sha256=digest(gpu),operating_conditions={
            'thermal_state':'nominal','power_source':'AC Power','power_mode':'automatic'})
        gpu['operating_conditions'] = scope['operating_conditions']
        scope['gpu_profile_sha256'] = digest(gpu)
        policy = SelectionPolicy(digest(scope),('en','ja','zh'),4096)
        def candidate(name,kind,seconds):
            return dict(candidate_id=name,kind=kind,trials=[dict(acquisition_id=f'{name}-{i}',scope_sha256=policy.scope_sha256,policy_sha256=policy.digest,e2e_seconds=s,ttft_p95_seconds=1,tpot_p95_seconds=.1,peak_memory_bytes=1024,quality=[['en',True],['ja',True],['zh',True]]) for i,s in enumerate(seconds)])
        return dict(policy=asdict(policy),scope=scope,gpu_profile=gpu,baseline=candidate('baseline-512','baseline',[10,10.1,10]),candidates=[candidate('candidate-128','kernel',[8,8.01,8])],independent_acquisition_verified=True)

    def test_recomputed_choice_applies_only_allowlisted_settings(self):
        payload = self.evidence()
        payload['selected_candidate_id'] = 'untrusted-winner'
        decision, settings = selected_settings(payload,payload['scope']['identity'],payload['scope']['operating_conditions'])
        self.assertEqual(decision['selected_candidate_id'],'candidate-128')
        self.assertEqual(settings[settings.index('--prefill-step-size')+1],'128')
        self.assertIn('--disable-prefix-cache',settings)
        self.assertFalse(decision['standard_adoption_eligible'])
        self.assertFalse(decision['automatic_application'])

    def test_failed_quality_or_independence_retains_baseline(self):
        payload = self.evidence()
        payload['candidates'][0]['trials'][0]['quality'][0][1] = False
        self.assertEqual(evaluate(payload)['selected_candidate_id'],'baseline-512')
        payload = self.evidence()
        payload['independent_acquisition_verified'] = False
        self.assertTrue(evaluate(payload)['baseline_retained'])

    def test_stale_identity_and_unsupported_configuration_rejected(self):
        payload = self.evidence()
        with self.assertRaises(ValueError):
            selected_settings(payload,{'other':'identity'})
        payload = copy.deepcopy(payload)
        payload['candidates'][0]['candidate_id'] = '--arbitrary-command'
        with self.assertRaises(ValueError):
            evaluate(payload)

    def test_gpu_evidence_tampering_rejected(self):
        payload = self.evidence()
        payload['gpu_profile']['suggested_tuning_dimension'] = None
        with self.assertRaises(ValueError):
            evaluate(payload)


class NativeHandlerConnectionTests(unittest.TestCase):
    def test_handler_is_wrapped_before_native_builds_its_factory(self):
        from types import SimpleNamespace

        from experiments.p3_mlx.server import install_profiled_http_server
        class Handler:
            def do_GET(self):
                return 'native-response'
        def native_run(host,port,generator,server_class,handler_class):
            # Native _run_http_server converts the class to a factory here.
            def factory():
                return handler_class()
            return factory()
        native = SimpleNamespace(_run_http_server=native_run)
        install_profiled_http_server(native)
        native._run_http_server.__defaults__ = (object,Handler)
        handler = native._run_http_server('127.0.0.1',19149,object())
        handler.path = '/v1/models'
        self.assertEqual(handler.do_GET(),'native-response')
