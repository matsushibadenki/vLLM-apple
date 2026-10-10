import unittest

from experiments.p3_mlx.conditions import require_matching_conditions
from experiments.p3_mlx.config import selected_settings
from tests import test_p3_gpu_connection


class OperatingConditionsTests(unittest.TestCase):
    def test_nominal_matching_conditions_apply(self):
        payload = test_p3_gpu_connection.SelectionConnectionTests().evidence()
        selected_settings(payload, payload['scope']['identity'],
                          payload['scope']['operating_conditions'])

    def test_power_change_unsafe_thermal_and_unknown_are_rejected(self):
        payload = test_p3_gpu_connection.SelectionConnectionTests().evidence()
        measured = payload['scope']['operating_conditions']
        for key, value in (('power_source', 'Battery Power'), ('power_mode', 'low_power'),
                           ('thermal_state', 'fair'), ('thermal_state', 'serious'),
                           ('thermal_state', 'critical'), ('thermal_state', 'unknown'),
                           ('power_source', 'unknown'), ('power_mode', 'unknown')):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                selected_settings(payload, payload['scope']['identity'], dict(measured, **{key: value}))
        with self.assertRaises(ValueError):
            require_matching_conditions(None, measured)

    def test_gpu_core_count_missing_unknown_or_changed_are_rejected(self):
        for cores in (None, 0, True, 513):
            payload = test_p3_gpu_connection.SelectionConnectionTests().evidence()
            payload['scope']['identity']['hardware']['gpu_core_count'] = cores
            with self.assertRaises(ValueError):
                selected_settings(payload, payload['scope']['identity'], payload['scope']['operating_conditions'])
        payload = test_p3_gpu_connection.SelectionConnectionTests().evidence()
        identity = {**payload['scope']['identity'], 'hardware': {'soc': 'test', 'gpu_core_count': 8}}
        with self.assertRaises(ValueError):
            selected_settings(payload, identity, payload['scope']['operating_conditions'])
