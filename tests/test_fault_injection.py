import unittest

from vllm_apple.fault_injection import (
    DeterministicFaultInjector,
    FaultAction,
    FaultPoint,
    FaultRule,
    InjectedFault,
)


class FaultInjectionTests(unittest.TestCase):
    def test_one_shot_rule_is_deterministic_and_bounded(self):
        injector = DeterministicFaultInjector((
            FaultRule(FaultPoint.BACKEND_EXECUTE, FaultAction.RETRYABLE, 2),
        ))
        injector.hit(FaultPoint.BACKEND_EXECUTE)
        with self.assertRaises(InjectedFault):
            injector.hit(FaultPoint.BACKEND_EXECUTE)
        injector.hit(FaultPoint.BACKEND_EXECUTE)
        snapshot = injector.snapshot()
        self.assertEqual(snapshot["hits"]["backend_execute"], 3)
        self.assertEqual(snapshot["injections"]["backend_execute"], 1)
        self.assertNotIn("request", repr(snapshot))

    def test_repeating_rule_starts_at_configured_hit(self):
        injector = DeterministicFaultInjector((
            FaultRule(FaultPoint.BACKEND_STOP, FaultAction.FATAL, 2, repeat=True),
        ))
        injector.hit(FaultPoint.BACKEND_STOP)
        for _ in range(2):
            with self.assertRaisesRegex(InjectedFault, "injected_backend_stop_fatal"):
                injector.hit(FaultPoint.BACKEND_STOP)

    def test_invalid_or_duplicate_rules_fail_closed(self):
        rule = FaultRule(FaultPoint.BACKEND_EXECUTE, FaultAction.TIMEOUT)
        with self.assertRaises(ValueError):
            DeterministicFaultInjector((rule, rule))
        with self.assertRaises(ValueError):
            FaultRule(FaultPoint.BACKEND_EXECUTE, FaultAction.TIMEOUT, 0)


if __name__ == "__main__":
    unittest.main()
