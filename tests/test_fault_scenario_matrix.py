import unittest

from vllm_apple.fault_scenario_matrix import (
    PlatformFaultObservation,
    PlatformFaultPoint,
    PlatformFaultScenarioMatrix,
    default_platform_fault_scenarios,
)


def observation(scenario, **changes):
    values = dict(
        recovery=scenario.expected_recovery,
        service_ready=True,
        active_reservations=0,
        temporary_files=0,
        stored_prompts=0,
        stored_outputs=0,
        last_known_good_restored=True,
    )
    values.update(changes)
    return PlatformFaultObservation(**values)


class FaultScenarioMatrixTests(unittest.TestCase):
    def test_default_matrix_covers_all_points_and_passes_cleanup_gate(self):
        scenarios = default_platform_fault_scenarios()
        report = PlatformFaultScenarioMatrix(scenarios).run({
            point: lambda scenario: observation(scenario)
            for point in PlatformFaultPoint
        })
        self.assertTrue(report.passed)
        self.assertEqual({result.point for result in report.results},
                         {point.value for point in PlatformFaultPoint})
        self.assertEqual(len(report.report_id), 64)

    def test_reservation_temporary_and_sensitive_leaks_fail_closed(self):
        scenarios = default_platform_fault_scenarios()
        changes = {
            PlatformFaultPoint.PROFILE_PERSISTENCE: {"last_known_good_restored": False},
            PlatformFaultPoint.SCHEDULER_ADMISSION: {"active_reservations": 1},
            PlatformFaultPoint.WORKER_CRASH: {"temporary_files": 1},
            PlatformFaultPoint.CLIENT_DISCONNECT: {"stored_outputs": 1},
        }
        report = PlatformFaultScenarioMatrix(scenarios).run({
            point: (lambda scenario, point=point: observation(scenario, **changes[point]))
            for point in PlatformFaultPoint
        })
        self.assertFalse(report.passed)
        self.assertEqual(
            {result.reason for result in report.results},
            {"last_known_good_not_restored", "reservation_leak",
             "temporary_file_leak", "sensitive_output_persisted"},
        )

    def test_handler_exception_is_bounded_result(self):
        scenarios = default_platform_fault_scenarios()
        handlers = {
            point: lambda scenario: observation(scenario)
            for point in PlatformFaultPoint
        }
        handlers[PlatformFaultPoint.WORKER_CRASH] = lambda _scenario: (_ for _ in ()).throw(
            RuntimeError("crash")
        )
        report = PlatformFaultScenarioMatrix(scenarios).run(handlers)
        failed = next(result for result in report.results if not result.passed)
        self.assertEqual(failed.reason, "handler_RuntimeError")
        self.assertIsNone(failed.observation)


if __name__ == "__main__":
    unittest.main()
