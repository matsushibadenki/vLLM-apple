"""Synthetic contract tests; none of these fixtures certifies a real backend."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.schema_validator import validate_instance
from tests.test_model_recommendation import hardware, model
from vllm_apple.architecture_evidence import (
    MAX_AGE_SECONDS,
    bind_evidence,
    capture_identity,
    validate_evidence,
    verify_startup_evidence,
)
from vllm_apple.model import inspect_model
from vllm_apple.model_recommendation import build_model_recommendation
from vllm_apple.qualification import QualificationConfig, save_qualification_report


def identity():
    return {**dict.fromkeys(("config_sha256", "model_root_sha256", "backend_sha256",
                            "environment_sha256", "hardware_sha256", "runtime_sha256"), "a" * 64),
            "backend": "mlx_lm"}


def body():
    return {
        "passed": True, "shutdown_clean": True, "backend": "mlx_lm",
        "requested_modes": ["text"],
        "promotion_probe": {"passed": True, "checks": dict.fromkeys(
            ("greedy_repeat_equal", "sampled_repeat_equal", "sampled_stream_equal",
             "stream_completed"), True)},
        "soak": {"passed": True, "elapsed_seconds": 1800, "process_alive": True,
                 "stability_window_met": True, "failures": 0, "successes": 20, "requests": 20,
                 "rss": {"peak_growth_bytes": 1024, "limit_bytes": 2048}},
        "context_reevaluation": {
            "passed": True, "effective_context_tokens": 1024, "enabled": False,
            "status": "unavailable", "configured_context_tokens": 1024,
            "capacity_context_tokens": None, "kv_capacity_bytes": None,
            "kv_bytes_per_token": None, "weights_bytes": None, "source": None,
            "reevaluations": 0,
        },
        "model_memory_fit": {"fits": True, "context_tokens": 1024},
        "qualification_limits": {"context_tokens": 1024, "concurrency": 1},
        "quality_smoke": {"passed": True, "checks": dict.fromkeys(
            ("english", "japanese", "simplified_chinese"), True)},
        "phase_profile": {"sample_count": 3},
    }


def bound(now=None):
    return bind_evidence(body(), identity(), context_tokens=1024, concurrency=1, now=now)


class ArchitectureEvidenceTests(unittest.TestCase):
    def test_bound_report_and_limits(self):
        report = bound(1000)
        validate_evidence(report, identity(), context_tokens=512, concurrency=1, now=1001)
        schema = json.loads(Path("schemas/runtime/qualification-report-v1.schema.json").read_text())
        validate_instance(report["architecture_evidence"], schema["properties"]["architecture_evidence"])
        for context, concurrency in ((1025, 1), (512, 2), (True, 1), (512, 0)):
            with self.subTest(context=context, concurrency=concurrency), self.assertRaises(ValueError):
                validate_evidence(report, identity(), context_tokens=context,
                                  concurrency=concurrency, now=1001)

    def test_missing_old_future_and_modified_evidence_rejected(self):
        for now in (999, 1001 + MAX_AGE_SECONDS, float("nan")):
            with self.subTest(now=now), self.assertRaises(ValueError):
                validate_evidence(bound(1000), identity(), context_tokens=512, concurrency=1, now=now)
        for mutate in (
            lambda r: r.pop("architecture_evidence"),
            lambda r: r["soak"].update(successes=21),
            lambda r: r["architecture_evidence"].update(scope="all_architectures"),
            lambda r: r["architecture_evidence"]["identity"].update(backend_sha256="b" * 64),
        ):
            report = bound()
            mutate(report)
            with self.assertRaises(ValueError):
                validate_evidence(report, identity(), context_tokens=512, concurrency=1)

    def test_failed_or_incomplete_results_cannot_be_bound(self):
        for section, changes in (
            ("soak", {"elapsed_seconds": 1799}), ("soak", {"failures": True}),
            ("soak", {"rss": {"peak_growth_bytes": 3000, "limit_bytes": 2048}}),
            ("phase_profile", {"sample_count": 2}),
            ("promotion_probe", {"checks": {}}),
            ("quality_smoke", {"checks": {"english": True}}),
            ("context_reevaluation", {"effective_context_tokens": 512}),
            ("qualification_limits", {"concurrency": 2}),
        ):
            report = body()
            report[section].update(changes)
            with self.subTest(section=section, changes=changes), self.assertRaises(ValueError):
                bind_evidence(report, identity(), context_tokens=1024, concurrency=1)

    def test_private_file_gate_and_early_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            save_qualification_report(bound(), path)
            with patch("vllm_apple.architecture_evidence.capture_identity", return_value=identity()) as capture:
                verify_startup_evidence(path, None, Path("/unused"), "mlx_lm", hardware(),
                                        context_tokens=512, concurrency=1)
                capture.assert_called_once()
                capture.reset_mock()
                path.chmod(0o644)
                with self.assertRaisesRegex(ValueError, "owner-only"):
                    verify_startup_evidence(path, None, Path("/unused"), "mlx_lm", hardware(),
                                            context_tokens=512, concurrency=1)
                path.chmod(0o600)
                path.write_text(json.dumps(body()))
                with self.assertRaises(ValueError):
                    verify_startup_evidence(path, None, Path("/unused"), "mlx_lm", hardware(),
                                            context_tokens=512, concurrency=1)
                capture.assert_not_called()

    def test_identity_changes_when_weights_change(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "vllm_apple.architecture_evidence.fingerprint_backend", return_value="a" * 64
        ):
            root = model(directory)
            first = capture_identity(inspect_model(root), Path("/unused"), "mlx_lm", hardware())
            (root / "weights.safetensors").write_bytes(b"changed")
            second = capture_identity(inspect_model(root), Path("/unused"), "mlx_lm", hardware())
            self.assertNotEqual(first["model_root_sha256"], second["model_root_sha256"])
            self.assertEqual(first["config_sha256"], second["config_sha256"])

    def test_recommendation_promotes_only_after_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = model(directory)
            path = Path(directory) / "evidence.json"
            save_qualification_report(bound(), path)
            with patch("vllm_apple.architecture_evidence.capture_identity", return_value=identity()):
                report = build_model_recommendation(
                    inspect_model(root), hardware(), backend="mlx_lm", architecture_evidence=path,
                    backend_executable=Path("/unused"), context_tokens=512,
                ).to_dict()
                self.assertTrue(report["runnable"])
                self.assertEqual(report["qualification"], "text_smoke_30min")
                validate_instance(report, json.loads(Path(
                    "schemas/runtime/model-recommendation-v2.schema.json").read_text()))
                stale = bound(time.time() - MAX_AGE_SECONDS - 1)
                save_qualification_report(stale, path)
                with self.assertRaisesRegex(ValueError, "expired"):
                    build_model_recommendation(
                        inspect_model(root), hardware(), backend="mlx_lm", architecture_evidence=path,
                        backend_executable=Path("/unused"), context_tokens=512,
                    )

    def test_opt_in_qualification_requires_full_protocol(self):
        base = dict(model="unused", executable=Path("/unused"), max_model_len=1024,
                    phase_samples=3, quality_smoke=True, bind_architecture_evidence=True)
        QualificationConfig(**base)
        for changes in ({"require_30_minute_window": False}, {"quality_smoke": False},
                        {"phase_samples": 2}, {"max_model_len": None},
                        {"allow_context_reduction": True}, {"requested_modes": ("vision",)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                QualificationConfig(**{**base, **changes})


class EvidenceIntegrationTests(unittest.TestCase):
    def test_producer_binds_after_clean_shutdown_and_rejects_identity_change(self):
        from contextlib import ExitStack

        from tests.test_qualification import FakeBackend
        from vllm_apple.qualification import qualify_model

        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = model(directory)
                for target, result in (
                    ("detect_hardware", hardware()),
                    ("evaluate_qualification_context", body()["context_reevaluation"]),
                    ("run_serving_promotion_probe", body()["promotion_probe"]),
                    ("run_phase_probe", body()["phase_profile"]),
                    ("run_serving_quality_smoke", body()["quality_smoke"]),
                    ("run_soak", body()["soak"]),
                    ("build_v2_hardware_fingerprint", "fixture"),
                ):
                    stack.enter_context(patch("vllm_apple.qualification." + target, return_value=result))
                later = {**identity(), "backend_sha256": "b" * 64} if changed else identity()
                stack.enter_context(patch("vllm_apple.qualification.capture_identity",
                                          side_effect=[identity(), later]))
                processes = []

                def factory(config):
                    process = FakeBackend(config)
                    processes.append(process)
                    return process

                config = QualificationConfig(
                    model=str(root), executable=Path("/unused"), backend_kind="mlx_lm",
                    max_model_len=1024, phase_samples=3, quality_smoke=True,
                    bind_architecture_evidence=True,
                )
                if changed:
                    with self.assertRaisesRegex(ValueError, "identity changed"):
                        qualify_model(config, process_factory=factory)
                else:
                    report = qualify_model(config, process_factory=factory)
                    validate_evidence(report, identity(), context_tokens=1024, concurrency=1)
                    schema = json.loads(Path("schemas/runtime/qualification-report-v1.schema.json").read_text())
                    validate_instance(report, schema)
                self.assertTrue(processes[0].stopped)

    def test_daemon_checks_evidence_before_constructing_backend_even_when_version_check_skipped(self):
        from types import SimpleNamespace

        from vllm_apple.daemon import serve

        for valid in (False, True):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as directory:
                root = model(directory)
                with patch("vllm_apple.daemon.detect_hardware", return_value=hardware()), patch(
                    "vllm_apple.daemon.inspect_mlx_lm_backend",
                    return_value=SimpleNamespace(compatible=False, architecture_features=()),
                ), patch("vllm_apple.daemon.verify_startup_evidence",
                         side_effect=None if valid else ValueError("invalid evidence")) as verify, patch(
                    "vllm_apple.daemon.BackendProcess", side_effect=RuntimeError("construction reached")
                ) as constructor:
                    with self.assertRaisesRegex((ValueError, RuntimeError),
                                                "construction reached" if valid else "invalid evidence"):
                        serve(model=str(root), backend_kind="mlx_lm", backend_executable=sys.executable,
                              max_model_len=512, max_concurrent_requests=1,
                              architecture_evidence=Path(directory) / "evidence.json",
                              require_compatible_backend=False, enable_metal_tuning=False)
                    verify.assert_called_once()
                    self.assertEqual(constructor.call_count, int(valid))

    def test_daemon_requires_explicit_model_and_context(self):
        from vllm_apple.daemon import serve

        for options in ({}, {"model": "unused"}):
            with self.assertRaisesRegex(ValueError, "explicit max_model_len"):
                serve(architecture_evidence=Path("/unused"), **options)

    def test_version_exception_requires_valid_evidence_and_only_mlx_matrix_issue(self):
        from vllm_apple.compat import MLXBackendCompatibility
        from vllm_apple.daemon import serve

        matrix = "mlx_lm_version_outside_verified_matrix"
        for evidence, issues, invalid, allowed in (
            (False, (matrix,), False, False),
            (True, (matrix,), False, True),
            (True, (matrix,), True, False),
            (True, (matrix, "mlx_lm_version_unavailable"), False, False),
            (True, (), False, False),
        ):
            with self.subTest(evidence=evidence, issues=issues, invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = model(directory)
                with patch("vllm_apple.daemon.detect_hardware", return_value=hardware()), patch(
                    "vllm_apple.daemon.inspect_mlx_lm_backend",
                    return_value=MLXBackendCompatibility("/fixture", "0.32.0", False, issues),
                ), patch("vllm_apple.daemon.verify_startup_evidence",
                         side_effect=ValueError("identity mismatch") if invalid else None) as verify, patch(
                    "vllm_apple.daemon.BackendProcess", side_effect=RuntimeError("construction reached")
                ) as constructor:
                    expected = "construction reached" if allowed else (
                        "identity mismatch" if invalid else "incompatible mlx_lm"
                    )
                    with self.assertRaisesRegex((ValueError, RuntimeError), expected):
                        serve(model=str(root), backend_kind="mlx_lm", backend_executable=sys.executable,
                              max_model_len=512, max_concurrent_requests=1,
                              architecture_evidence=Path(directory) / "evidence.json" if evidence else None,
                              enable_metal_tuning=False)
                    self.assertEqual(constructor.call_count, int(allowed))
                    self.assertEqual(verify.call_count, int(evidence))

    def test_evidence_does_not_override_model_capability_failure(self):
        from vllm_apple.compat import MLXBackendCompatibility
        from vllm_apple.daemon import serve
        from vllm_apple.model import ModelCapabilityError

        with tempfile.TemporaryDirectory() as directory:
            root = model(directory)
            with patch("vllm_apple.daemon.detect_hardware", return_value=hardware()), patch(
                "vllm_apple.daemon.inspect_mlx_lm_backend",
                return_value=MLXBackendCompatibility("/fixture", "0.32.0", False,
                                                     ("mlx_lm_version_outside_verified_matrix",)),
            ), patch("vllm_apple.daemon.verify_startup_evidence"), patch(
                "vllm_apple.daemon.ensure_model_backend_compatible",
                side_effect=ModelCapabilityError("missing capability"),
            ), patch("vllm_apple.daemon.BackendProcess") as constructor:
                with self.assertRaisesRegex(ModelCapabilityError, "missing capability"):
                    serve(model=str(root), backend_kind="mlx_lm", backend_executable=sys.executable,
                          max_model_len=512, max_concurrent_requests=1,
                          architecture_evidence=Path(directory) / "evidence.json",
                          enable_metal_tuning=False)
                constructor.assert_not_called()

    def test_mlx_evidence_exception_does_not_apply_to_vllm(self):
        from types import SimpleNamespace

        from vllm_apple.daemon import serve

        with tempfile.TemporaryDirectory() as directory:
            root = model(directory)
            with patch("vllm_apple.daemon.detect_hardware", return_value=hardware()), patch(
                "vllm_apple.daemon.inspect_backend",
                return_value=SimpleNamespace(compatible=False, architecture_features=(),
                                             issues=("mlx_lm_version_outside_verified_matrix",)),
            ), patch("vllm_apple.daemon.verify_startup_evidence"), patch(
                "vllm_apple.daemon.BackendProcess"
            ) as constructor:
                with self.assertRaisesRegex(RuntimeError, "incompatible vllm_metal"):
                    serve(model=str(root), backend_kind="vllm_metal", backend_executable=sys.executable,
                          max_model_len=512, max_concurrent_requests=1,
                          architecture_evidence=Path(directory) / "evidence.json",
                          enable_metal_tuning=False)
                constructor.assert_not_called()
