import io
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tests.schema_validator import SchemaValidationError, validate_instance
from vllm_apple.model import InspectedModel
from vllm_apple.optimizer import (
    ArtifactManifest,
    ArtifactTransaction,
    ArtifactValidationError,
    AdapterRegistry,
    CalibrationManifest,
    CancellationToken,
    CheckpointError,
    CheckpointLeaseError,
    CheckpointManifest,
    CheckpointStage,
    CheckpointStore,
    GenerationEvaluationReport,
    GenerationSampleResult,
    IsolatedConversionWorker,
    MLXOptimizationAdapter,
    MLXExportReport,
    OptimizationObjective,
    OptimizationPerformanceProfile,
    PerplexityEvaluationReport,
    PerplexitySlice,
    OptimizerErrorCode,
    OptimizationPathError,
    OptimizerFailure,
    OptimizerEventBus,
    OptimizerState,
    QualityBudget,
    compare_perplexity_reports,
    compare_generation_reports,
    Recoverability,
    ResourceBudget,
    ResumeAction,
    build_dry_run_plan,
    decide_resume,
    execution_fingerprint,
    generation_token_fingerprint,
    profile_optimizer_io,
    persist_artifact_manifest,
    persist_evaluation_report,
    validate_immutable_output_path,
)
from vllm_apple.optimizer.cli import main as optimizer_main
from vllm_apple.optimizer.mlx_generate_evaluate import (
    _expectation_score,
    _formatted_prompt,
    _parse_generation_record,
    _validated_filters,
)
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo, ModelMemorySpec


def hardware() -> HardwareInfo:
    return HardwareInfo(
        platform="Darwin",
        architecture="arm64",
        soc="Test M4",
        physical_cpu_count=10,
        logical_cpu_count=10,
        gpu_core_count=10,
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=20 * GIB),
        is_apple_silicon=True,
        os_version="test",
    )


def schema(name: str) -> dict:
    path = Path("schemas/optimizer") / name
    return json.loads(path.read_text(encoding="utf-8"))


def compatible_mlx_adapter() -> MLXOptimizationAdapter:
    versions = {"mlx": "0.27.1", "mlx-lm": "0.26.2"}
    return MLXOptimizationAdapter(
        lambda package: versions[package],
        lambda: ("Darwin", "arm64"),
        sys.executable,
    )


class OptimizerFoundationTests(unittest.TestCase):
    def test_dry_run_is_deterministic_bounded_and_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            model = InspectedModel(
                model_id="test/model",
                path=source,
                config={"torch_dtype": "float16", "hidden_size": 1024},
                memory_spec=ModelMemorySpec("test/model", 2 * GIB, 131_072),
                kv_dtype_bytes=2,
            )
            calibration = CalibrationManifest(
                "local-v1", "a" * 64, 100, ("en", "ja", "zh-Hans"), ("code",)
            )
            plan = build_dry_run_plan(
                model,
                hardware(),
                Path(directory) / "artifact",
                OptimizationObjective.MEMORY,
                ResourceBudget(4 * GIB, 4 * GIB),
                QualityBudget({"code": 0.02}),
                calibration,
                "test-license",
                plan_id="plan-1",
                created_at="2026-08-22T00:00:00+00:00",
            )
            payload = plan.to_dict()
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["candidates"][0]["target_weight_bits"], 4)
            self.assertFalse(payload["candidates"][0]["executable"])
            self.assertIn(
                "adapter_unavailable",
                payload["candidates"][0]["blocking_reasons"],
            )
            validate_instance(payload, schema("optimization-plan-v1.schema.json"))
            validate_instance(
                calibration.to_dict(), schema("calibration-manifest-v1.schema.json")
            )
            self.assertFalse((Path(directory) / "artifact").exists())

    def test_resource_budget_blocks_candidates_without_allocating(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            model = InspectedModel(
                "large", source, {}, ModelMemorySpec("large", 8 * GIB, 1), 2
            )
            plan = build_dry_run_plan(
                model,
                hardware(),
                Path(directory) / "artifact",
                OptimizationObjective.BALANCED,
                ResourceBudget(1 * GIB, 1 * GIB),
            )
            self.assertTrue(all(not item.within_budget for item in plan.candidates))
            self.assertTrue(
                all("memory_budget_exceeded" in item.blocking_reasons for item in plan.candidates)
            )

    def test_output_path_rejects_overlap_and_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            with self.assertRaises(OptimizationPathError):
                validate_immutable_output_path(source, source / "artifact")
            existing = Path(directory) / "existing"
            existing.mkdir()
            with self.assertRaises(OptimizationPathError):
                validate_immutable_output_path(source, existing)

    def test_optimizer_events_are_bounded_and_schema_valid(self) -> None:
        bus = OptimizerEventBus(capacity=2)
        bus.publish("plan", "inspect", OptimizerState.PLANNING, 0.0, "optimizer.inspect")
        bus.publish("plan", "plan", OptimizerState.READY, 0.5, "optimizer.ready")
        latest = bus.publish("plan", "done", OptimizerState.COMPLETED, 1.0, "optimizer.done")
        self.assertEqual([event.event_id for event in bus.snapshot()], ["2", "3"])
        validate_instance(latest.to_dict(), schema("optimizer-event-v1.schema.json"))
        with self.assertRaises(ValueError):
            bus.publish("plan", "bad", OptimizerState.RUNNING, float("nan"), "bad")

    def test_artifact_manifest_records_provenance_and_matches_schema(self) -> None:
        manifest = ArtifactManifest(
            artifact_id="artifact-1",
            created_at="2026-08-22T00:00:00+00:00",
            plan_id="plan-1",
            source_hash="a" * 64,
            output_hash="b" * 64,
            output_bytes=1024,
            elapsed_milliseconds=250,
            peak_child_rss_bytes=2048,
            transforms=({"type": "int8"},),
            tool_versions={"vllm-apple": "0.1.0"},
            calibration_fingerprint=None,
            evaluation={"code": 0.98},
            license="test",
        )
        validate_instance(manifest.to_dict(), schema("artifact-manifest-v1.schema.json"))

    def test_perplexity_quality_gate_is_slice_bound_and_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)

            def report(model_hash: str, values: tuple[float, float, float]):
                slices = tuple(
                    PerplexitySlice("general", language, 1, 10, math.log(value), value)
                    for language, value in zip(("en", "ja", "zh-CN"), values)
                )
                return PerplexityEvaluationReport(
                    model_path=str(root / model_hash[:4]),
                    model_hash=model_hash,
                    dataset_path=str(root / "dataset.jsonl"),
                    dataset_fingerprint="d" * 64,
                    sample_count=3,
                    token_count=30,
                    mean_negative_log_likelihood=2.0,
                    perplexity=math.exp(2.0),
                    elapsed_milliseconds=100,
                    peak_rss_bytes=1024,
                    slices=slices,
                )

            baseline = report("a" * 64, (10.0, 20.0, 30.0))
            candidate = report("b" * 64, (10.5, 21.0, 40.0))
            validate_instance(
                baseline.to_dict(),
                schema("perplexity-evaluation-v1.schema.json"),
            )
            rejected = compare_perplexity_reports(baseline, candidate, 0.10)
            self.assertFalse(rejected.approved)
            self.assertEqual([value.passed for value in rejected.slices], [True, True, False])
            validate_instance(rejected.to_dict(), schema("quality-gate-v1.schema.json"))
            baseline_path = persist_evaluation_report(
                baseline.to_dict(),
                root / "baseline.json",
            )
            candidate_path = persist_evaluation_report(
                candidate.to_dict(),
                root / "candidate.json",
            )
            gate_output = root / "gate.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = optimizer_main(
                    [
                        "quality-gate",
                        "--baseline",
                        str(baseline_path),
                        "--candidate",
                        str(candidate_path),
                        "--max-perplexity-regression",
                        "0.10",
                        "--output",
                        str(gate_output),
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertEqual(gate_output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(stdout.getvalue()), json.loads(gate_output.read_text()))

    def test_generation_quality_gate_is_bounded_and_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)

            def sample(
                sample_id: str,
                language: str,
                tokens: tuple[int, ...],
                expectation_score: float,
            ) -> GenerationSampleResult:
                return GenerationSampleResult(
                    sample_id=sample_id,
                    domain="general",
                    language=language,
                    prompt_token_count=8,
                    token_ids=tokens,
                    output_fingerprint=generation_token_fingerprint(tokens),
                    expectation_score=expectation_score,
                )

            def report(
                model_hash: str,
                samples: tuple[GenerationSampleResult, ...],
            ) -> GenerationEvaluationReport:
                return GenerationEvaluationReport(
                    model_path=str(root / model_hash[:4]),
                    model_hash=model_hash,
                    dataset_path=str(root / "generation.jsonl"),
                    dataset_fingerprint="d" * 64,
                    prompt_format="raw",
                    maximum_prompt_tokens=128,
                    maximum_new_tokens=4,
                    elapsed_milliseconds=100,
                    peak_rss_bytes=1024,
                    samples=samples,
                )

            baseline = report(
                "a" * 64,
                (
                    sample("en", "en", (1, 2, 3, 4), 1.0),
                    sample("ja", "ja", (5, 6, 7, 8), 1.0),
                ),
            )
            candidate = report(
                "b" * 64,
                (
                    sample("en", "en", (1, 2, 3, 4), 1.0),
                    sample("ja", "ja", (5, 9, 7, 10), 0.0),
                ),
            )
            validate_instance(
                baseline.to_dict(),
                schema("generation-evaluation-v1.schema.json"),
            )
            rejected = compare_generation_reports(
                baseline,
                candidate,
                minimum_token_agreement=0.75,
                maximum_expectation_regression=0.0,
            )
            self.assertFalse(rejected.approved)
            self.assertEqual([value.passed for value in rejected.samples], [True, False])
            validate_instance(
                rejected.to_dict(),
                schema("generation-quality-gate-v1.schema.json"),
            )
            baseline_path = persist_evaluation_report(
                baseline.to_dict(),
                root / "baseline-generation.json",
            )
            candidate_path = persist_evaluation_report(
                candidate.to_dict(),
                root / "candidate-generation.json",
            )
            gate_output = root / "generation-gate.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = optimizer_main(
                    [
                        "generation-quality-gate",
                        "--baseline",
                        str(baseline_path),
                        "--candidate",
                        str(candidate_path),
                        "--min-token-agreement",
                        "0.75",
                        "--max-expectation-regression",
                        "0",
                        "--output",
                        str(gate_output),
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertEqual(gate_output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(stdout.getvalue()), json.loads(gate_output.read_text()))

    def test_multilingual_task_suite_and_filters_are_bounded(self) -> None:
        dataset = Path("docs/evaluation/task-suite-multilingual-v1.jsonl")
        records = tuple(
            _parse_generation_record(line)
            for line in dataset.read_bytes().splitlines(keepends=True)
        )
        self.assertEqual(len(records), 9)
        self.assertEqual(
            {record["domain"] for record in records},
            {"code", "mathematics", "long-context"},
        )
        self.assertEqual(
            {record["language"] for record in records},
            {"en", "ja", "zh-CN"},
        )
        self.assertEqual(
            _validated_filters("domain", ("code", "code", "mathematics")),
            frozenset({"code", "mathematics"}),
        )
        self.assertEqual(_expectation_score(" 8\n", ["8"], "prefix"), 1.0)
        self.assertEqual(_expectation_score(" 18\n", ["8"], "prefix"), 0.0)
        self.assertEqual(_expectation_score("answer: 8", ["8"], "contains"), 1.0)

        class FakeTokenizer:
            def apply_chat_template(self, messages, **options):
                self.messages = messages
                self.options = options
                return "<user>question<model>"

        tokenizer = FakeTokenizer()
        self.assertEqual(
            _formatted_prompt(tokenizer, "question", True),
            "<user>question<model>",
        )
        self.assertEqual(tokenizer.messages[0]["content"], "question")
        self.assertTrue(tokenizer.options["add_generation_prompt"])
        with self.assertRaises(ValueError):
            _validated_filters("domain", tuple(str(value) for value in range(17)))

    def test_bounded_io_profile_enables_duration_estimate(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory) / "source"
            workspace = Path(directory) / "workspace"
            source.mkdir()
            workspace.mkdir()
            (source / "weights.safetensors").write_bytes(bytes(1024 * 1024))
            profile = profile_optimizer_io(source, workspace, hardware(), 1024 * 1024)
            validate_instance(profile.to_dict(), schema("performance-profile-v1.schema.json"))
            model = InspectedModel(
                "model", source, {}, ModelMemorySpec("model", GIB, 1), 2
            )
            plan = build_dry_run_plan(
                model,
                hardware(),
                Path(directory) / "artifact",
                OptimizationObjective.BALANCED,
                ResourceBudget(4 * GIB, 4 * GIB, 1),
                performance_profile=profile,
            )
            self.assertIsNotNone(plan.candidates[0].estimated_duration_seconds)
            self.assertFalse(any(workspace.iterdir()))

    def test_structured_optimizer_failure_matches_schema(self) -> None:
        failure = OptimizerFailure(
            OptimizerErrorCode.INSUFFICIENT_DISK,
            "optimizer.error.insufficient_disk",
            Recoverability.USER_ACTION_REQUIRED,
            "free space is below the plan budget",
        )
        validate_instance(failure.to_dict(), schema("optimizer-error-v1.schema.json"))

    def test_cli_emits_structured_error(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = optimizer_main(
                ["plan", "/model/does-not-exist", "--output", "/tmp/artifact"]
            )
        payload = json.loads(stderr.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["error"]["code"], "invalid_model")
        validate_instance(payload["error"], schema("optimizer-error-v1.schema.json"))

    def test_performance_profile_rejects_untyped_input(self) -> None:
        with self.assertRaises(ValueError):
            OptimizationPerformanceProfile.from_dict(
                {
                    "schema_version": 1,
                    "profile_id": "profile",
                    "measured_at": "2026-08-22T00:00:00+00:00",
                    "hardware_fingerprint": "a" * 64,
                    "sample_bytes": True,
                    "read_bytes_per_second": 1,
                    "write_bytes_per_second": 1,
                }
            )

    def test_mlx_adapter_capabilities_are_versioned_and_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory)
            (source / "weights.safetensors").write_bytes(b"weights")
            model = InspectedModel(
                "model",
                source,
                {"torch_dtype": "bfloat16"},
                ModelMemorySpec("model", 7, 1),
                2,
            )
            adapter = compatible_mlx_adapter()
            report = AdapterRegistry((adapter,)).detect(model)
            payload = report.to_dict()
            capability = payload["adapters"][0]
            self.assertTrue(capability["available"])
            self.assertTrue(capability["compatible"])
            self.assertTrue(capability["executable"])
            self.assertEqual(capability["issues"], [])
            validate_instance(payload, schema("adapter-capabilities-v1.schema.json"))

    def test_adapter_registry_rejects_duplicate_identifiers(self) -> None:
        adapter = MLXOptimizationAdapter(lambda _: None)
        with self.assertRaises(ValueError):
            AdapterRegistry((adapter, adapter))

    def test_adapter_detection_rejects_gguf_for_mlx_without_loading_weights(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory)
            (source / "weights.gguf").write_bytes(b"weights")
            model = InspectedModel(
                "model",
                source,
                {"torch_dtype": "float16"},
                ModelMemorySpec("model", 7, 1),
                2,
            )
            capability = AdapterRegistry(
                (compatible_mlx_adapter(),)
            ).detect(model).adapters[0]
            self.assertFalse(capability.compatible)
            self.assertIn("source_format_unsupported:gguf", capability.issues)

    def test_capabilities_cli_emits_schema_valid_report(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory)
            config = {
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_size": 8,
                "torch_dtype": "float16",
            }
            (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (source / "weights.safetensors").write_bytes(b"weights")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = optimizer_main(["capabilities", str(source)])
            self.assertEqual(exit_code, 0)
            validate_instance(
                json.loads(stdout.getvalue()),
                schema("adapter-capabilities-v1.schema.json"),
            )

    def test_mlx_adapter_rejects_untested_dependency_version(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory)
            (source / "weights.safetensors").write_bytes(b"weights")
            model = InspectedModel(
                "model",
                source,
                {"torch_dtype": "float16"},
                ModelMemorySpec("model", 7, 1),
                2,
            )
            versions = {"mlx": "0.32.0", "mlx-lm": "0.32.0"}
            adapter = MLXOptimizationAdapter(
                lambda package: versions[package],
                lambda: ("Darwin", "arm64"),
                sys.executable,
            )
            capability = AdapterRegistry((adapter,)).detect(model).adapters[0]
            self.assertFalse(capability.executable)
            self.assertIn("dependency_version_unsupported:mlx:0.32.0", capability.issues)

    def test_mlx_adapter_detects_dtype_from_bounded_safetensors_header(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory)
            header = json.dumps(
                {
                    "weight": {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 4],
                    }
                }
            ).encode("utf-8")
            (source / "model.safetensors").write_bytes(
                len(header).to_bytes(8, "little") + header + bytes(4)
            )
            model = InspectedModel(
                "gpt-like",
                source,
                {},
                ModelMemorySpec("gpt-like", 4, 1),
                2,
            )
            capability = AdapterRegistry((compatible_mlx_adapter(),)).detect(model)
            self.assertEqual(capability.source_dtype, "float32")
            self.assertTrue(capability.adapters[0].executable)

    def test_mlx_export_invocation_is_bounded_and_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text('{"torch_dtype":"float16"}')
            (source / "weights.safetensors").write_bytes(b"weights")
            model = InspectedModel(
                "model",
                source,
                {"torch_dtype": "float16"},
                ModelMemorySpec("model", 7, 1),
                2,
            )
            invocation = compatible_mlx_adapter().build_export_invocation(
                model,
                root / "artifact",
                target_weight_bits=4,
                group_size=64,
                maximum_output_bytes=1024,
            )
            (source / "weights.safetensors").write_bytes(b"WEIGHTS")
            changed_invocation = compatible_mlx_adapter().build_export_invocation(
                model,
                root / "changed-artifact",
                target_weight_bits=4,
                group_size=64,
                maximum_output_bytes=1024,
            )
            self.assertNotEqual(
                invocation.source_fingerprint,
                changed_invocation.source_fingerprint,
            )
            (root / "artifact").mkdir()
            with self.assertRaises(OptimizationPathError):
                compatible_mlx_adapter().build_export_invocation(
                    model,
                    root / "artifact",
                    target_weight_bits=4,
                    group_size=64,
                    maximum_output_bytes=1024,
                )
            resumed_invocation = compatible_mlx_adapter().build_export_invocation(
                model,
                root / "artifact",
                target_weight_bits=4,
                group_size=64,
                maximum_output_bytes=1024,
                allow_existing_output=True,
            )
            self.assertEqual(resumed_invocation.output_path, invocation.output_path)
            payload = invocation.to_dict()
            self.assertEqual(payload["command"][0], str(Path(sys.executable).absolute()))
            self.assertEqual(payload["command"][1:4], ["-m", "mlx_lm", "convert"])
            mlx_path_index = payload["command"].index("--mlx-path")
            self.assertEqual(payload["command"][mlx_path_index + 1], "mlx-output")
            self.assertEqual(payload["command"][-2:], ["--q-bits", "4"])
            self.assertNotIn("--trust-remote-code", payload["command"])
            self.assertNotIn("--upload-repo", payload["command"])
            validate_instance(payload, schema("mlx-export-invocation-v1.schema.json"))
            invalid_payload = dict(payload)
            invalid_payload["source_fingerprint"] = payload["source_fingerprint"] + "0"
            with self.assertRaises(SchemaValidationError):
                validate_instance(
                    invalid_payload,
                    schema("mlx-export-invocation-v1.schema.json"),
                )
            calls = {}
            sentinel = object()

            class RecordingWorker:
                def run(self, **arguments):
                    calls.update(arguments)
                    return sentinel

            returned = compatible_mlx_adapter().execute_export(
                invocation,
                plan_id="plan",
                worker=RecordingWorker(),
                checkpoint_store=CheckpointStore(root / "checkpoints"),
            )
            self.assertIs(returned, sentinel)
            self.assertEqual(calls["source_fingerprint"], invocation.source_fingerprint)
            self.assertEqual(calls["command"], invocation.command)

    def test_export_cli_defaults_to_side_effect_free_invocation(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            config = {
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_size": 8,
                "torch_dtype": "float16",
            }
            (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (source / "weights.safetensors").write_bytes(b"weights")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = optimizer_main(
                    [
                        "export",
                        str(source),
                        "--output",
                        str(root / "artifact"),
                        "--checkpoint-root",
                        str(root / "checkpoints"),
                        "--plan-id",
                        "plan",
                        "--max-output-gb",
                        "0.001",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertFalse((root / "artifact").exists())
            self.assertFalse((root / "checkpoints").exists())
            validate_instance(
                json.loads(stdout.getvalue()),
                schema("mlx-export-invocation-v1.schema.json"),
            )

    def test_mlx_export_report_persists_private_idempotent_provenance(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text('{"torch_dtype":"float16"}')
            (source / "weights.safetensors").write_bytes(b"source-weights")
            model = InspectedModel(
                "model",
                source,
                {"torch_dtype": "float16"},
                ModelMemorySpec("model", 14, 1),
                2,
            )
            adapter = compatible_mlx_adapter()
            invocation = adapter.build_export_invocation(
                model,
                root / "artifact",
                target_weight_bits=4,
                group_size=64,
                maximum_output_bytes=1024,
            )
            result = IsolatedConversionWorker().run(
                plan_id="plan-report",
                source=source,
                output=root / "artifact",
                command=(
                    sys.executable,
                    "-c",
                    "from pathlib import Path;Path('model.safetensors').write_bytes(b'converted')",
                ),
                maximum_output_bytes=1024,
            )
            manifest = adapter.build_artifact_manifest(
                invocation,
                result,
                plan_id="plan-report",
                license_name="apache-2.0",
            )
            manifest_path, persisted = persist_artifact_manifest(
                manifest,
                root / "artifact.manifest.json",
                source_path=source,
            )
            report = MLXExportReport(result, persisted, str(manifest_path))
            validate_instance(report.to_dict(), schema("mlx-export-report-v1.schema.json"))
            self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(result.output_hash), 64)
            self.assertGreater(result.peak_child_rss_bytes, 0)
            second_path, second_manifest = persist_artifact_manifest(
                adapter.build_artifact_manifest(
                    invocation,
                    result,
                    plan_id="plan-report",
                    license_name="apache-2.0",
                ),
                root / "artifact.manifest.json",
                source_path=source,
            )
            self.assertEqual(second_path, manifest_path)
            self.assertEqual(second_manifest.created_at, persisted.created_at)

    def test_artifact_transaction_atomically_promotes_private_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "artifact"
            source.mkdir()
            with ArtifactTransaction(
                source,
                output,
                maximum_output_bytes=1024,
            ) as transaction:
                self.assertEqual(transaction.workspace.stat().st_mode & 0o777, 0o700)
                (transaction.workspace / "weights.bin").write_bytes(b"weights")
                file_count, output_bytes = transaction.promote()
                first_hash = transaction.output_hash
            self.assertEqual((file_count, output_bytes), (1, 7))
            with ArtifactTransaction(
                source,
                root / "artifact-copy",
                maximum_output_bytes=1024,
            ) as copy_transaction:
                (copy_transaction.workspace / "weights.bin").write_bytes(b"weights")
                copy_transaction.promote()
                self.assertEqual(copy_transaction.output_hash, first_hash)
            self.assertEqual((output / "weights.bin").read_bytes(), b"weights")
            self.assertFalse(any(root.glob(".artifact.work-*")))

    def test_artifact_transaction_rejects_symlink_and_cleans_workspace(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "artifact"
            source.mkdir()
            with self.assertRaises(ArtifactValidationError):
                with ArtifactTransaction(
                    source,
                    output,
                    maximum_output_bytes=1024,
                ) as transaction:
                    os.symlink(source, transaction.workspace / "unsafe")
                    transaction.promote()
            self.assertFalse(output.exists())
            self.assertFalse(any(root.glob(".artifact.work-*")))

    def test_artifact_transaction_never_replaces_output_that_appears(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "artifact"
            source.mkdir()
            with self.assertRaises(OptimizationPathError):
                with ArtifactTransaction(
                    source,
                    output,
                    maximum_output_bytes=1024,
                ) as transaction:
                    (transaction.workspace / "weights.bin").write_bytes(b"new")
                    output.mkdir()
                    (output / "existing").write_bytes(b"keep")
                    transaction.promote()
            self.assertEqual((output / "existing").read_bytes(), b"keep")
            self.assertFalse((output / "weights.bin").exists())

    def test_isolated_worker_bounds_logs_and_publishes_validated_artifact(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "artifact"
            source.mkdir()
            worker = IsolatedConversionWorker(log_tail_bytes=1024)
            script = (
                "from pathlib import Path;"
                "Path('weights.bin').write_bytes(b'weights');"
                "print('x' * 100000)"
            )
            result = worker.run(
                plan_id="plan-worker",
                source=source,
                output=output,
                command=(sys.executable, "-c", script),
                maximum_output_bytes=1024,
                timeout_seconds=5,
            )
            self.assertEqual(result.state, OptimizerState.COMPLETED)
            self.assertLessEqual(len(result.stdout_tail.encode()), 1024)
            self.assertEqual(len(result.output_hash), 64)
            self.assertGreater(result.elapsed_milliseconds, 0)
            self.assertGreater(result.peak_child_rss_bytes, 0)
            self.assertEqual((output / "weights.bin").read_bytes(), b"weights")
            validate_instance(result.to_dict(), schema("worker-result-v1.schema.json"))

    def test_running_worker_can_be_cancelled_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "artifact"
            source.mkdir()
            token = CancellationToken()
            worker = IsolatedConversionWorker(poll_interval=0.01, terminate_grace_seconds=0.2)
            results = []

            def run_worker() -> None:
                results.append(
                    worker.run(
                        plan_id="plan-cancel",
                        source=source,
                        output=output,
                        command=(sys.executable, "-c", "import time;time.sleep(30)"),
                        maximum_output_bytes=1024,
                        cancellation=token,
                    )
                )

            thread = threading.Thread(target=run_worker)
            thread.start()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if any(event.state == OptimizerState.RUNNING for event in worker.events.snapshot()):
                    break
                time.sleep(0.01)
            token.cancel()
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(results[0].state, OptimizerState.CANCELLED)
            self.assertFalse(output.exists())
            self.assertFalse(any(root.glob(".artifact.work-*")))

    def test_failed_worker_cleans_workspace_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "artifact"
            source.mkdir()
            result = IsolatedConversionWorker().run(
                plan_id="plan-failed",
                source=source,
                output=output,
                command=(sys.executable, "-c", "raise SystemExit(7)"),
                maximum_output_bytes=1024,
                timeout_seconds=5,
            )
            self.assertEqual(result.state, OptimizerState.FAILED)
            self.assertEqual(result.exit_code, 7)
            self.assertFalse(output.exists())
            self.assertFalse(any(root.glob(".artifact.work-*")))
            validate_instance(result.to_dict(), schema("worker-result-v1.schema.json"))

    def test_checkpoint_store_is_private_atomic_and_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            store = CheckpointStore(root / "checkpoints")
            fingerprint = execution_fingerprint((sys.executable, "-m", "adapter"))
            checkpoint = CheckpointManifest.create(
                plan_id="../../plan",
                source_path=source,
                source_fingerprint="a" * 64,
                output_path=root / "artifact",
                execution_fingerprint=fingerprint,
                maximum_output_bytes=1024,
            )
            path = store.save(checkpoint)
            self.assertEqual(path.parent, store.root)
            self.assertEqual(store.root.stat().st_mode & 0o777, 0o700)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            loaded = store.load("../../plan")
            self.assertEqual(loaded, checkpoint)
            validate_instance(loaded.to_dict(), schema("checkpoint-manifest-v1.schema.json"))

    def test_checkpoint_store_atomically_replaces_attempt(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            store = CheckpointStore(root / "checkpoints")
            checkpoint = CheckpointManifest.create(
                plan_id="plan",
                source_path=source,
                source_fingerprint="a" * 64,
                output_path=root / "artifact",
                execution_fingerprint=execution_fingerprint(("adapter",)),
                maximum_output_bytes=1024,
            )
            store.save(checkpoint)
            store.save(checkpoint.next_attempt())
            self.assertEqual(store.load("plan").attempt, 2)
            self.assertFalse(any(store.root.glob(".*.json.*")))

    def test_resume_decision_is_bound_to_conversion_and_safe_workspace(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "artifact"
            workspace = root / ".artifact.work-resume"
            source.mkdir()
            workspace.mkdir(mode=0o700)
            fingerprint = execution_fingerprint(("adapter", "--int4"), {"MODE": "safe"})
            checkpoint = CheckpointManifest.create(
                plan_id="plan",
                source_path=source,
                source_fingerprint="a" * 64,
                output_path=output,
                execution_fingerprint=fingerprint,
                maximum_output_bytes=1024,
            ).transition(CheckpointStage.CONVERTED, workspace_path=workspace)
            action = decide_resume(
                checkpoint,
                source_path=source,
                source_fingerprint="a" * 64,
                output_path=output,
                execution_fingerprint_value=fingerprint,
                maximum_output_bytes=1024,
            )
            self.assertEqual(action, ResumeAction.RESUME_VALIDATION)
            with self.assertRaises(CheckpointError):
                decide_resume(
                    checkpoint,
                    source_path=source,
                    source_fingerprint="b" * 64,
                    output_path=output,
                    execution_fingerprint_value=fingerprint,
                    maximum_output_bytes=1024,
                )

    def test_checkpoint_parser_rejects_boolean_integer(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            payload = CheckpointManifest.create(
                plan_id="plan",
                source_path=source,
                source_fingerprint="a" * 64,
                output_path=Path(directory) / "artifact",
                execution_fingerprint=execution_fingerprint(("adapter",)),
                maximum_output_bytes=1024,
            ).to_dict()
            payload["attempt"] = True
            with self.assertRaises(CheckpointError):
                CheckpointManifest.from_dict(payload)
            payload["attempt"] = 1
            payload["updated_at"] = "not-a-timestamp"
            with self.assertRaises(CheckpointError):
                CheckpointManifest.from_dict(payload)

    def test_checkpoint_lease_rejects_concurrent_plan_and_releases(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory) / "checkpoints"
            first_store = CheckpointStore(root)
            second_store = CheckpointStore(root)
            with first_store.acquire("plan"):
                with self.assertRaises(CheckpointLeaseError):
                    second_store.acquire("plan")
            with second_store.acquire("plan") as lease:
                self.assertTrue(lease.path.is_file())
                self.assertEqual(lease.path.stat().st_mode & 0o777, 0o600)

    def test_checkpoint_lease_is_recovered_by_kernel_after_process_exit(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            store = CheckpointStore(Path(directory) / "checkpoints")
            script = (
                "import os,sys;"
                "from pathlib import Path;"
                "from vllm_apple.optimizer import CheckpointStore;"
                "lease=CheckpointStore(Path(sys.argv[1])).acquire('plan');"
                "os._exit(9)"
            )
            completed = subprocess.run(
                (sys.executable, "-c", script, str(store.root)),
                check=False,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 9)
            with store.acquire("plan"):
                pass

    def test_checkpointed_worker_records_completed_stage(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "artifact"
            source.mkdir()
            store = CheckpointStore(root / "checkpoints")
            command = (
                sys.executable,
                "-c",
                "from pathlib import Path;Path('weights.bin').write_bytes(b'weights')",
            )
            result = IsolatedConversionWorker().run(
                plan_id="plan",
                source=source,
                source_fingerprint="a" * 64,
                output=output,
                command=command,
                maximum_output_bytes=1024,
                checkpoint_store=store,
            )
            checkpoint = store.load("plan")
            self.assertEqual(result.state, OptimizerState.COMPLETED)
            self.assertEqual(checkpoint.stage, CheckpointStage.COMPLETED)
            self.assertEqual(checkpoint.output_bytes, 7)
            self.assertEqual(checkpoint.output_hash, result.output_hash)
            self.assertEqual(checkpoint.elapsed_milliseconds, result.elapsed_milliseconds)
            self.assertEqual(checkpoint.peak_child_rss_bytes, result.peak_child_rss_bytes)
            resumed = IsolatedConversionWorker().run(
                plan_id="plan",
                source=source,
                source_fingerprint="a" * 64,
                output=output,
                command=command,
                maximum_output_bytes=1024,
                checkpoint_store=store,
                resume=True,
            )
            self.assertEqual(resumed.state, OptimizerState.COMPLETED)
            self.assertEqual(resumed.output_hash, result.output_hash)
            self.assertEqual(store.load("plan").attempt, 1)

    def test_worker_resumes_converted_workspace_without_rerunning_command(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "artifact"
            workspace = root / ".artifact.work-resume"
            source.mkdir()
            workspace.mkdir(mode=0o700)
            (workspace / "weights.bin").write_bytes(b"converted")
            store = CheckpointStore(root / "checkpoints")
            command = (sys.executable, "-c", "raise SystemExit(99)")
            checkpoint = CheckpointManifest.create(
                plan_id="plan",
                source_path=source,
                source_fingerprint="a" * 64,
                output_path=output,
                execution_fingerprint=execution_fingerprint(command),
                maximum_output_bytes=1024,
            ).transition(CheckpointStage.CONVERTED, workspace_path=workspace)
            store.save(checkpoint)
            result = IsolatedConversionWorker().run(
                plan_id="plan",
                source=source,
                source_fingerprint="a" * 64,
                output=output,
                command=command,
                maximum_output_bytes=1024,
                checkpoint_store=store,
                resume=True,
            )
            self.assertEqual(result.state, OptimizerState.COMPLETED)
            self.assertEqual((output / "weights.bin").read_bytes(), b"converted")
            self.assertEqual(store.load("plan").stage, CheckpointStage.COMPLETED)

    def test_worker_reconciles_promoted_artifact_after_checkpoint_gap(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "artifact"
            workspace = root / ".artifact.work-gap"
            source.mkdir()
            workspace.mkdir(mode=0o700)
            (workspace / "weights.bin").write_bytes(b"converted")
            command = (sys.executable, "-c", "raise SystemExit(99)")
            checkpoint = CheckpointManifest.create(
                plan_id="plan-gap",
                source_path=source,
                source_fingerprint="a" * 64,
                output_path=output,
                execution_fingerprint=execution_fingerprint(command),
                maximum_output_bytes=1024,
            ).transition(CheckpointStage.CONVERTED, workspace_path=workspace)
            os.rename(workspace, output)
            store = CheckpointStore(root / "checkpoints")
            store.save(checkpoint)
            result = IsolatedConversionWorker().run(
                plan_id="plan-gap",
                source=source,
                source_fingerprint="a" * 64,
                output=output,
                command=command,
                maximum_output_bytes=1024,
                checkpoint_store=store,
                resume=True,
            )
            self.assertEqual(result.state, OptimizerState.COMPLETED)
            self.assertEqual(store.load("plan-gap").stage, CheckpointStage.COMPLETED)


if __name__ == "__main__":
    unittest.main()
