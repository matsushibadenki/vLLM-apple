import Foundation
import Testing
@testable import VLLMAppleKit

@Test func decodesHealthEnvelope() throws {
    let data = Data("""
    {
      "status": "degraded",
      "control_ready": true,
      "inference_ready": false,
      "api_version": "v1",
      "schema_version": 1,
      "runtime_version": "0.1.0",
      "minimum_client_version": "0.1.0"
    }
    """.utf8)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let health = try decoder.decode(HealthStatus.self, from: data)
    #expect(health.status == .degraded)
    #expect(health.controlReady)
    #expect(!health.inferenceReady)
}

@Test func runtimeErrorsExposeLocalizableKeys() {
    #expect(RuntimeClientError.invalidResponse.messageKey == "runtime.error.invalid_response")
    #expect(ManagedRuntimeError.readinessTimedOut.messageKey == "runtime.error.readiness_timed_out")
}

@Test func decodesRuntimeEventPayload() throws {
    let data = Data("""
    {
      "schema_version": 1,
      "event_id": "42",
      "type": "runtime.state",
      "timestamp": "2026-08-21T00:00:00Z",
      "payload": {"state": "ready", "inference_ready": true}
    }
    """.utf8)
    let event = try JSONDecoder().decode(RuntimeEvent.self, from: data)
    #expect(event.eventID == "42")
    #expect(event.payload["state"] == .string("ready"))
}

@Test func runtimeFailureEventExposesRecoverability() throws {
    let data = Data("""
    {
      "schema_version": 1,
      "event_id": "43",
      "type": "runtime.failure",
      "timestamp": "2026-08-28T00:00:00Z",
      "payload": {
        "state": "failed",
        "failure": {
          "schema_version": 1,
          "code": "backend_readiness_timeout",
          "message_key": "runtime.error.backend_readiness_timeout",
          "recoverability": "retryable",
          "detail_fingerprint": "0123456789abcdef01234567"
        }
      }
    }
    """.utf8)
    let event = try JSONDecoder().decode(RuntimeEvent.self, from: data)
    #expect(event.runtimeFailure?.code == "backend_readiness_timeout")
    #expect(event.runtimeFailure?.recoverability == .retryable)
    #expect(event.runtimeFailure?.messageKey == "runtime.error.backend_readiness_timeout")
}

@Test func decodesUnifiedMemoryBudgetWithoutDoubleCountingMetalEnvelope() throws {
    let data = Data("""
    {
      "capacity_bytes": 1000,
      "known_component_bytes": 650,
      "known_remaining_bytes": 350,
      "overcommitted_bytes": 0,
      "unknown_components": [],
      "overlap_envelope_bytes": 700,
      "components": {
        "weights": {"current_bytes": 400, "peak_bytes": 400, "source": "manifest", "accounting": "additive"},
        "kv": {"current_bytes": 100, "peak_bytes": 100, "source": "vllm", "accounting": "additive"},
        "prefix": {"current_bytes": 50, "peak_bytes": 50, "source": "semantic", "accounting": "additive"},
        "scratch": {"current_bytes": 25, "peak_bytes": 25, "source": "scheduler", "accounting": "additive"},
        "metal_heap": {"current_bytes": 700, "peak_bytes": 700, "source": "ioreg", "accounting": "overlap_envelope"},
        "coreml": {"current_bytes": 75, "peak_bytes": 75, "source": "coreml", "accounting": "additive"}
      }
    }
    """.utf8)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let budget = try decoder.decode(MemoryBudget.self, from: data)
    #expect(budget.knownComponentBytes == 650)
    #expect(budget.overlapEnvelopeBytes == 700)
    #expect(budget.components.metalHeap.accounting == .overlapEnvelope)
}

@Test func decodesKVCalibrationProvenance() throws {
    let data = Data("""
    {
      "enabled": true,
      "status": "applied",
      "backend": "vllm_metal",
      "evaluation_id": "0123456789abcdef01234567",
      "calibrated_bytes_per_token": 11902,
      "maximum_observed_context": 3927,
      "sample_count": 3,
      "safety_margin_ratio": 0.25
    }
    """.utf8)
    let calibration = try JSONDecoder().decode(KVCalibrationProvenance.self, from: data)
    #expect(calibration.status == .applied)
    #expect(calibration.backend == "vllm_metal")
    #expect(calibration.calibratedBytesPerToken == 11902)
    #expect(calibration.maximumObservedContext == 3927)
    #expect(calibration.sampleCount == 3)
}

@Test func decodesNativeV2TuningStateAndEvent() throws {
    let stateData = Data("""
    {
      "enabled": true,
      "status": "applied",
      "run_id": 3,
      "profile_id": "0123456789abcdef01234567",
      "error_code": null
    }
    """.utf8)
    let state = try JSONDecoder().decode(NativeV2TuningState.self, from: stateData)
    #expect(state.status == .applied)
    #expect(state.enabled)
    #expect(state.runID == 3)
    #expect(state.profileID == "0123456789abcdef01234567")

    let eventData = Data("""
    {
      "schema_version": 1,
      "event_id": "45",
      "type": "runtime.native_v2_tuning",
      "timestamp": "2026-08-29T00:00:00Z",
      "payload": {"status": "running", "run_id": 4}
    }
    """.utf8)
    let event = try JSONDecoder().decode(RuntimeEvent.self, from: eventData)
    #expect(event.nativeV2Tuning?.status == .running)
    #expect(event.nativeV2Tuning?.runID == 4)
}

@Test func contextReevaluationEventExposesReducedLimit() throws {
    let data = Data("""
    {
      "schema_version": 1,
      "event_id": "44",
      "type": "runtime.context_reevaluation",
      "timestamp": "2026-08-28T00:00:00Z",
      "payload": {
        "status": "reduced",
        "configured_context_tokens": 4096,
        "effective_context_tokens": 2048,
        "capacity_context_tokens": 2048
      }
    }
    """.utf8)
    let event = try JSONDecoder().decode(RuntimeEvent.self, from: data)
    #expect(event.contextReevaluation?.status == .reduced)
    #expect(event.contextReevaluation?.configuredContextTokens == 4096)
    #expect(event.contextReevaluation?.effectiveContextTokens == 2048)
}

@Test func decodesQualificationReportWithTypedContextResult() throws {
    let data = Data("""
    {
      "schema_version": 1,
      "model": "Qwen/example",
      "backend": "vllm_metal",
      "requested_modes": ["text"],
      "load_seconds": 3.25,
      "shutdown_clean": true,
      "promotion_probe": {"passed": true},
      "phase_profile": {
        "schema_version": 1,
        "profile_id": "0123456789abcdef01234567",
        "hardware_fingerprint": "apple-m4-32gb",
        "model_id": "Qwen/example",
        "backend": "mlx_lm",
        "sample_count": 3,
        "prefill": {
          "prompt_tokens": 96,
          "ttft": {"mean_ms": 125.5, "p50_upper_bound_ms": 250, "p95_upper_bound_ms": 500, "max_ms": 300.0}
        },
        "decode": {
          "output_tokens": 96,
          "token_intervals": 93,
          "duration_ms": 1500.0,
          "tpot": {"mean_ms": 16.1, "p50_upper_bound_ms": 25, "p95_upper_bound_ms": 50, "max_ms": 40.0},
          "tokens_per_second": 62.0
        },
        "peak_memory_bytes": 17179869184,
        "storage": {"latency_bucket_count": 26, "raw_sample_count": 0}
      },
      "model_memory_fit": {
        "artifact_bytes": 12000000000,
        "estimated_resident_bytes": 18000000000,
        "hard_ceiling_bytes": 24000000000,
        "context_tokens": 262144,
        "fits": true
      },
      "quality_smoke": {
        "schema_version": 1,
        "sample_count": 3,
        "checks": {"english": true, "japanese": true, "simplified_chinese": true},
        "stores_generated_text": false,
        "passed": true
      },
      "soak": {"passed": true, "requests": 24},
      "context_reevaluation": {
        "enabled": true,
        "status": "reduced",
        "configured_context_tokens": 4096,
        "effective_context_tokens": 2048,
        "capacity_context_tokens": 2048,
        "kv_capacity_bytes": 1048576,
        "kv_bytes_per_token": 512,
        "weights_bytes": 4096,
        "source": "vllm-prometheus-cache-config-v1",
        "reevaluations": 1,
        "passed": false
      },
      "passed": false
    }
    """.utf8)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let report = try decoder.decode(QualificationReport.self, from: data)
    #expect(report.model == "Qwen/example")
    #expect(report.contextReevaluation.status == .reduced)
    #expect(report.contextReevaluation.effectiveContextTokens == 2048)
    #expect(report.phaseProfile?.prefill.ttft.meanMs == 125.5)
    #expect(report.phaseProfile?.decode.tpot.meanMs == 16.1)
    #expect(report.phaseProfile?.decode.tokensPerSecond == 62.0)
    #expect(report.phaseProfile?.peakMemoryBytes == 17_179_869_184)
    #expect(report.modelMemoryFit?.contextTokens == 262_144)
    #expect(report.modelMemoryFit?.fits == true)
    #expect(report.hasValidPhaseEvidence)
    #expect(report.hasValidMemoryFitEvidence)
    #expect(report.hasValidQualityEvidence)
    #expect(report.requestedModes == ["text"])
    #expect(!report.passed)
}

@Test func decodesAndValidatesArtifactAdmissionEvidence() throws {
    let data = Data("""
    {
      "schema_version": 1,
      "model": "Qwen/Qwen3.8-Flash-Next",
      "artifact_bytes": 112742891520,
      "estimated_resident_bytes": 112742891520,
      "memory_hard_ceiling_bytes": 150323855360,
      "disk_free_bytes": 214748364800,
      "disk_required_bytes": 118380036096,
      "fits_memory": true,
      "fits_disk": true,
      "eligible": true
    }
    """.utf8)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let report = try decoder.decode(ArtifactAdmissionReport.self, from: data)
    #expect(report.hasValidEvidence)
    #expect(report.model == "Qwen/Qwen3.8-Flash-Next")
    #expect(report.isValidEvidence(forModel: "Qwen/Qwen3.8-Flash-Next"))
    #expect(!report.isValidEvidence(forModel: "Qwen/another-model"))
    #expect(report.estimatedResidentBytes == 112_742_891_520)
}
