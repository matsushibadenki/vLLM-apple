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

@Test func decodesHardwareThermalAndPowerStateWithLegacyFallback() throws {
    let current = Data("""
    {
      "platform": "Darwin",
      "architecture": "arm64",
      "soc": "Apple M4",
      "physical_cpu_count": 10,
      "logical_cpu_count": 10,
      "gpu_core_count": 10,
      "memory": {
        "total_bytes": 34359738368,
        "available_bytes": 17179869184,
        "process_resident_bytes": 1048576,
        "pressure": "normal",
        "source": "sysctl"
      },
      "is_apple_silicon": true,
      "os_version": "26.6.2",
      "thermal_state": "fair",
      "power_mode": "low_power"
    }
    """.utf8)
    let legacy = Data("""
    {
      "platform": "Darwin",
      "architecture": "arm64",
      "soc": "Apple M4",
      "physical_cpu_count": 10,
      "logical_cpu_count": 10,
      "gpu_core_count": 10,
      "memory": {
        "total_bytes": 34359738368,
        "available_bytes": 17179869184,
        "process_resident_bytes": 1048576,
        "pressure": "normal",
        "source": "sysctl"
      },
      "is_apple_silicon": true,
      "os_version": "26.6.2"
    }
    """.utf8)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase

    let detected = try decoder.decode(HardwareInfo.self, from: current)
    let oldProfile = try decoder.decode(HardwareInfo.self, from: legacy)

    #expect(detected.thermalState == .fair)
    #expect(detected.powerMode == .lowPower)
    #expect(oldProfile.thermalState == nil)
    #expect(oldProfile.powerMode == nil)
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
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let event = try decoder.decode(RuntimeEvent.self, from: data)
    #expect(event.eventID == "42")
    #expect(event.payload["state"] == .string("ready"))
}

@Test func decodesDevicePlacementStateAndTypedEvent() throws {
    let stateData = Data("""
    {
      "enabled": true,
      "active_plan_id": "0123456789abcdef01234567",
      "pending_plan_id": null,
      "placement_count": 1,
      "valid_until_unix_seconds": 1800000000,
      "placements": [{
        "operator": "attention",
        "phase": "decode",
        "precision": "int8",
        "dimensions": [1, 4096],
        "batch_size": 1,
        "backend": "coreml_draft",
        "improvement_ratio": 0.2
      }]
    }
    """.utf8)
    let state = try JSONDecoder().decode(DevicePlacementState.self, from: stateData)
    #expect(state.hasValidEvidence)
    #expect(state.placements.first?.operatorName == "attention")
    #expect(state.placements.first?.backend == "coreml_draft")

    let event = RuntimeEvent(
        schemaVersion: 1,
        eventID: "45",
        type: "runtime.device_placement",
        timestamp: "2026-09-14T00:00:00Z",
        payload: [
            "status": .string("applied"),
            "plan_id": .string("0123456789abcdef01234567")
        ]
    )
    #expect(event.devicePlacement?.status == "applied")
    #expect(event.devicePlacement?.planID == "0123456789abcdef01234567")
}

@Test func rejectsInconsistentDevicePlacementEvidence() throws {
    let data = Data("""
    {
      "enabled": true,
      "active_plan_id": "0123456789abcdef01234567",
      "pending_plan_id": null,
      "placement_count": 2,
      "valid_until_unix_seconds": 1800000000,
      "placements": []
    }
    """.utf8)
    let state = try JSONDecoder().decode(DevicePlacementState.self, from: data)
    #expect(!state.hasValidEvidence)
}

@Test func decodesTypedDeviceContentionControlResult() throws {
    let data = Data(#"""
    {
      "accepted": true,
      "device_contention": {
        "contention_profile_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "qualified_contention_pairs": 3,
        "contention_profile_loaded": true,
        "messages": {"en":"reloaded","ja":"再読み込み","zh-Hans":"重新加载"}
      }
    }
    """#.utf8)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let result = try decoder.decode(DeviceContentionControlResult.self, from: data)
    #expect(result.accepted)
    #expect(result.deviceContention.loaded)
    #expect(result.deviceContention.qualifiedPairs == 3)
    #expect(result.deviceContention.hasValidEvidence)
    #expect(DeviceContentionControlAction.reload.rawValue == "reload")
    #expect(DeviceContentionControlAction.rollback.rawValue == "rollback")
}

@Test func decodesStrictSchedulingObservabilityAndLegacyFallback() throws {
    let data = Data(#"""
    {
      "assignments": {"cpu": 2, "vllm_metal": 0, "native_mlx": 1, "native_metal": 0, "coreml_draft": 0},
      "queue_wait_buckets": {"under_1ms": 1, "1_to_10ms": 1, "10_to_100ms": 0, "100ms_or_more": 0},
      "fallback_attempts": 1,
      "fallback_exhausted": 0,
      "contention_rejections": 0,
      "steals": 1,
      "adaptive_transitions": {"applied": 1, "deferred": 0, "ignored": 0},
      "adaptive_policy": {
        "level": 1, "maximum_active_requests": 2, "maximum_batch_size": 4,
        "pressure": "warning", "thermal": "nominal", "power": "automatic", "pending_level": null
      }
    }
    """#.utf8)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let state = try decoder.decode(SchedulingObservabilityState.self, from: data)
    #expect(state.hasValidEvidence)
    #expect(state.assignments.total == 3)
    #expect(state.queueWaitBuckets.from1To10ms == 1)
    #expect(state.adaptivePolicy.pressure == .warning)
    #expect(!state.adaptivePolicy.preferenceAvailable)
    #expect(!SchedulingObservabilityState.unavailable.available)
    var object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    var preferencePolicy = try #require(object["adaptive_policy"] as? [String: Any])
    preferencePolicy["preference"] = "high_performance"
    object["adaptive_policy"] = preferencePolicy
    let current = try decoder.decode(
        SchedulingObservabilityState.self,
        from: JSONSerialization.data(withJSONObject: object)
    )
    #expect(current.adaptivePolicy.preferenceAvailable)
    #expect(current.adaptivePolicy.preference == .highPerformance)
    #expect(current.hasValidEvidence)
    let controlData = try JSONSerialization.data(withJSONObject: [
        "accepted": true,
        "transition": "deferred",
        "scheduling_observability": object,
    ])
    let control = try decoder.decode(SchedulingPreferenceControlResult.self, from: controlData)
    #expect(control.hasValidEvidence)
    object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    object["fallback_attempts"] = -1
    let invalid = try decoder.decode(
        SchedulingObservabilityState.self,
        from: JSONSerialization.data(withJSONObject: object)
    )
    #expect(!invalid.hasValidEvidence)

    object["fallback_attempts"] = 1
    var policy = try #require(object["adaptive_policy"] as? [String: Any])
    policy["maximum_batch_size"] = 8
    object["adaptive_policy"] = policy
    let mismatchedPolicy = try decoder.decode(
        SchedulingObservabilityState.self,
        from: JSONSerialization.data(withJSONObject: object)
    )
    #expect(!mismatchedPolicy.hasValidEvidence)

    object["adaptive_policy"] = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )["adaptive_policy"]
    var assignments = try #require(object["assignments"] as? [String: Any])
    assignments.removeValue(forKey: "cpu")
    object["assignments"] = assignments
    #expect(throws: Error.self) {
        try decoder.decode(
            SchedulingObservabilityState.self,
            from: JSONSerialization.data(withJSONObject: object)
        )
    }
}

@Test func decodesAndValidatesDeviceContentionEvidence() throws {
    let decoder = JSONDecoder()
    let valid = try decoder.decode(DeviceContentionState.self, from: Data("""
    {
      "contention_profile_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "qualified_contention_pairs": 3,
      "contention_profile_loaded": true
    }
    """.utf8))
    #expect(valid.hasValidEvidence)
    #expect(valid.qualifiedPairs == 3)

    let invalid = DeviceContentionState(
        profileID: String(repeating: "b", count: 64), qualifiedPairs: 0, loaded: true
    )
    #expect(!invalid.hasValidEvidence)
    #expect(DeviceContentionState.unavailable.hasValidEvidence)
}

@Test func decodesTypedOperatingStateEventAndRejectsUnknownCurrentValues() throws {
    let data = Data("""
    {
      "schema_version": 1,
      "event_id": "43",
      "type": "runtime.operating_state",
      "timestamp": "2026-09-10T00:00:00Z",
      "payload": {
        "thermal_state": "fair",
        "power_mode": "low_power",
        "previous_thermal_state": "nominal",
        "previous_power_mode": "automatic"
      }
    }
    """.utf8)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let event = try decoder.decode(RuntimeEvent.self, from: data)

    #expect(event.operatingState?.thermalState == .fair)
    #expect(event.operatingState?.powerMode == .lowPower)
    #expect(event.operatingState?.previousThermalState == .nominal)
    #expect(event.operatingState?.previousPowerMode == .automatic)

    let invalid = RuntimeEvent(
        schemaVersion: 1,
        eventID: "44",
        type: "runtime.operating_state",
        timestamp: "2026-09-10T00:00:01Z",
        payload: ["thermal_state": .string("future"), "power_mode": .string("automatic")]
    )
    #expect(invalid.operatingState == nil)
}

@Test func runtimeEventPreservesWireFormatAcrossDecoderStrategies() throws {
    let event = RuntimeEvent(
        schemaVersion: 1, eventID: "99", type: "runtime.operating_state",
        timestamp: "2026-09-10T00:00:00Z",
        payload: [
            "thermal_state": .string("unknown"),
            "power_mode": .string("automatic"),
            "future_field": .object(["nested_key": .bool(true)])
        ]
    )
    for encoding in [JSONEncoder.KeyEncodingStrategy.useDefaultKeys, .convertToSnakeCase] {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = encoding
        let data = try encoder.encode(event)
        let wire = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(wire["schema_version"] as? Int == 1)
        #expect(wire["event_id"] as? String == "99")
        for decoding in [JSONDecoder.KeyDecodingStrategy.useDefaultKeys, .convertFromSnakeCase] {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = decoding
            let decoded = try decoder.decode(RuntimeEvent.self, from: data)
            #expect(decoded == event)
            #expect(decoded.operatingState?.thermalState == .unknown)
            #expect(decoded.operatingState?.previousPowerMode == nil)
        }
    }
}

@Test func decodesStructuredStartupProgressEvent() throws {
    let data = Data("""
    {
      "schema_version": 1,
      "event_id": "45",
      "type": "runtime.startup_progress",
      "timestamp": "2026-08-30T00:00:00Z",
      "payload": {
        "schema_version": 1,
        "stage": "loading_model",
        "completed_units": 4,
        "total_units": 6,
        "message_key": "startup.loading_model",
        "percent": 66
      }
    }
    """.utf8)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let event = try decoder.decode(RuntimeEvent.self, from: data)
    #expect(event.startupProgress?.stage == "loading_model")
    #expect(event.startupProgress?.percent == 66)
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
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let event = try decoder.decode(RuntimeEvent.self, from: data)
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
    let eventDecoder = JSONDecoder()
    eventDecoder.keyDecodingStrategy = .convertFromSnakeCase
    let event = try eventDecoder.decode(RuntimeEvent.self, from: eventData)
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
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let event = try decoder.decode(RuntimeEvent.self, from: data)
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
      "backend_versions": {
        "vllm": "0.28.0",
        "vllm_metal": "0.3.0.dev1",
        "transformers": "5.15.0"
      },
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
    #expect(report.backendVersions?.vllm == "0.28.0")
    #expect(report.hasValidVLLMStackEvidence)
    #expect(report.hasVLLMStackEvidence(
        vllm: "0.28.0",
        vllmMetal: "0.3.0.dev1",
        transformers: "5.15.0"
    ))
    #expect(!report.hasVLLMStackEvidence(
        vllm: "0.28.1",
        vllmMetal: "0.3.0.dev1",
        transformers: "5.15.0"
    ))
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
    let matchingFit = QualificationModelMemoryFit(
        artifactBytes: 112_742_891_520,
        estimatedResidentBytes: 112_742_891_520,
        hardCeilingBytes: 150_323_855_360,
        contextTokens: 262_144,
        fits: true
    )
    #expect(report.isValidEvidence(forModel: report.model, memoryFit: matchingFit))
    let differentQuantization = QualificationModelMemoryFit(
        artifactBytes: 70_000_000_000,
        estimatedResidentBytes: 75_000_000_000,
        hardCeilingBytes: 150_323_855_360,
        contextTokens: 262_144,
        fits: true
    )
    #expect(!report.isValidEvidence(forModel: report.model, memoryFit: differentQuantization))
    #expect(report.estimatedResidentBytes == 112_742_891_520)
}
