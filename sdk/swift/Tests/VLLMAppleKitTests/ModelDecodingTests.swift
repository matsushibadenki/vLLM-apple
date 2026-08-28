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
