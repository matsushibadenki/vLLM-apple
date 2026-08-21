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

