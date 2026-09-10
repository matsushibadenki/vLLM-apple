import Foundation
import Testing
@testable import VLLMAppleKit

private struct LegacyClient: VLLMAppleRuntimeClient {
    func hardware() async throws -> HardwareInfo { throw RuntimeClientError.invalidResponse }
    func runtimeProfile() async throws -> RuntimeProfile { throw RuntimeClientError.invalidResponse }
    func memoryBudget() async throws -> MemoryBudget { throw RuntimeClientError.invalidResponse }
    func kvCalibration() async throws -> KVCalibrationProvenance { throw RuntimeClientError.invalidResponse }
    func nativeV2Tuning() async throws -> NativeV2TuningState { throw RuntimeClientError.invalidResponse }
    func controlNativeV2Tuning(_ action: NativeV2TuningControlAction) async throws -> NativeV2TuningControlResult {
        throw RuntimeClientError.invalidResponse
    }
    func restoreNativeV2Tuning(profileID: String) async throws -> NativeV2TuningControlResult {
        throw RuntimeClientError.invalidResponse
    }
    func health() async throws -> HealthStatus { throw RuntimeClientError.invalidResponse }
    func chat(_ request: ChatRequest) async throws -> ChatResponse { throw RuntimeClientError.invalidResponse }
    func streamChat(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
    func runtimeEvents(afterEventID: String?) -> AsyncThrowingStream<RuntimeEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}

@Test func legacyClientRetainsConformanceWithoutPreviewImplementation() async throws {
    let client: any VLLMAppleRuntimeClient = LegacyClient()
    let result = try await client.executionPlanPreview().validated()
    #expect(!result.available)
    #expect(result.reason == "client_preview_unsupported")
    #expect(result.plan == nil)
}

private func decodePreview(_ object: [String: Any]) throws -> ExecutionPlanPreviewResult {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(
        ExecutionPlanPreviewResult.self,
        from: JSONSerialization.data(withJSONObject: object)
    ).validated()
}

@Test func pythonGeneratedPreviewIsAcceptedBySwift() throws {
    let url = try #require(Bundle.module.url(
        forResource: "execution-plan-preview", withExtension: "json", subdirectory: "Fixtures"
    ))
    let object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    let result = try decodePreview(object)
    #expect(result.available)
    #expect(result.plan?.modelId == "test/model")
    #expect(result.plan?.dryRun == true)
    #expect(result.plan?.decisionReasons.contains("thermal:unknown") == true)
}

private final class PreviewURLProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let url = request.url else { return }
        let valid = url.path == "/v1/execution-plan/preview"
            && request.httpMethod == "GET"
            && request.value(forHTTPHeaderField: "Authorization") == "Bearer preview-test-token"
        let response = HTTPURLResponse(
            url: url, statusCode: valid ? 200 : 401, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
        let body = valid
            ? #"{"schema_version":1,"available":false,"reason":"chip_profile_unavailable","plan":null}"#
            : #"{"error":{"message":"invalid test request"}}"#
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@Test func httpPreviewUsesAuthenticatedEndpointThroughProtocolDispatch() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [PreviewURLProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    let client: any VLLMAppleRuntimeClient = HTTPRuntimeClient(
        baseURL: URL(string: "http://preview.invalid")!, session: session,
        sessionToken: "preview-test-token"
    )
    let result = try await client.executionPlanPreview()
    #expect(!result.available)
    #expect(result.reason == "chip_profile_unavailable")
}

@Test func executionPreviewDecodesPlanAndRejectsUnsafeContracts() throws {
    let plan: [String: Any] = [
        "schema_version": 1, "plan_id": String(repeating: "a", count: 24), "model_id": "test/model",
        "hardware_fingerprint": "test-chip", "context_tokens": 512,
        "memory_ceiling_bytes": 1000, "estimated_peak_bytes": 900,
        "prefill": ["phase": "prefill", "backend": "vllm_metal", "batch_size": 2,
                    "state_precision": "fp16"],
        "decode": ["phase": "decode", "backend": "vllm_metal", "batch_size": 1,
                   "state_precision": "fp16"],
        "fallback_chain": ["cpu"], "decision_reasons": ["thermal:fair"], "dry_run": true
    ]
    let response: [String: Any] = ["schema_version": 1, "available": true, "plan": plan]
    let result = try decodePreview(response)
    #expect(result.plan?.prefill.batchSize == 2)
    #expect(result.plan?.decisionReasons == ["thermal:fair"])
    for (key, value) in [("dry_run", false as Any), ("estimated_peak_bytes", 1001 as Any),
                         ("context_tokens", -1 as Any), ("schema_version", 2 as Any),
                         ("plan_id", "short" as Any), ("decision_reasons", [] as [String]),
                         ("fallback_chain", ["unsupported"] as Any)] {
        var changedPlan = plan
        changedPlan[key] = value
        var changedResponse = response
        changedResponse["plan"] = changedPlan
        #expect(throws: (any Error).self) { try decodePreview(changedResponse) }
    }
    for key in ["prefill", "decode"] {
        for (field, value) in [("backend", "unsupported"), ("state_precision", "")] {
            var changedPhase = try #require(plan[key] as? [String: Any])
            changedPhase[field] = value
            var changedPlan = plan
            changedPlan[key] = changedPhase
            var changedResponse = response
            changedResponse["plan"] = changedPlan
            #expect(throws: (any Error).self) { try decodePreview(changedResponse) }
        }
    }
}

@Test func executionPreviewPreservesUnavailableReasonAndRejectsContradictions() throws {
    let unavailable: [String: Any] = [
        "schema_version": 1, "available": false, "reason": "chip_profile_unavailable"
    ]
    #expect(try decodePreview(unavailable).reason == "chip_profile_unavailable")
    var invalid = unavailable
    invalid["available"] = true
    #expect(throws: (any Error).self) { try decodePreview(invalid) }
    invalid = unavailable
    invalid.removeValue(forKey: "reason")
    #expect(throws: (any Error).self) { try decodePreview(invalid) }
}
