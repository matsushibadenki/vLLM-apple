import Foundation
import Testing
@testable import VLLMAppleOptimizer

private final class OptimizerDaemonURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let (response, data) = try Self.handler!(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }
    override func stopLoading() {}
}

private func daemonTestSession() -> URLSession {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [OptimizerDaemonURLProtocol.self]
    return URLSession(configuration: configuration)
}

private func requestBody(_ request: URLRequest) throws -> Data {
    if let body = request.httpBody { return body }
    let stream = try #require(request.httpBodyStream)
    stream.open()
    defer { stream.close() }
    var result = Data()
    var buffer = [UInt8](repeating: 0, count: 4096)
    while stream.hasBytesAvailable {
        let count = stream.read(&buffer, maxLength: buffer.count)
        guard count >= 0 else { throw try #require(stream.streamError) }
        if count == 0 { break }
        result.append(buffer, count: count)
        guard result.count <= OptimizerDaemonTransport.maximumMessageBytes else {
            throw OptimizerDaemonTransportError.requestTooLarge
        }
    }
    return result
}

@Test func optimizerDaemonTransportRejectsNonLoopbackEndpoint() {
    #expect(throws: OptimizerDaemonTransportError.self) {
        try OptimizerDaemonTransport(endpoint: URL(string: "https://example.com/optimizer")!)
    }
    #expect(throws: OptimizerDaemonTransportError.self) {
        try OptimizerDaemonTransport(endpoint: URL(string: "http://127.0.0.1/optimizer?token=x")!)
    }
}

@Test func optimizerDaemonTransportBindsVersionAndRequestIdentity() async throws {
    let request = OptimizerDaemonRequest(
        operation: .plan,
        payload: Data(#"{"model_bookmark_id":"model"}"#.utf8),
        requestID: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!
    )
    OptimizerDaemonURLProtocol.handler = { urlRequest in
        #expect(urlRequest.url?.host == "127.0.0.1")
        #expect(urlRequest.httpMethod == "POST")
        #expect(urlRequest.value(forHTTPHeaderField: "Cache-Control") == "no-store")
        let received = try JSONDecoder().decode(
            OptimizerDaemonRequest.self, from: try requestBody(urlRequest)
        )
        #expect(received.schemaVersion == 1)
        #expect(received.requestID == request.requestID)
        let response = OptimizerDaemonResponse(
            schemaVersion: 1,
            requestID: received.requestID,
            succeeded: true,
            payload: Data(#"{"plan_id":"plan-1"}"#.utf8),
            errorCode: nil
        )
        return (
            HTTPURLResponse(
                url: try #require(urlRequest.url), statusCode: 200,
                httpVersion: "HTTP/1.1", headerFields: ["Content-Type": "application/json"]
            )!,
            try JSONEncoder().encode(response)
        )
    }
    let transport = try OptimizerDaemonTransport(
        endpoint: URL(string: "http://127.0.0.1:19003/v1/optimizer")!,
        session: daemonTestSession()
    )
    let response = try await transport.send(request)
    #expect(response.succeeded)
    #expect(response.requestID == request.requestID)
}

@Test func optimizerDaemonTransportRejectsOversizedRequestBeforeNetwork() async throws {
    let transport = try OptimizerDaemonTransport(
        endpoint: URL(string: "http://localhost:19003/v1/optimizer")!,
        session: daemonTestSession()
    )
    let request = OptimizerDaemonRequest(
        operation: .execute,
        payload: Data(repeating: 1, count: OptimizerDaemonTransport.maximumMessageBytes)
    )
    await #expect(throws: OptimizerDaemonTransportError.self) {
        try await transport.send(request)
    }
}

@Test func persistsAndRestoresBoundedSecurityScopedBookmark() throws {
    let suite = "vllm-apple.optimizer.tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let store = SecurityScopedBookmarkStore(defaults: defaults)
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("vllm-apple-bookmark-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    try store.save(directory, for: .model)
    let restored = try #require(store.restore(.model))
    #expect(restored.standardizedFileURL == directory.standardizedFileURL)
    #expect(defaults.data(forKey: SecurityScopedBookmarkStore.Selection.model.defaultsKey) != nil)
}

@Test func rejectsAndClearsOversizedBookmark() throws {
    let suite = "vllm-apple.optimizer.tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let store = SecurityScopedBookmarkStore(defaults: defaults)
    defaults.set(
        Data(repeating: 1, count: 1_048_577),
        forKey: SecurityScopedBookmarkStore.Selection.output.defaultsKey
    )
    #expect(store.restore(.output) == nil)
    #expect(defaults.data(forKey: SecurityScopedBookmarkStore.Selection.output.defaultsKey) == nil)
}

@Test func restoredBookmarkSupportsLargeModelFileMetadataWithoutLoadingIt() throws {
    let suite = "vllm-apple.optimizer.tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("vllm-apple-large-model-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let weight = root.appendingPathComponent("model.safetensors")
    #expect(FileManager.default.createFile(atPath: weight.path, contents: nil))
    let handle = try FileHandle(forWritingTo: weight)
    try handle.truncate(atOffset: 5 * 1_024 * 1_024 * 1_024)
    try handle.close()

    let store = SecurityScopedBookmarkStore(defaults: defaults)
    try store.save(root, for: .model)
    let restored = try #require(store.restore(.model))
    let access = restored.startAccessingSecurityScopedResource()
    defer { if access { restored.stopAccessingSecurityScopedResource() } }
    let size = try weight.resourceValues(forKeys: [.fileSizeKey]).fileSize
    #expect(size == 5 * 1_024 * 1_024 * 1_024)
}

@Test func decodesAndValidatesPlan() throws {
    let data = Data(#"{"schema_version":1,"plan_id":"plan-1","created_at":"2026-09-21T00:00:00Z","objective":"balanced","source":{"model_id":"local","path":"/model","weights_bytes":1000,"metadata_fingerprint":"0123456789abcdef","license":null},"output_path":"/output","hardware_fingerprint":"0123456789abcdef","quality_budget":{"maximum_regression":{}},"resource_budget":{"maximum_memory_bytes":2000,"maximum_disk_bytes":3000,"maximum_duration_seconds":60},"calibration":null,"candidates":[{"strategy":"mlx-affine","target_weight_bits":4,"estimated_output_bytes":500,"required_disk_bytes":600,"estimated_peak_memory_bytes":700,"estimated_duration_seconds":10,"within_budget":true,"executable":true,"blocking_reasons":[]}],"warnings":[],"dry_run":true}"#.utf8)
    let plan = try OptimizationPlan.decodeValidated(data)
    #expect(plan.planID == "plan-1")
    #expect(plan.candidates.first?.targetWeightBits == 4)
}

@Test func rejectsNonDryRunContract() throws {
    let data = Data(#"{"schema_version":1,"plan_id":"plan-1","created_at":"2026-09-21T00:00:00Z","objective":"memory","source":{"model_id":"local","path":"/model","weights_bytes":1000,"metadata_fingerprint":"0123456789abcdef","license":null},"output_path":"/output","hardware_fingerprint":"0123456789abcdef","resource_budget":{"maximum_memory_bytes":2000,"maximum_disk_bytes":3000,"maximum_duration_seconds":null},"candidates":[{"strategy":"mlx-affine","target_weight_bits":8,"estimated_output_bytes":500,"required_disk_bytes":600,"estimated_peak_memory_bytes":700,"estimated_duration_seconds":null,"within_budget":true,"executable":false,"blocking_reasons":[]}],"warnings":[],"dry_run":false}"#.utf8)
    #expect(throws: PlanValidationError.self) { try OptimizationPlan.decodeValidated(data) }
}

@Test func validatesCompletedExportReport() throws {
    let hash = String(repeating: "a", count: 64)
    let data = Data(#"{"schema_version":1,"worker_result":{"state":"completed","output_path":"/output","output_bytes":123,"file_count":2,"output_hash":"\#(hash)","elapsed_milliseconds":42,"peak_child_rss_bytes":456,"error_code":null},"artifact_manifest":{"artifact_id":"sha256-aaaaaaaaaaaaaaaa","source_hash":"bbbbbbbbbbbbbbbb","output_hash":"\#(hash)","output_bytes":123,"transforms":[{"type":"affine_quantization","backend":"mlx-lm","weight_bits":4,"group_size":64}],"tool_versions":{"mlx-lm":"0.29.1"},"evaluation":{},"license":"apache-2.0"},"manifest_path":"/output.manifest.json"}"#.utf8)
    let report = try ExportReport.decodeValidated(data)
    #expect(report.workerResult.fileCount == 2)
    #expect(report.artifactManifest.transforms.first?.weightBits == 4)
}

@Test func validatesQualityComparisonAndUntestedCapabilities() throws {
    let hash = String(repeating: "c", count: 64)
    let data = Data(#"{"schema_version":1,"created_at":"2026-09-22T00:00:00Z","dataset_fingerprint":"\#(hash)","baseline_model_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","candidate_model_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","approved":true,"slices":[{"domain":"general","language":"ja","baseline_perplexity":2.0,"candidate_perplexity":2.01,"relative_regression":0.005,"maximum_regression":0.02,"passed":true}],"untested_capabilities":["long_context"]}"#.utf8)
    let report = try QualityComparisonReport.decodeValidated(data)
    #expect(report.approved)
    #expect(report.untestedCapabilities == ["long_context"])
}

@Test func validatesDeterministicGenerationComparison() throws {
    let data = Data(#"{"schema_version":1,"created_at":"2026-09-22T00:00:00Z","dataset_fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","baseline_model_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","candidate_model_hash":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","approved":true,"samples":[{"sample_id":"ja-1","domain":"general","language":"ja","token_agreement":1.0,"minimum_token_agreement":1.0,"baseline_expectation_score":1.0,"candidate_expectation_score":1.0,"expectation_regression":0.0,"maximum_expectation_regression":0.0,"passed":true}],"untested_capabilities":["long_context"]}"#.utf8)
    let report = try GenerationComparisonReport.decodeValidated(data)
    #expect(report.approved)
    #expect(report.samples.first?.tokenAgreement == 1)
}
