import CryptoKit
import Foundation
import Testing
@testable import VLLMAppleKit

@Test func macReleaseVerifierBindsArchiveNotaryAndSourceEvidence() throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let archive = root.appending(path: "VLLMAppleChat-notarized-arm64.zip")
    let archiveData = Data("bounded-archive".utf8)
    try archiveData.write(to: archive)
    let report = Data(#"{"id":"submission-1","status":"Accepted"}"#.utf8)
    try report.write(to: root.appending(path: "notary-result.json"))
    func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    let componentHash = String(repeating: "c", count: 64)
    let manifest: [String: Any] = [
        "schema_version": 1,
        "created_at": "2026-08-31T00:00:00+00:00",
        "source": ["commit": String(repeating: "a", count: 40)],
        "build": ["run_id": "123", "runner_image": "macos-15"],
        "notarization": [
            "id": "submission-1", "status": "Accepted",
            "report_sha256": hash(report), "report_size_bytes": report.count
        ],
        "artifact": [
            "filename": archive.lastPathComponent,
            "sha256": hash(archiveData), "size_bytes": archiveData.count,
            "bundle_identifier": "dev.vllm-apple.chat", "bundle_version": "1",
            "short_version": "0.1.0", "minimum_system_version": "13.0",
            "components": [
                ["path": "VLLMAppleChat.app/Contents/MacOS/VLLMAppleChat", "sha256": componentHash, "size_bytes": 1],
                ["path": "VLLMAppleChat.app/Contents/MacOS/vllm-appled", "sha256": componentHash, "size_bytes": 1]
            ]
        ]
    ]
    let manifestData = try JSONSerialization.data(withJSONObject: manifest)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let decoded = try decoder.decode(MacReleaseManifest.self, from: manifestData)
    #expect(decoded.artifact.components.count == 2)
    try manifestData.write(to: root.appending(path: "release-manifest-v1.json"))
    let verified = try MacReleaseVerifier().verify(directoryURL: root)
    #expect(verified.source.commit == String(repeating: "a", count: 40))

    try Data(repeating: 0x41, count: archiveData.count).write(to: archive)
    #expect(throws: MacReleaseVerificationError.checksumMismatch) {
        try MacReleaseVerifier().verify(directoryURL: root)
    }
}
