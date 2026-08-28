import Foundation
import Testing
@testable import VLLMAppleKit

private func qualificationPayload(model: String, passed: Bool = true) -> Data {
    Data("""
    {
      "schema_version": 1,
      "model": "\(model)",
      "backend": "vllm_metal",
      "load_seconds": 1,
      "shutdown_clean": true,
      "promotion_probe": {"passed": true},
      "soak": {"passed": true},
      "context_reevaluation": {
        "enabled": false,
        "status": "unavailable",
        "configured_context_tokens": null,
        "effective_context_tokens": null,
        "capacity_context_tokens": null,
        "kv_capacity_bytes": null,
        "kv_bytes_per_token": null,
        "weights_bytes": null,
        "source": null,
        "reevaluations": 0,
        "passed": true
      },
      "passed": \(passed)
    }
    """.utf8)
}

@Test func reportStoreLoadsNewestWithinBoundsAndSkipsUnsafeFiles() throws {
    let manager = FileManager.default
    let directory = manager.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    try manager.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? manager.removeItem(at: directory) }

    let old = directory.appending(path: "old.json")
    let newest = directory.appending(path: "new.json")
    try qualificationPayload(model: "old").write(to: old)
    try qualificationPayload(model: "new").write(to: newest)
    try manager.setAttributes([.modificationDate: Date(timeIntervalSince1970: 1)], ofItemAtPath: old.path)
    try manager.setAttributes([.modificationDate: Date(timeIntervalSince1970: 2)], ofItemAtPath: newest.path)
    try Data("not-json".utf8).write(to: directory.appending(path: "broken.json"))
    try Data(repeating: 0, count: 2_048).write(to: directory.appending(path: "oversized.json"))
    try manager.createSymbolicLink(
        at: directory.appending(path: "linked.json"),
        withDestinationURL: newest
    )

    let reports = QualificationReportStore(
        directoryURL: directory,
        maximumReports: 1,
        maximumFileBytes: 1_024
    ).load()
    #expect(reports.count == 1)
    #expect(reports.first?.report.model == "new")
}
