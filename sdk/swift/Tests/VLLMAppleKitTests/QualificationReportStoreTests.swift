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

private let artifactAdmissionPayload = Data("""
{
  "schema_version": 1,
  "model": "Qwen/Qwen3.8-Flash-Next",
  "artifact_bytes": 100,
  "estimated_resident_bytes": 120,
  "memory_hard_ceiling_bytes": 200,
  "disk_free_bytes": 200,
  "disk_required_bytes": 105,
  "fits_memory": true,
  "fits_disk": true,
  "eligible": true
}
""".utf8)

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

@Test func artifactAdmissionStoreIsBoundedAndRejectsSymlinks() throws {
    let manager = FileManager.default
    let directory = manager.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    try manager.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? manager.removeItem(at: directory) }

    let report = directory.appending(path: "artifact-admission.json")
    try artifactAdmissionPayload.write(to: report)
    #expect(ArtifactAdmissionReportStore(fileURL: report).load()?.hasValidEvidence == true)

    let link = directory.appending(path: "linked-admission.json")
    try manager.createSymbolicLink(at: link, withDestinationURL: report)
    #expect(ArtifactAdmissionReportStore(fileURL: link).load() == nil)
    #expect(ArtifactAdmissionReportStore(fileURL: report, maximumFileBytes: 8).load() == nil)
}
