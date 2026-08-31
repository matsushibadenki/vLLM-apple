import Foundation
import CryptoKit
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

@Test func promotionBundleRecomputesEvidenceAndRejectsTampering() throws {
    let manager = FileManager.default
    let root = manager.temporaryDirectory.appending(
        path: UUID().uuidString, directoryHint: .isDirectory
    )
    try manager.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? manager.removeItem(at: root) }
    let qualificationData = Data("""
    {
      "schema_version":1,"model":"Qwen/example","backend":"vllm_metal",
      "backend_versions":{"vllm":"0.28.0","vllm_metal":"0.3.0","transformers":"5.15.0"},
      "requested_modes":["text"],"load_seconds":1,"shutdown_clean":true,
      "promotion_probe":{"passed":true},"phase_profile":null,"model_memory_fit":null,
      "quality_smoke":null,"soak":{"passed":true},
      "context_reevaluation":{"enabled":false,"status":"unavailable",
      "configured_context_tokens":null,"effective_context_tokens":null,
      "capacity_context_tokens":null,"kv_capacity_bytes":null,"kv_bytes_per_token":null,
      "weights_bytes":null,"source":null,"reevaluations":0,"passed":true},"passed":true
    }
    """.utf8)
    let preflightData = Data("{\"schema_version\":1,\"eligible\":true}".utf8)
    let qualificationURL = root.appending(path: "qualification.json")
    let preflightURL = root.appending(path: "preflight.json")
    try qualificationData.write(to: qualificationURL)
    try preflightData.write(to: preflightURL)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let qualification = try decoder.decode(QualificationReport.self, from: qualificationData)
    func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    let evidence = [
        "preflight.json": digest(preflightData),
        "qualification.json": digest(qualificationData)
    ]
    let body: [String: Any] = [
        "schema_version": 1,
        "model_sha256": digest(Data("Qwen/example".utf8)),
        "backend": "vllm_metal",
        "requested_modes": ["text"],
        "backend_versions": [
            "vllm": "0.28.0", "vllm_metal": "0.3.0", "transformers": "5.15.0"
        ],
        "evidence_sha256": evidence,
        "passed": true
    ]
    let canonical = try JSONSerialization.data(
        withJSONObject: body, options: [.sortedKeys, .withoutEscapingSlashes]
    )
    var bundle = body
    bundle["bundle_id"] = digest(canonical)
    let bundleURL = root.appending(path: "promotion-bundle.json")
    try JSONSerialization.data(withJSONObject: bundle, options: [.sortedKeys]).write(to: bundleURL)
    let store = QualificationPromotionBundleStore(fileURL: bundleURL)
    #expect(store.loadValidated(reportsDirectoryURL: root, qualification: qualification) != nil)

    let importedRoot = manager.temporaryDirectory.appending(
        path: UUID().uuidString, directoryHint: .isDirectory
    )
    defer { try? manager.removeItem(at: importedRoot) }
    let imported = try QualificationPromotionBundleImporter(
        destinationRootURL: importedRoot
    ).importDirectory(root)
    #expect(imported.lastPathComponent == bundle["bundle_id"] as? String)
    #expect(QualificationReportStore(directoryURL: importedRoot).load().count == 1)
    #expect(throws: QualificationPromotionImportError.destinationExists) {
        try QualificationPromotionBundleImporter(
            destinationRootURL: importedRoot
        ).importDirectory(root)
    }

    try Data("tampered".utf8).write(to: preflightURL)
    #expect(store.loadValidated(reportsDirectoryURL: root, qualification: qualification) == nil)
}
