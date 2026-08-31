import Darwin
import Foundation
import VLLMAppleKit

private enum ExitCode: Int32 {
    case success = 0
    case invalidArguments = 2
    case noDecodableReport = 3
    case qualificationFailed = 4
    case requiredEvidenceMissing = 5
}

private func finish(_ code: ExitCode, status: String, count: Int = 0) -> Never {
    // Do not print model identifiers: CI logs may be retained more broadly than reports.
    let payload: [String: Any] = [
        "schema_version": 1,
        "status": status,
        "decodable_reports": count
    ]
    if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
       let value = String(data: data, encoding: .utf8) {
        FileHandle.standardOutput.write(Data((value + "\n").utf8))
    }
    Foundation.exit(code.rawValue)
}

let arguments = CommandLine.arguments
guard arguments.count >= 2 else {
    finish(.invalidArguments, status: "invalid_arguments")
}

let rawOptions = Array(arguments.dropFirst(2))
let knownOptions: Set<String> = [
    "--require-phase-profile", "--require-memory-fit", "--require-quality-smoke",
    "--require-text-only", "--require-artifact-admission", "--require-backend-versions"
]
let expectedPrefixes = [
    "--expected-vllm-version=", "--expected-vllm-metal-version=",
    "--expected-transformers-version="
]
let options = Set(rawOptions.filter { !$0.contains("=") })
let expectedOptions = rawOptions.filter { $0.contains("=") }
guard options.count + expectedOptions.count == rawOptions.count,
      options.isSubset(of: knownOptions),
      expectedOptions.allSatisfy({ option in
          expectedPrefixes.contains(where: { option.hasPrefix($0) })
      }),
      Set(expectedOptions.map { $0.split(separator: "=", maxSplits: 1)[0] }).count
        == expectedOptions.count else {
    finish(.invalidArguments, status: "invalid_arguments")
}

func expectedValue(_ prefix: String) -> String? {
    expectedOptions.first(where: { $0.hasPrefix(prefix) }).map {
        String($0.dropFirst(prefix.count))
    }
}

let expectedVLLM = expectedValue("--expected-vllm-version=")
let expectedVLLMMetal = expectedValue("--expected-vllm-metal-version=")
let expectedTransformers = expectedValue("--expected-transformers-version=")
let expectedStackCount = [expectedVLLM, expectedVLLMMetal, expectedTransformers]
    .compactMap { $0 }.count
guard expectedStackCount == 0 || expectedStackCount == 3 else {
    finish(.invalidArguments, status: "invalid_arguments")
}

let directory = URL(fileURLWithPath: arguments[1], isDirectory: true)
let reports = QualificationReportStore(directoryURL: directory).load()
guard let newest = reports.first else {
    finish(.noDecodableReport, status: "no_decodable_report")
}
guard newest.report.passed,
      newest.report.shutdownClean,
      newest.report.contextReevaluation.passed else {
    finish(.qualificationFailed, status: "qualification_failed", count: reports.count)
}
if options.contains("--require-phase-profile"), !newest.report.hasValidPhaseEvidence {
    finish(.requiredEvidenceMissing, status: "phase_evidence_missing", count: reports.count)
}
if options.contains("--require-memory-fit"), !newest.report.hasValidMemoryFitEvidence {
    finish(.requiredEvidenceMissing, status: "memory_fit_evidence_missing", count: reports.count)
}
if options.contains("--require-quality-smoke"), !newest.report.hasValidQualityEvidence {
    finish(.requiredEvidenceMissing, status: "quality_evidence_missing", count: reports.count)
}
if options.contains("--require-text-only"), newest.report.requestedModes != ["text"] {
    finish(.requiredEvidenceMissing, status: "text_only_evidence_missing", count: reports.count)
}
if options.contains("--require-backend-versions"), !newest.report.hasValidVLLMStackEvidence {
    finish(.requiredEvidenceMissing, status: "backend_version_evidence_missing", count: reports.count)
}
if let expectedVLLM, let expectedVLLMMetal, let expectedTransformers,
   !newest.report.hasVLLMStackEvidence(
       vllm: expectedVLLM,
       vllmMetal: expectedVLLMMetal,
       transformers: expectedTransformers
   ) {
    finish(.requiredEvidenceMissing, status: "backend_version_mismatch", count: reports.count)
}
if options.contains("--require-artifact-admission") {
    let admissionURL = directory.appending(path: "artifact-admission.json", directoryHint: .notDirectory)
    let admission = ArtifactAdmissionReportStore(fileURL: admissionURL).load()
    if admission?.isValidEvidence(
        forModel: newest.report.model,
        memoryFit: newest.report.modelMemoryFit
    ) != true {
        finish(.requiredEvidenceMissing, status: "artifact_admission_missing", count: reports.count)
    }
}
finish(.success, status: "passed", count: reports.count)
