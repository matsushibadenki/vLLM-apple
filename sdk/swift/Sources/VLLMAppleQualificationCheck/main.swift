import Darwin
import Foundation
import VLLMAppleKit

private enum ExitCode: Int32 {
    case success = 0
    case invalidArguments = 2
    case noDecodableReport = 3
    case qualificationFailed = 4
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
guard arguments.count == 2 else {
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
finish(.success, status: "passed", count: reports.count)
