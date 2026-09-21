import Foundation
import Testing
@testable import VLLMAppleKit

@Test func decodesAndValidatesNumericRouteDiagnostic() throws {
    let payload = Data(#"{"schema_version":1,"valid":true,"source_format":"nvfp4_e2m1","runtime_format":"int8","compute_format":"fp16","tensor_role":"weight","route":"cached_convert","error_budget":{"maximum_absolute_error":0.01,"maximum_rmse":0.001},"fallback_reason":null,"message_key":"numeric_route_ready_for_inspection","message":"ready","language":"en"}"#.utf8)
    let result = try NumericRouteDiagnostic.decodeValidated(payload)
    #expect(result.sourceFormat == .nvfp4E2M1)
    #expect(result.route == .cachedConvert)
    #expect(result.errorBudget.maximumRMSE == 0.001)
}

@Test func rejectsInvalidNumericRouteEvidenceAndSymlink() throws {
    let invalid = Data(#"{"schema_version":2,"valid":true,"source_format":"int8","runtime_format":"int8","compute_format":"int8","tensor_role":"weight","route":"load_convert","error_budget":{"maximum_absolute_error":0,"maximum_rmse":0},"fallback_reason":null,"message_key":"numeric_route_ready_for_inspection","message":"ready","language":"en"}"#.utf8)
    #expect(throws: NumericRouteDiagnosticError.invalidEvidence) {
        try NumericRouteDiagnostic.decodeValidated(invalid)
    }
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let target = root.appending(path: "target.json")
    try invalid.write(to: target)
    let link = root.appending(path: "link.json")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
    #expect(throws: NumericRouteDiagnosticError.unsafeFile) {
        try NumericRouteDiagnosticLoader().load(fileURL: link)
    }
}
