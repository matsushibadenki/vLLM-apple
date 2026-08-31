import Foundation
import Testing
@testable import VLLMAppleKit

private func runOpenSSL(_ arguments: [String]) throws -> Data {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/openssl")
    process.arguments = arguments
    let output = Pipe()
    process.standardOutput = output
    process.standardError = FileHandle.nullDevice
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else { throw CocoaError(.executableRuntimeMismatch) }
    return output.fileHandleForReading.readDataToEndOfFile()
}

@Test func nativeCMSVerifierChecksDetachedContentTrustAndSignerIdentity() throws {
    let manager = FileManager.default
    let root = manager.temporaryDirectory.appending(
        path: UUID().uuidString, directoryHint: .isDirectory
    )
    try manager.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? manager.removeItem(at: root) }
    let bundle = root.appending(path: "promotion-bundle.json")
    let key = root.appending(path: "signer.key")
    let certificate = root.appending(path: "signer.pem")
    let signature = root.appending(path: "promotion-bundle.cms")
    try Data("{\"schema_version\":1,\"passed\":true}".utf8).write(to: bundle)
    _ = try runOpenSSL([
        "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", key.path, "-out", certificate.path, "-days", "1",
        "-subj", "/CN=Swift Qualification Test Signer"
    ])
    _ = try runOpenSSL([
        "cms", "-sign", "-binary", "-md", "sha256", "-in", bundle.path,
        "-signer", certificate.path, "-inkey", key.path, "-outform", "DER",
        "-out", signature.path
    ])
    let fingerprintData = try runOpenSSL([
        "x509", "-in", certificate.path, "-noout", "-fingerprint", "-sha256"
    ])
    let fingerprint = String(decoding: fingerprintData, as: UTF8.self)
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .split(separator: "=", maxSplits: 1)[1]
        .replacingOccurrences(of: ":", with: "")
    let verifier = QualificationCMSVerifier()
    try verifier.verify(
        bundleURL: bundle,
        signatureURL: signature,
        trustedCAURL: certificate,
        expectedSignerSHA256: fingerprint
    )
    #expect(throws: QualificationCMSError.signerIdentityMismatch) {
        try verifier.verify(
            bundleURL: bundle,
            signatureURL: signature,
            trustedCAURL: certificate,
            expectedSignerSHA256: String(repeating: "0", count: 64)
        )
    }
    try Data("tampered".utf8).write(to: bundle)
    #expect(throws: QualificationCMSError.invalidSignature) {
        try verifier.verify(
            bundleURL: bundle,
            signatureURL: signature,
            trustedCAURL: certificate,
            expectedSignerSHA256: fingerprint
        )
    }
}
