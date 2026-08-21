import Darwin
import Foundation
import Testing
@testable import VLLMAppleKit

@Test func resourceResolverCreatesPrivateStableResources() throws {
    let root = URL(fileURLWithPath: "/tmp/vlar-\(UUID().uuidString.prefix(8))", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let support = root.appending(path: "support", directoryHint: .isDirectory)
    let temporary = root.appending(path: "tmp", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)

    let resolver = RuntimeResourceResolver(
        applicationIdentifier: "dev.vllm-apple.tests",
        daemonExecutableURL: URL(fileURLWithPath: "/bin/echo"),
        applicationSupportRoot: support,
        temporaryRoot: temporary
    )
    let first = try resolver.resolve()
    let firstToken = try String(contentsOf: first.sessionTokenURL, encoding: .utf8)
    let second = try resolver.resolve()
    let secondToken = try String(contentsOf: second.sessionTokenURL, encoding: .utf8)

    #expect(first == second)
    #expect(firstToken == secondToken)
    #expect(firstToken.trimmingCharacters(in: .whitespacesAndNewlines).count == 64)
    #expect(first.socketURL.path.utf8.count <= RuntimeResourceResolver.maximumUnixSocketPathBytes)
    let tokenAttributes = try FileManager.default.attributesOfItem(
        atPath: first.sessionTokenURL.path
    )
    let directoryAttributes = try FileManager.default.attributesOfItem(
        atPath: first.applicationSupportURL.path
    )
    #expect((tokenAttributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
    #expect((directoryAttributes[.posixPermissions] as? NSNumber)?.intValue == 0o700)
}

@Test func resourceResolverRejectsUnsafeIdentifiers() throws {
    let resolver = RuntimeResourceResolver(applicationIdentifier: "../unsafe")
    #expect(throws: RuntimeResourceError.invalidApplicationIdentifier) {
        try resolver.resolve(createSessionToken: false)
    }
}

@Test func resourceResolverAllowsExternalDaemonFallback() throws {
    let root = URL(fileURLWithPath: "/tmp/vlaf-\(UUID().uuidString.prefix(8))", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let resolver = RuntimeResourceResolver(
        applicationIdentifier: "dev.vllm-apple.fallback",
        applicationSupportRoot: root.appending(path: "support"),
        temporaryRoot: root.appending(path: "tmp")
    )
    let resources = try resolver.resolve()
    #expect(resources.daemonExecutableURL == nil)
}
