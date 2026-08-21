import Darwin
import Foundation
import Testing
@testable import VLLMAppleKit

private func repositoryRoot() throws -> URL {
    var candidate = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    for _ in 0..<8 {
        if FileManager.default.fileExists(
            atPath: candidate.appending(path: "vllm_apple").path
        ) {
            return candidate
        }
        candidate.deleteLastPathComponent()
    }
    throw ManagedRuntimeError.daemonNotExecutable
}

@Test func boundedLogBufferDropsOldestEntries() async {
    let buffer = BoundedRuntimeLogBuffer(capacity: 2)
    await buffer.append(channel: .stdout, message: "one")
    await buffer.append(channel: .stderr, message: "two")
    await buffer.append(channel: .lifecycle, message: "three")
    let entries = await buffer.snapshot()
    #expect(entries.map(\.message) == ["two", "three"])
}

@Test func managedRuntimeRestartsOnceAndReconnectsOverUnixSocket() async throws {
    let root = try repositoryRoot()
    let shortID = UUID().uuidString.prefix(8)
    let temporary = URL(fileURLWithPath: "/tmp/vla-\(shortID)", isDirectory: true)
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }

    let socketURL = temporary.appending(path: "runtime.sock")
    let tokenURL = temporary.appending(path: "session.token")
    let markerURL = temporary.appending(path: "crashed-once")
    let executableURL = temporary.appending(path: "test-daemon")
    let token = String(repeating: "s", count: 32)
    try Data((token + "\n").utf8).write(to: tokenURL)
    try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tokenURL.path)

    let escapedRoot = root.path.replacingOccurrences(of: "'", with: "'\\''")
    let escapedMarker = markerURL.path.replacingOccurrences(of: "'", with: "'\\''")
    let script = """
    #!/bin/sh
    export PYTHONPATH='\(escapedRoot)'
    echo managed-runtime-test >&2
    if [ ! -f '\(escapedMarker)' ]; then
      touch '\(escapedMarker)'
      (sleep 2; kill -9 $$) &
    fi
    exec python3 -m vllm_apple.daemon "$@"
    """
    try Data(script.utf8).write(to: executableURL)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executableURL.path)

    let runtime = try ManagedRuntime(
        executableURL: executableURL,
        port: 0,
        socketPath: socketURL.path,
        sessionTokenFileURL: tokenURL,
        restartPolicy: .onFailure(maxAttempts: 1, delay: .milliseconds(50)),
        logCapacity: 20
    )
    do {
        try await runtime.start(timeout: .seconds(5))
    } catch {
        let startupLogs = await runtime.logs.snapshot()
        print("ManagedRuntime startup logs: \(startupLogs)")
        throw error
    }
    #expect(try await runtime.client.health().controlReady)

    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: .seconds(8))
    var recovered = false
    while clock.now < deadline {
        if await runtime.restartAttempts == 1,
           let health = try? await runtime.client.health(),
           health.controlReady {
            recovered = true
            break
        }
        try await Task.sleep(for: .milliseconds(100))
    }
    #expect(await runtime.restartAttempts == 1)
    #expect(recovered)
    let logs = await runtime.logs.snapshot()
    #expect(logs.contains { $0.message.contains("managed-runtime-test") })
    #expect(logs.contains { $0.message.contains("Restarting runtime") })
    await runtime.stop()
    #expect(await !runtime.isRunning)
}
