import Darwin
import Foundation
import Testing
@testable import VLLMAppleKit

@Test func boundedLogBufferDropsOldestEntries() async {
    let buffer = BoundedRuntimeLogBuffer(capacity: 2)
    await buffer.append(channel: .stdout, message: "one")
    await buffer.append(channel: .stderr, message: "two")
    await buffer.append(channel: .lifecycle, message: "three")
    let entries = await buffer.snapshot()
    #expect(entries.map(\.message) == ["two", "three"])
}

@Test func managedRuntimeRestartsOnceAndReconnectsOverUnixSocket() async throws {
    let shortID = UUID().uuidString.prefix(8)
    let temporary = URL(fileURLWithPath: "/tmp/vla-\(shortID)", isDirectory: true)
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }

    let socketURL = temporary.appending(path: "runtime.sock")
    let tokenURL = temporary.appending(path: "session.token")
    let markerURL = temporary.appending(path: "crashed-once")
    let crashTriggerURL = temporary.appending(path: "trigger-crash")
    let executableURL = temporary.appending(path: "test-daemon")
    let token = String(repeating: "s", count: 32)
    try Data((token + "\n").utf8).write(to: tokenURL)
    try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tokenURL.path)

    let script = """
    #!/usr/bin/env python3
    import json
    import os
    import signal
    import socket
    import sys
    import threading
    import time

    def argument(name):
        index = sys.argv.index(name)
        return sys.argv[index + 1]

    socket_path = argument("--socket-path")
    marker_path = \(String(reflecting: markerURL.path))
    trigger_path = \(String(reflecting: crashTriggerURL.path))
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    os.chmod(socket_path, 0o600)
    server.listen(8)
    print("managed-runtime-test", file=sys.stderr, flush=True)

    if not os.path.exists(marker_path):
        open(marker_path, "xb").close()
        def crash_once():
            while not os.path.exists(trigger_path):
                time.sleep(0.01)
            os.kill(os.getpid(), signal.SIGKILL)
        threading.Thread(target=crash_once, daemon=True).start()

    payload = json.dumps({
        "api_version": "v1",
        "schema_version": 1,
        "runtime_version": "test",
        "minimum_client_version": "0.1.0",
        "status": "ready",
        "control_ready": True,
        "inference_ready": False,
    }, separators=(",", ":")).encode()
    response = (
        b"HTTP/1.1 200 OK\\r\\nContent-Type: application/json\\r\\n"
        + b"Content-Length: " + str(len(payload)).encode() + b"\\r\\n"
        + b"Connection: close\\r\\n\\r\\n" + payload
    )
    while True:
        connection, _ = server.accept()
        with connection:
            received = bytearray()
            while b"\\r\\n\\r\\n" not in received and len(received) < 16384:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                received.extend(chunk)
            connection.sendall(response)
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
    try Data().write(to: crashTriggerURL)

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
