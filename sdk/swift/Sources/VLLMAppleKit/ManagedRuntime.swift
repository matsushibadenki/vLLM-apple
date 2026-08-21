import Darwin
import Foundation

public enum ManagedRuntimeError: Error, Sendable, Equatable {
    case alreadyRunning
    case daemonNotExecutable
    case readinessTimedOut
    case daemonExited(Int32)
    case invalidSessionTokenFile

    public var messageKey: String {
        switch self {
        case .alreadyRunning: "runtime.error.already_running"
        case .daemonNotExecutable: "runtime.error.daemon_not_executable"
        case .readinessTimedOut: "runtime.error.readiness_timed_out"
        case .daemonExited: "runtime.error.daemon_exited"
        case .invalidSessionTokenFile: "runtime.error.invalid_session_token_file"
        }
    }
}

public enum RuntimeLogChannel: String, Sendable {
    case stdout
    case stderr
    case lifecycle
}

public struct RuntimeLogEntry: Sendable, Equatable {
    public let timestamp: Date
    public let channel: RuntimeLogChannel
    public let message: String
}

public actor BoundedRuntimeLogBuffer {
    public let capacity: Int
    private var entries: [RuntimeLogEntry] = []

    public init(capacity: Int = 400) {
        self.capacity = max(1, capacity)
        entries.reserveCapacity(self.capacity)
    }

    public func append(channel: RuntimeLogChannel, message: String) {
        let boundedMessage = String(message.suffix(8 * 1024))
            .trimmingCharacters(in: .newlines)
        guard !boundedMessage.isEmpty else { return }
        entries.append(RuntimeLogEntry(timestamp: Date(), channel: channel, message: boundedMessage))
        if entries.count > capacity {
            entries.removeFirst(entries.count - capacity)
        }
    }

    public func snapshot() -> [RuntimeLogEntry] {
        entries
    }
}

public enum RestartPolicy: Sendable {
    case never
    case onFailure(maxAttempts: Int, delay: Duration)
}

public actor ManagedRuntime {
    public let client: any VLLMAppleRuntimeClient
    public let logs: BoundedRuntimeLogBuffer

    private let executableURL: URL
    private let host: String
    private let port: UInt16
    private let socketPath: String?
    private let sessionTokenFileURL: URL?
    private let restartPolicy: RestartPolicy
    private var process: Process?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var monitorTask: Task<Void, Never>?
    private var stopRequested = false
    private var startupTimeout: Duration = .seconds(30)

    public private(set) var restartAttempts = 0
    public private(set) var lastTerminationStatus: Int32?

    public init(
        executableURL: URL,
        host: String = "127.0.0.1",
        port: UInt16 = 8000,
        socketPath: String? = nil,
        sessionTokenFileURL: URL? = nil,
        restartPolicy: RestartPolicy = .never,
        logCapacity: Int = 400
    ) throws {
        self.executableURL = executableURL
        self.host = host
        self.port = port
        self.socketPath = socketPath
        self.sessionTokenFileURL = sessionTokenFileURL
        self.restartPolicy = restartPolicy
        self.logs = BoundedRuntimeLogBuffer(capacity: logCapacity)

        let token: String?
        if let sessionTokenFileURL {
            guard let value = try? String(contentsOf: sessionTokenFileURL, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines), value.count >= 32 else {
                throw ManagedRuntimeError.invalidSessionTokenFile
            }
            token = value
        } else {
            token = nil
        }
        if let socketPath {
            self.client = UnixSocketRuntimeClient(socketPath: socketPath, sessionToken: token)
        } else {
            self.client = HTTPRuntimeClient(
                baseURL: URL(string: "http://\(host):\(port)")!,
                sessionToken: token
            )
        }
    }

    public func start(timeout: Duration = .seconds(30)) async throws {
        guard process == nil else { throw ManagedRuntimeError.alreadyRunning }
        guard FileManager.default.isExecutableFile(atPath: executableURL.path) else {
            throw ManagedRuntimeError.daemonNotExecutable
        }
        stopRequested = false
        restartAttempts = 0
        lastTerminationStatus = nil
        startupTimeout = timeout
        try await launchAndWaitForReadiness()
        monitorTask = Task { [weak self] in
            await self?.monitorProcess()
        }
    }

    public func stop() async {
        stopRequested = true
        monitorTask?.cancel()
        monitorTask = nil
        guard let process else {
            closeLogPipes()
            return
        }
        await terminate(process, grace: .seconds(5))
        lastTerminationStatus = process.terminationStatus
        self.process = nil
        closeLogPipes()
        await logs.append(channel: .lifecycle, message: "Runtime stop requested")
    }

    public var isRunning: Bool {
        process?.isRunning == true
    }

    private func launchAndWaitForReadiness() async throws {
        let process = Process()
        process.executableURL = executableURL
        var arguments = ["--host", host, "--port", String(port)]
        if let socketPath {
            arguments.append(contentsOf: ["--socket-path", socketPath])
        }
        if let sessionTokenFileURL {
            arguments.append(contentsOf: ["--session-token-file", sessionTokenFileURL.path])
        }
        process.arguments = arguments
        configureLogPipes(for: process)
        try process.run()
        self.process = process
        await logs.append(
            channel: .lifecycle,
            message: "Runtime process started (pid \(process.processIdentifier))"
        )

        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: startupTimeout)
        while clock.now < deadline {
            if !process.isRunning {
                self.process = nil
                closeLogPipes()
                throw ManagedRuntimeError.daemonExited(process.terminationStatus)
            }
            if let health = try? await client.health(), health.controlReady {
                return
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        await terminate(process, grace: .seconds(2))
        self.process = nil
        closeLogPipes()
        throw ManagedRuntimeError.readinessTimedOut
    }

    private func configureLogPipes(for process: Process) {
        closeLogPipes()
        let stdout = Pipe()
        let stderr = Pipe()
        let logBuffer = logs
        stdout.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            if let value = String(data: data, encoding: .utf8) {
                Task { await logBuffer.append(channel: .stdout, message: value) }
            }
        }
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            if let value = String(data: data, encoding: .utf8) {
                Task { await logBuffer.append(channel: .stderr, message: value) }
            }
        }
        process.standardOutput = stdout
        process.standardError = stderr
        stdoutPipe = stdout
        stderrPipe = stderr
    }

    private func closeLogPipes() {
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        try? stdoutPipe?.fileHandleForReading.close()
        try? stderrPipe?.fileHandleForReading.close()
        stdoutPipe = nil
        stderrPipe = nil
    }

    private func terminate(_ process: Process, grace: Duration) async {
        guard process.isRunning else { return }
        process.terminate()
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: grace)
        while process.isRunning, clock.now < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }
        if process.isRunning {
            Darwin.kill(process.processIdentifier, SIGKILL)
            while process.isRunning {
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
    }

    private func monitorProcess() async {
        while !Task.isCancelled, !stopRequested {
            guard let activeProcess = process else { return }
            while activeProcess.isRunning, !Task.isCancelled, !stopRequested {
                try? await Task.sleep(for: .milliseconds(250))
            }
            if Task.isCancelled || stopRequested { return }
            let status = activeProcess.terminationStatus
            lastTerminationStatus = status
            process = nil
            closeLogPipes()
            await logs.append(channel: .lifecycle, message: "Runtime exited with status \(status)")
            guard status != 0, await restartAfterFailure() else { return }
        }
    }

    private func restartAfterFailure() async -> Bool {
        while shouldRestart, !Task.isCancelled, !stopRequested {
            restartAttempts += 1
            await logs.append(
                channel: .lifecycle,
                message: "Restarting runtime (attempt \(restartAttempts))"
            )
            try? await Task.sleep(for: restartDelay)
            if Task.isCancelled || stopRequested { return false }
            do {
                try await launchAndWaitForReadiness()
                return true
            } catch {
                await logs.append(channel: .lifecycle, message: "Restart failed: \(error)")
            }
        }
        return false
    }

    private var shouldRestart: Bool {
        switch restartPolicy {
        case .never:
            return false
        case .onFailure(let maxAttempts, _):
            return restartAttempts < max(0, maxAttempts)
        }
    }

    private var restartDelay: Duration {
        switch restartPolicy {
        case .never: return .zero
        case .onFailure(_, let delay): return delay
        }
    }
}
