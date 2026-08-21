import Foundation

public enum ManagedRuntimeError: Error, Sendable, Equatable {
    case alreadyRunning
    case daemonNotExecutable
    case readinessTimedOut
    case daemonExited(Int32)

    public var messageKey: String {
        switch self {
        case .alreadyRunning: "runtime.error.already_running"
        case .daemonNotExecutable: "runtime.error.daemon_not_executable"
        case .readinessTimedOut: "runtime.error.readiness_timed_out"
        case .daemonExited: "runtime.error.daemon_exited"
        }
    }
}

public actor ManagedRuntime {
    public let client: HTTPRuntimeClient
    private let executableURL: URL
    private let host: String
    private let port: UInt16
    private var process: Process?

    public init(executableURL: URL, host: String = "127.0.0.1", port: UInt16 = 8000) {
        self.executableURL = executableURL
        self.host = host
        self.port = port
        self.client = HTTPRuntimeClient(baseURL: URL(string: "http://\(host):\(port)")!)
    }

    public func start(timeout: Duration = .seconds(30)) async throws {
        guard process == nil else { throw ManagedRuntimeError.alreadyRunning }
        guard FileManager.default.isExecutableFile(atPath: executableURL.path) else {
            throw ManagedRuntimeError.daemonNotExecutable
        }
        let process = Process()
        process.executableURL = executableURL
        process.arguments = ["--host", host, "--port", String(port)]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        self.process = process

        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            if !process.isRunning {
                self.process = nil
                throw ManagedRuntimeError.daemonExited(process.terminationStatus)
            }
            if let health = try? await client.health(), health.controlReady {
                return
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        process.terminate()
        self.process = nil
        throw ManagedRuntimeError.readinessTimedOut
    }

    public func stop() async {
        guard let process else { return }
        if process.isRunning {
            process.terminate()
        }
        self.process = nil
    }

    public var isRunning: Bool {
        process?.isRunning == true
    }
}

