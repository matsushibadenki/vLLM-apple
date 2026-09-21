import Darwin
import Foundation

struct ProcessResult: Sendable {
    let status: Int32
    let standardOutput: Data
    let standardError: Data
    let cancelled: Bool
}

final class ProcessCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var requested = false

    func attach(_ process: Process) {
        lock.lock()
        self.process = process
        lock.unlock()
    }

    func cancel() {
        lock.lock()
        requested = true
        let current = process
        lock.unlock()
        current?.terminate()
    }

    var isRequested: Bool {
        lock.lock()
        defer { lock.unlock() }
        return requested
    }

    func pause() { send(signal: SIGUSR1) }
    func resume() { send(signal: SIGUSR2) }

    private func send(signal: Int32) {
        lock.lock()
        let identifier = process?.processIdentifier
        lock.unlock()
        if let identifier { Darwin.kill(identifier, signal) }
    }
}

private final class BoundedData: @unchecked Sendable {
    private let limit: Int
    private let lock = NSLock()
    private var storage = Data()
    private var overflowed = false

    init(limit: Int) { self.limit = limit }

    func append(_ data: Data) {
        lock.lock()
        defer { lock.unlock() }
        let remaining = max(0, limit - storage.count)
        storage.append(data.prefix(remaining))
        overflowed = overflowed || data.count > remaining
    }

    func value() -> Data {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    func didOverflow() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return overflowed
    }
}

enum ProcessRunnerError: LocalizedError {
    case unsafeExecutable, timedOut, outputTooLarge

    var errorDescription: String? {
        switch self {
        case .unsafeExecutable: "The optimizer executable is missing or unsafe."
        case .timedOut: "The optimizer did not finish before the timeout."
        case .outputTooLarge: "The optimizer produced more than 1 MiB of output."
        }
    }
}

enum BoundedProcessRunner {
    static func run(
        executable: URL,
        arguments: [String],
        timeout: TimeInterval,
        cancellation: ProcessCancellation
    ) throws -> ProcessResult {
        let values = try executable.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true,
              values.isSymbolicLink != true,
              FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw ProcessRunnerError.unsafeExecutable
        }

        let process = Process()
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        let output = BoundedData(limit: 1_048_576)
        let error = BoundedData(limit: 1_048_576)
        outputPipe.fileHandleForReading.readabilityHandler = { handle in output.append(handle.availableData) }
        errorPipe.fileHandleForReading.readabilityHandler = { handle in error.append(handle.availableData) }
        process.executableURL = executable
        process.arguments = arguments
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        cancellation.attach(process)
        try process.run()
        if cancellation.isRequested { process.terminate() }

        let deadline = Date().addingTimeInterval(timeout)
        var timedOut = false
        while process.isRunning {
            if Date() >= deadline {
                timedOut = true
                process.terminate()
                break
            }
            Thread.sleep(forTimeInterval: 0.05)
        }
        process.waitUntilExit()
        outputPipe.fileHandleForReading.readabilityHandler = nil
        errorPipe.fileHandleForReading.readabilityHandler = nil
        output.append(outputPipe.fileHandleForReading.readDataToEndOfFile())
        error.append(errorPipe.fileHandleForReading.readDataToEndOfFile())
        if timedOut { throw ProcessRunnerError.timedOut }
        if output.didOverflow() || error.didOverflow() { throw ProcessRunnerError.outputTooLarge }
        return ProcessResult(
            status: process.terminationStatus,
            standardOutput: output.value(),
            standardError: error.value(),
            cancelled: process.terminationReason == .uncaughtSignal
        )
    }
}
