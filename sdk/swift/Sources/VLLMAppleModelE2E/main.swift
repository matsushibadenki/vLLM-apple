import Darwin
import Foundation
import VLLMAppleKit

private struct Arguments {
    let daemon: URL
    let backend: URL
    let model: String
    let backendKind: String
    let requestModel: String
    let timeoutSeconds: Int

    init(_ values: [String]) throws {
        var options: [String: String] = [:]
        var index = 0
        while index < values.count {
            guard values[index].hasPrefix("--"), index + 1 < values.count else {
                throw E2EError.invalidArguments
            }
            options[values[index]] = values[index + 1]
            index += 2
        }
        guard let daemon = options["--daemon"],
              let backend = options["--backend"],
              let model = options["--model"],
              !model.isEmpty else {
            throw E2EError.invalidArguments
        }
        let kind = options["--backend-kind"] ?? "mlx_lm"
        guard ["mlx_lm", "vllm_metal"].contains(kind) else {
            throw E2EError.invalidArguments
        }
        let timeout = Int(options["--timeout"] ?? "600") ?? 0
        guard 1...1_800 ~= timeout else { throw E2EError.invalidArguments }
        self.daemon = URL(fileURLWithPath: daemon)
        self.backend = URL(fileURLWithPath: backend)
        self.model = model
        self.backendKind = kind
        self.requestModel = options["--request-model"] ?? "default_model"
        self.timeoutSeconds = timeout
    }
}

private enum E2EError: Error {
    case invalidArguments
    case inferenceReadinessTimedOut
    case streamProducedNoData
    case streamDidNotComplete
    case shutdownFailed
}

@main
private struct ModelE2E {
    static func main() async {
        do {
            let arguments = try Arguments(Array(CommandLine.arguments.dropFirst()))
            try await run(arguments)
        } catch {
            FileHandle.standardError.write(Data("model-e2e failed: \(error)\n".utf8))
            Darwin.exit(1)
        }
    }

    private static func run(_ arguments: Arguments) async throws {
        let identifier = "e2e-\(UUID().uuidString.prefix(8))"
        let root = FileManager.default.temporaryDirectory.appending(
            path: "vla-\(identifier)",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = try RuntimeResourceResolver(
            applicationIdentifier: identifier,
            daemonExecutableURL: arguments.daemon,
            applicationSupportRoot: root.appending(path: "support"),
            temporaryRoot: URL(fileURLWithPath: "/tmp", isDirectory: true)
        ).resolve()
        defer { try? FileManager.default.removeItem(at: resources.socketURL.deletingLastPathComponent()) }
        guard let executable = resources.daemonExecutableURL else {
            throw ManagedRuntimeError.daemonNotExecutable
        }
        let runtime = try ManagedRuntime(
            executableURL: executable,
            port: 0,
            socketPath: resources.socketURL.path,
            sessionTokenFileURL: resources.sessionTokenURL,
            daemonArguments: [
                arguments.model,
                "--backend-executable", arguments.backend.path,
                "--backend-kind", arguments.backendKind,
                "--backend-startup-timeout", String(arguments.timeoutSeconds),
                "--skip-runtime-probes",
                "--disable-metal-tuning",
                "--disable-kv-calibration",
                "--disable-native-v2-idle-tuning",
            ]
        )
        do {
            try await runtime.start(timeout: .seconds(arguments.timeoutSeconds))
        } catch {
            let logs = await runtime.logs.snapshot()
            for entry in logs {
                FileHandle.standardError.write(
                    Data("daemon[\(entry.channel.rawValue)]: \(entry.message)\n".utf8)
                )
            }
            throw error
        }
        var streamedEvents = 0
        do {
            let client = await runtime.client
            try await waitForInference(client, timeoutSeconds: arguments.timeoutSeconds)
            var completed = false
            let request = ChatRequest(
                model: arguments.requestModel,
                messages: [ChatMessage(role: "user", content: "Reply with OK.")],
                temperature: 0,
                maxTokens: 8,
                stream: true
            )
            for try await event in client.streamChat(request) {
                switch event {
                case .data: streamedEvents += 1
                case .completed: completed = true
                }
            }
            guard streamedEvents > 0 else { throw E2EError.streamProducedNoData }
            guard completed else { throw E2EError.streamDidNotComplete }
            await runtime.stop()
        } catch {
            await runtime.stop()
            throw error
        }
        let stillRunning = await runtime.isRunning
        guard !stillRunning else { throw E2EError.shutdownFailed }
        print(
            "model-e2e passed: streamed_events=\(streamedEvents) generated_text_stored=false"
        )
    }

    private static func waitForInference(
        _ client: any VLLMAppleRuntimeClient,
        timeoutSeconds: Int
    ) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(timeoutSeconds))
        while ContinuousClock.now < deadline {
            if let health = try? await client.health(), health.inferenceReady { return }
            try await Task.sleep(for: .milliseconds(250))
        }
        throw E2EError.inferenceReadinessTimedOut
    }
}
