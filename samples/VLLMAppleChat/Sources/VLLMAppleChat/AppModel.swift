import Foundation
import VLLMAppleKit

struct TranscriptMessage: Identifiable, Equatable {
    enum Role: String {
        case user
        case assistant
    }

    let id: UUID
    let role: Role
    var content: String

    init(id: UUID = UUID(), role: Role, content: String) {
        self.id = id
        self.role = role
        self.content = content
    }
}

enum ConnectionPhase: Equatable {
    case disconnected
    case connecting
    case ready
    case degraded
    case failed

    var localizationKey: String {
        switch self {
        case .disconnected: "status.disconnected"
        case .connecting: "status.connecting"
        case .ready: "status.ready"
        case .degraded: "status.degraded"
        case .failed: "status.failed"
        }
    }
}

@MainActor
final class AppModel: ObservableObject {
    private static let maximumTranscriptMessages = 200
    private static let maximumPromptCharacters = 32_768
    private static let maximumContextCharacters = 65_536
    private static let maximumResponseCharacters = 131_072

    @Published var phase: ConnectionPhase = .disconnected
    @Published var modelID = "local-model"
    @Published var prompt = ""
    @Published private(set) var messages: [TranscriptMessage] = []
    @Published private(set) var isSending = false
    @Published private(set) var errorKey: String?
    @Published private(set) var detail = ""
    @Published private(set) var transportLabel = "HTTP · 127.0.0.1:8000"

    private let resolver: RuntimeResourceResolver
    private var client: (any VLLMAppleRuntimeClient)?
    private var managedRuntime: ManagedRuntime?
    private var eventTask: Task<Void, Never>?
    private var streamTask: Task<Void, Never>?

    init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        let daemonURL = environment["VLLM_APPLE_DAEMON_PATH"].map {
            URL(fileURLWithPath: $0)
        }
        resolver = RuntimeResourceResolver(
            applicationIdentifier: "dev.vllm-apple.chat",
            daemonExecutableURL: daemonURL
        )
    }

    func connect() async {
        guard phase != .connecting else { return }
        await shutdown(clearStatus: false)
        phase = .connecting
        errorKey = nil
        detail = ""

        do {
            let resources = try resolver.resolve()
            if let daemonURL = resources.daemonExecutableURL {
                let runtime = try ManagedRuntime(
                    executableURL: daemonURL,
                    port: 0,
                    socketPath: resources.socketURL.path,
                    sessionTokenFileURL: resources.sessionTokenURL,
                    restartPolicy: .onFailure(maxAttempts: 2, delay: .seconds(1))
                )
                managedRuntime = runtime
                transportLabel = "UDS · \(resources.socketURL.lastPathComponent)"
                try await runtime.start(timeout: .seconds(30))
                client = await runtime.client
            } else {
                client = HTTPRuntimeClient(baseURL: URL(string: "http://127.0.0.1:8000")!)
                transportLabel = "HTTP · 127.0.0.1:8000"
            }

            guard let client else { return }
            let health = try await client.health()
            apply(health.status)
            beginEventMonitoring(client: client)
        } catch let error as RuntimeResourceError {
            fail(key: error.messageKey, detail: String(describing: error))
        } catch let error as ManagedRuntimeError {
            fail(key: error.messageKey, detail: String(describing: error))
        } catch let error as RuntimeClientError {
            fail(key: error.messageKey, detail: String(describing: error))
        } catch {
            fail(key: "runtime.error.connection", detail: error.localizedDescription)
        }
    }

    func send() {
        let cleanPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !isSending, phase == .ready, !cleanPrompt.isEmpty else { return }
        guard cleanPrompt.count <= Self.maximumPromptCharacters else {
            report(key: "chat.error.prompt_too_long", detail: "")
            return
        }
        guard !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            report(key: "chat.error.model_required", detail: "")
            return
        }

        errorKey = nil
        detail = ""
        prompt = ""
        append(TranscriptMessage(role: .user, content: cleanPrompt))
        let assistantID = UUID()
        append(TranscriptMessage(id: assistantID, role: .assistant, content: ""))
        isSending = true

        let requestMessages = boundedRequestMessages()
        let request = ChatRequest(
            model: modelID,
            messages: requestMessages,
            temperature: 0.7,
            stream: true
        )
        guard let client else {
            report(key: "runtime.error.connection", detail: "")
            isSending = false
            return
        }

        streamTask = Task { [weak self] in
            do {
                for try await event in client.streamChat(request) {
                    try Task.checkCancellation()
                    guard let self else { return }
                    switch event {
                    case .data(let value):
                        if let delta = Self.decodeDelta(value) {
                            try self.appendDelta(delta, to: assistantID)
                        }
                    case .completed:
                        self.isSending = false
                    }
                }
                self?.isSending = false
            } catch is CancellationError {
                self?.isSending = false
            } catch let error as RuntimeClientError {
                self?.report(key: error.messageKey, detail: String(describing: error))
                self?.isSending = false
            } catch {
                self?.report(key: "chat.error.streaming", detail: error.localizedDescription)
                self?.isSending = false
            }
        }
    }

    func cancelGeneration() {
        streamTask?.cancel()
        streamTask = nil
        isSending = false
    }

    func clearTranscript() {
        cancelGeneration()
        messages.removeAll(keepingCapacity: true)
        errorKey = nil
        detail = ""
    }

    func shutdown(clearStatus: Bool = true) async {
        cancelGeneration()
        eventTask?.cancel()
        eventTask = nil
        if let managedRuntime {
            await managedRuntime.stop()
        }
        managedRuntime = nil
        client = nil
        if clearStatus {
            phase = .disconnected
        }
    }

    private func append(_ message: TranscriptMessage) {
        messages.append(message)
        if messages.count > Self.maximumTranscriptMessages {
            messages.removeFirst(messages.count - Self.maximumTranscriptMessages)
        }
    }

    private func appendDelta(_ delta: String, to id: UUID) throws {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        guard messages[index].content.count + delta.count <= Self.maximumResponseCharacters else {
            throw ChatViewModelError.responseTooLong
        }
        messages[index].content.append(delta)
    }

    private func boundedRequestMessages() -> [ChatMessage] {
        var selected: [ChatMessage] = []
        var characterCount = 0
        for message in messages.reversed() {
            guard !message.content.isEmpty else { continue }
            if characterCount + message.content.count > Self.maximumContextCharacters { break }
            characterCount += message.content.count
            selected.append(ChatMessage(role: message.role.rawValue, content: message.content))
        }
        return selected.reversed()
    }

    private static func decodeDelta(_ value: String) -> String? {
        guard let data = value.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = root["choices"] as? [[String: Any]],
              let delta = choices.first?["delta"] as? [String: Any] else {
            return nil
        }
        return delta["content"] as? String
    }

    private func beginEventMonitoring(client: any VLLMAppleRuntimeClient) {
        eventTask?.cancel()
        eventTask = Task { [weak self] in
            do {
                for try await event in client.runtimeEvents(afterEventID: nil) {
                    guard let self else { return }
                    if case .string(let value) = event.payload["state"],
                       let state = RuntimeState(rawValue: value) {
                        self.apply(state)
                    }
                }
            } catch is CancellationError {
                return
            } catch {
                self?.phase = .degraded
                self?.errorKey = "runtime.error.events"
                self?.detail = error.localizedDescription
            }
        }
    }

    private func apply(_ state: RuntimeState) {
        switch state {
        case .ready:
            phase = .ready
        case .degraded:
            phase = .degraded
        case .failed:
            phase = .failed
        case .starting, .profiling, .loadingModel, .stopping:
            phase = .connecting
        case .stopped:
            phase = .disconnected
        }
    }

    private func fail(key: String, detail: String) {
        phase = .failed
        report(key: key, detail: detail)
    }

    private func report(key: String, detail: String) {
        errorKey = key
        self.detail = String(detail.prefix(2_048))
    }
}

private enum ChatViewModelError: Error {
    case responseTooLong
}
