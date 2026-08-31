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
    @Published private(set) var contextWarning: RuntimeContextReevaluation?
    @Published private(set) var kvCalibration: KVCalibrationProvenance?
    @Published private(set) var nativeV2Tuning: NativeV2TuningState = .idle
    @Published private(set) var startupProgress: StartupProgress?
    @Published private(set) var qualificationReports: [QualificationReportRecord]
    @Published private(set) var verifiedPromotionReportURLs: Set<URL>
    @Published private(set) var signedPromotionReportURLs: Set<URL>
    @Published private(set) var verifiedMacRelease: MacReleaseManifest?

    private let resolver: RuntimeResourceResolver
    private let qualificationStore: QualificationReportStore
    private let promotionTrustedCAURL: URL?
    private let promotionSignerSHA256: String?
    private var client: (any VLLMAppleRuntimeClient)?
    private var managedRuntime: ManagedRuntime?
    private var eventTask: Task<Void, Never>?
    private var streamTask: Task<Void, Never>?

    init(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        qualificationStore: QualificationReportStore = QualificationReportStore()
    ) {
        let daemonURL = environment["VLLM_APPLE_DAEMON_PATH"].map {
            URL(fileURLWithPath: $0)
        }
        resolver = RuntimeResourceResolver(
            applicationIdentifier: "dev.vllm-apple.chat",
            daemonExecutableURL: daemonURL
        )
        self.qualificationStore = qualificationStore
        promotionTrustedCAURL = environment["VLLM_APPLE_PROMOTION_TRUSTED_CA"].map {
            URL(fileURLWithPath: $0)
        }
        promotionSignerSHA256 = environment["VLLM_APPLE_PROMOTION_SIGNER_SHA256"]
        qualificationReports = []
        verifiedPromotionReportURLs = []
        signedPromotionReportURLs = []
        verifiedMacRelease = nil
        reloadQualificationReports()
    }

    func connect() async {
        guard phase != .connecting else { return }
        await shutdown(clearStatus: false)
        phase = .connecting
        errorKey = nil
        detail = ""
        contextWarning = nil
        kvCalibration = nil
        nativeV2Tuning = .idle
        startupProgress = nil
        reloadQualificationReports()

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
            kvCalibration = try await client.kvCalibration()
            nativeV2Tuning = try await client.nativeV2Tuning()
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

    func controlNativeV2Tuning(_ action: NativeV2TuningControlAction) async {
        guard let client else { return }
        do {
            let result = try await client.controlNativeV2Tuning(action)
            nativeV2Tuning = result.nativeV2Tuning
        } catch let error as RuntimeClientError {
            report(key: error.messageKey, detail: String(describing: error))
        } catch {
            report(key: "runtime.error.connection", detail: error.localizedDescription)
        }
    }

    func restoreNativeV2Tuning(profileID: String) async {
        guard let client else { return }
        do {
            let result = try await client.restoreNativeV2Tuning(profileID: profileID)
            nativeV2Tuning = result.nativeV2Tuning
        } catch let error as RuntimeClientError {
            report(key: error.messageKey, detail: String(describing: error))
        } catch {
            report(key: "runtime.error.connection", detail: error.localizedDescription)
        }
    }

    func clearTranscript() {
        cancelGeneration()
        messages.removeAll(keepingCapacity: true)
        errorKey = nil
        detail = ""
    }

    func hasVerifiedPromotionBundle(for record: QualificationReportRecord) -> Bool {
        verifiedPromotionReportURLs.contains(record.fileURL)
    }

    func hasSignedPromotionBundle(for record: QualificationReportRecord) -> Bool {
        signedPromotionReportURLs.contains(record.fileURL)
    }

    func importQualificationDirectory(_ sourceURL: URL) {
        let accessed = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if accessed { sourceURL.stopAccessingSecurityScopedResource() }
        }
        do {
            if (promotionTrustedCAURL == nil) != (promotionSignerSHA256 == nil) {
                report(key: "qualification.import.trust_invalid", detail: "")
                return
            }
            let signatureRequirement = promotionTrustedCAURL.flatMap { caURL in
                promotionSignerSHA256.map {
                    QualificationPromotionSignatureRequirement(
                        trustedCAURL: caURL, expectedSignerSHA256: $0
                    )
                }
            }
            _ = try QualificationPromotionBundleImporter(
                destinationRootURL: qualificationStore.directoryURL
            ).importDirectory(sourceURL, signatureRequirement: signatureRequirement)
            errorKey = nil
            detail = ""
            reloadQualificationReports()
        } catch let error as QualificationPromotionImportError {
            report(key: "qualification.import.failed", detail: String(describing: error))
        } catch {
            report(key: "qualification.import.failed", detail: error.localizedDescription)
        }
    }

    func verifyMacReleaseDirectory(_ sourceURL: URL) {
        let accessed = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if accessed { sourceURL.stopAccessingSecurityScopedResource() }
        }
        do {
            verifiedMacRelease = try MacReleaseVerifier().verify(directoryURL: sourceURL)
            errorKey = nil
            detail = ""
        } catch {
            verifiedMacRelease = nil
            report(key: "release.verify.failed", detail: String(describing: error))
        }
    }

    private func reloadQualificationReports() {
        let reports = qualificationStore.load()
        var verified: Set<URL> = []
        var signed: Set<URL> = []
        for record in reports {
            let directory = record.fileURL.deletingLastPathComponent()
            let bundleURL = directory.appending(
                path: "promotion-bundle.json", directoryHint: .notDirectory
            )
            let store = QualificationPromotionBundleStore(fileURL: bundleURL)
            if store.loadValidated(
                reportsDirectoryURL: directory,
                qualification: record.report
            ) != nil {
                verified.insert(record.fileURL)
                if let caURL = promotionTrustedCAURL,
                   let fingerprint = promotionSignerSHA256,
                   (try? QualificationCMSVerifier().verify(
                    bundleURL: bundleURL,
                    signatureURL: directory.appending(path: "promotion-bundle.cms"),
                    trustedCAURL: caURL,
                    expectedSignerSHA256: fingerprint
                   )) != nil {
                    signed.insert(record.fileURL)
                }
            }
        }
        qualificationReports = reports
        verifiedPromotionReportURLs = verified
        signedPromotionReportURLs = signed
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
            contextWarning = nil
            kvCalibration = nil
            nativeV2Tuning = .idle
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
                    if let reevaluation = event.contextReevaluation {
                        self.contextWarning = reevaluation.status == .reduced ? reevaluation : nil
                    }
                    if let tuning = event.nativeV2Tuning {
                        self.nativeV2Tuning = tuning
                    }
                    if let progress = event.startupProgress {
                        self.startupProgress = progress
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
