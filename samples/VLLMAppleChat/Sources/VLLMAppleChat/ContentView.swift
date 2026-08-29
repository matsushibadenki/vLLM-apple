import SwiftUI
import VLLMAppleKit

struct ContentView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 232, ideal: 272, max: 320)
        } detail: {
            transcriptWorkbench
        }
        .tint(DesignTokens.cobalt)
        .task {
            if model.phase == .disconnected {
                await model.connect()
            }
        }
        .onDisappear {
            Task { await model.shutdown() }
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: DesignTokens.section) {
            VStack(alignment: .leading, spacing: DesignTokens.compact) {
                Label("app.title", systemImage: "cpu")
                    .font(DesignTokens.display(size: 15))
                Text("app.subtitle")
                    .font(.caption)
                    .foregroundStyle(DesignTokens.secondaryInk)
            }

            Divider()

            VStack(alignment: .leading, spacing: DesignTokens.standard) {
                sectionLabel("sidebar.runtime")
                StatusRow(phase: model.phase)
                LabeledContent("sidebar.transport") {
                    Text(model.transportLabel)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(model.transportLabel)
                }
                .font(.caption)
            }

            VStack(alignment: .leading, spacing: DesignTokens.compact) {
                sectionLabel("sidebar.model")
                TextField("sidebar.model.placeholder", text: $model.modelID)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("model-id")
            }

            if let errorKey = model.errorKey {
                VStack(alignment: .leading, spacing: DesignTokens.compact) {
                    Label(localized(errorKey), systemImage: "exclamationmark.triangle")
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(DesignTokens.danger)
                    if !model.detail.isEmpty {
                        Text(model.detail)
                            .font(.caption.monospaced())
                            .foregroundStyle(DesignTokens.secondaryInk)
                            .lineLimit(5)
                            .textSelection(.enabled)
                    }
                }
                .accessibilityElement(children: .combine)
            }

            if let warning = model.contextWarning {
                VStack(alignment: .leading, spacing: DesignTokens.compact) {
                    Label("context.warning.title", systemImage: "memorychip")
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(DesignTokens.warning)
                    Text(
                        String(
                            format: localized("context.warning.body"),
                            Int64(warning.configuredContextTokens),
                            Int64(warning.effectiveContextTokens)
                        )
                    )
                    .font(.caption)
                    .foregroundStyle(DesignTokens.secondaryInk)
                    .fixedSize(horizontal: false, vertical: true)
                }
                .accessibilityElement(children: .combine)
            }

            if let calibration = model.kvCalibration {
                CalibrationDiagnostic(calibration: calibration)
            }

            NativeV2TuningDiagnostic(state: model.nativeV2Tuning) { action in
                Task { await model.controlNativeV2Tuning(action) }
            }

            if !model.qualificationReports.isEmpty {
                VStack(alignment: .leading, spacing: DesignTokens.compact) {
                    sectionLabel("sidebar.qualification")
                    ForEach(model.qualificationReports.prefix(3)) { record in
                        HStack(alignment: .firstTextBaseline, spacing: DesignTokens.compact) {
                            Image(systemName: record.report.passed ? "checkmark.seal.fill" : "xmark.seal.fill")
                                .foregroundStyle(record.report.passed ? DesignTokens.success : DesignTokens.danger)
                                .accessibilityHidden(true)
                            VStack(alignment: .leading, spacing: DesignTokens.micro) {
                                Text(record.report.model)
                                    .font(.caption.weight(.medium))
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Text(record.report.passed ? "qualification.passed" : "qualification.failed")
                                    .font(.caption2)
                                    .foregroundStyle(DesignTokens.secondaryInk)
                            }
                        }
                        .help(record.fileURL.path)
                        .accessibilityElement(children: .combine)
                    }
                }
            }

            Spacer(minLength: DesignTokens.standard)

            VStack(spacing: DesignTokens.compact) {
                Button {
                    Task { await model.connect() }
                } label: {
                    Label("action.reconnect", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.phase == .connecting)

                Button(role: .destructive) {
                    Task { await model.shutdown() }
                } label: {
                    Label("action.stop_runtime", systemImage: "stop.fill")
                }
                .buttonStyle(.bordered)
                .disabled(model.phase == .disconnected)
            }
        }
        .padding(DesignTokens.standard)
        .background(DesignTokens.surface)
    }

    private var transcriptWorkbench: some View {
        VStack(spacing: 0) {
            HStack(spacing: DesignTokens.standard) {
                VStack(alignment: .leading, spacing: DesignTokens.micro) {
                    Text("chat.title")
                        .font(DesignTokens.display(size: 17))
                    Text("chat.caption")
                        .font(.caption)
                        .foregroundStyle(DesignTokens.secondaryInk)
                }
                Spacer()
                Button {
                    model.clearTranscript()
                } label: {
                    Label("action.clear", systemImage: "trash")
                }
                .buttonStyle(.borderless)
                .disabled(model.messages.isEmpty)
            }
            .padding(.horizontal, DesignTokens.standard)
            .padding(.vertical, DesignTokens.dense)

            Divider()

            Group {
                if model.messages.isEmpty {
                    EmptyTranscriptView(phase: model.phase)
                } else {
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 0) {
                                ForEach(model.messages) { message in
                                    MessageRow(message: message)
                                        .id(message.id)
                                    Divider()
                                }
                            }
                            .padding(.horizontal, DesignTokens.standard)
                        }
                        .onChange(of: model.messages) { messages in
                            if let id = messages.last?.id {
                                proxy.scrollTo(id, anchor: .bottom)
                            }
                        }
                    }
                }
            }

            Divider()
            composer
        }
        .background(DesignTokens.page)
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: DesignTokens.compact) {
            TextField("chat.prompt.placeholder", text: $model.prompt, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(2...6)
                .padding(DesignTokens.dense)
                .background(DesignTokens.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: DesignTokens.controlRadius)
                        .stroke(DesignTokens.rule, lineWidth: 1)
                }
                .accessibilityIdentifier("chat-prompt")

            HStack {
                Text("chat.prompt.hint")
                    .font(.caption)
                    .foregroundStyle(DesignTokens.secondaryInk)
                Spacer()
                if model.isSending {
                    Button {
                        model.cancelGeneration()
                    } label: {
                        Label("action.cancel", systemImage: "stop.circle")
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button {
                        model.send()
                    } label: {
                        Label("action.send", systemImage: "arrow.up")
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(model.phase != .ready || model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }
        .padding(DesignTokens.standard)
    }

    private func sectionLabel(_ key: LocalizedStringKey) -> some View {
        Text(key)
            .font(.caption2.monospaced().weight(.semibold))
            .foregroundStyle(DesignTokens.secondaryInk)
            .textCase(.uppercase)
            .tracking(0.8)
    }

    private func localized(_ key: String) -> String {
        String(localized: String.LocalizationValue(key), bundle: .module)
    }
}

private struct CalibrationDiagnostic: View {
    let calibration: KVCalibrationProvenance

    var body: some View {
        VStack(alignment: .leading, spacing: DesignTokens.compact) {
            Text("sidebar.calibration")
                .font(.caption2.monospaced().weight(.semibold))
                .foregroundStyle(DesignTokens.secondaryInk)
                .textCase(.uppercase)
                .tracking(0.8)
            Label(statusKey, systemImage: symbol)
                .font(.caption.weight(.semibold))
                .foregroundStyle(color)
            if calibration.status == .applied,
               let bytes = calibration.calibratedBytesPerToken,
               let context = calibration.maximumObservedContext {
                Text(
                    String(
                        format: localized("calibration.applied.body"),
                        bytes,
                        Int64(context)
                    )
                )
                .font(.caption2)
                .foregroundStyle(DesignTokens.secondaryInk)
                .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var statusKey: LocalizedStringKey {
        switch calibration.status {
        case .applied: "calibration.status.applied"
        case .invalid: "calibration.status.invalid"
        case .notFound: "calibration.status.not_found"
        case .disabled: "calibration.status.disabled"
        case .notConfigured: "calibration.status.not_configured"
        }
    }

    private var symbol: String {
        switch calibration.status {
        case .applied: "checkmark.shield.fill"
        case .invalid: "exclamationmark.shield.fill"
        case .notFound, .disabled, .notConfigured: "shield.lefthalf.filled"
        }
    }

    private var color: Color {
        switch calibration.status {
        case .applied: DesignTokens.success
        case .invalid: DesignTokens.warning
        case .notFound, .disabled, .notConfigured: DesignTokens.secondaryInk
        }
    }

    private func localized(_ key: String) -> String {
        String(localized: String.LocalizationValue(key), bundle: .module)
    }
}

private struct NativeV2TuningDiagnostic: View {
    let state: NativeV2TuningState
    let onControl: (NativeV2TuningControlAction) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: DesignTokens.compact) {
            Text("sidebar.native_v2_tuning")
                .font(.caption2.monospaced().weight(.semibold))
                .foregroundStyle(DesignTokens.secondaryInk)
                .textCase(.uppercase)
                .tracking(0.8)
            Label(statusKey, systemImage: symbol)
                .font(.caption.weight(.semibold))
                .foregroundStyle(color)
            if let profileID = state.profileID {
                Text(profileID)
                    .font(.caption2.monospaced())
                    .foregroundStyle(DesignTokens.secondaryInk)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(profileID)
            }
            if state.quarantinedProfiles > 0 {
                Text(
                    String(
                        format: localized("native_v2.quarantine.body"),
                        Int64(state.quarantinedProfiles)
                    )
                )
                .font(.caption2)
                .foregroundStyle(DesignTokens.warning)
                .fixedSize(horizontal: false, vertical: true)
                .help(state.latestQuarantinedProfileID ?? "")
            }
            HStack(spacing: DesignTokens.compact) {
                Button(state.enabled ? "native_v2.action.disable" : "native_v2.action.enable") {
                    onControl(state.enabled ? .disable : .enable)
                }
                .buttonStyle(.borderless)
                .disabled(state.status == .running)

                if state.status == .failed {
                    Button("native_v2.action.retry") {
                        onControl(.retry)
                    }
                    .buttonStyle(.borderless)
                    .disabled(!state.enabled)
                }
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var statusKey: LocalizedStringKey {
        switch state.status {
        case .disabled: "native_v2.status.disabled"
        case .idle: "native_v2.status.idle"
        case .waitingForIdle: "native_v2.status.waiting"
        case .running: "native_v2.status.running"
        case .applied: "native_v2.status.applied"
        case .failed: "native_v2.status.failed"
        }
    }

    private var symbol: String {
        switch state.status {
        case .disabled: "pause.circle"
        case .idle: "speedometer"
        case .waitingForIdle: "clock"
        case .running: "gauge.with.dots.needle.67percent"
        case .applied: "checkmark.circle.fill"
        case .failed: "exclamationmark.triangle.fill"
        }
    }

    private var color: Color {
        switch state.status {
        case .disabled: DesignTokens.secondaryInk
        case .idle: DesignTokens.secondaryInk
        case .waitingForIdle, .running: DesignTokens.cobalt
        case .applied: DesignTokens.success
        case .failed: DesignTokens.warning
        }
    }

    private func localized(_ key: String) -> String {
        String(localized: String.LocalizationValue(key), bundle: .module)
    }
}

private struct StatusRow: View {
    let phase: ConnectionPhase

    var body: some View {
        HStack(spacing: DesignTokens.compact) {
            Image(systemName: symbol)
                .foregroundStyle(color)
            Text(String(localized: String.LocalizationValue(phase.localizationKey), bundle: .module))
                .font(.callout.weight(.medium))
        }
        .accessibilityElement(children: .combine)
    }

    private var symbol: String {
        switch phase {
        case .ready: "checkmark.circle.fill"
        case .connecting: "clock.arrow.circlepath"
        case .degraded: "exclamationmark.circle.fill"
        case .failed: "xmark.octagon.fill"
        case .disconnected: "circle.dashed"
        }
    }

    private var color: Color {
        switch phase {
        case .ready: DesignTokens.success
        case .connecting: DesignTokens.cobalt
        case .degraded: DesignTokens.warning
        case .failed: DesignTokens.danger
        case .disconnected: DesignTokens.secondaryInk
        }
    }
}

private struct EmptyTranscriptView: View {
    let phase: ConnectionPhase

    var body: some View {
        VStack(spacing: DesignTokens.standard) {
            Image(systemName: phase == .ready ? "ellipsis.message" : "bolt.horizontal.circle")
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(phase == .ready ? DesignTokens.cobalt : DesignTokens.secondaryInk)
                .accessibilityHidden(true)
            VStack(spacing: DesignTokens.compact) {
                Text(localized(phase == .ready ? "chat.empty.ready.title" : "chat.empty.offline.title"))
                    .font(DesignTokens.display(size: 15))
                Text(localized(phase == .ready ? "chat.empty.ready.body" : "chat.empty.offline.body"))
                    .font(.callout)
                    .foregroundStyle(DesignTokens.secondaryInk)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(.horizontal, DesignTokens.reading)
        .padding(.vertical, DesignTokens.spacious)
    }

    private func localized(_ key: String) -> String {
        String(localized: String.LocalizationValue(key), bundle: .module)
    }
}

private struct MessageRow: View {
    let message: TranscriptMessage

    var body: some View {
        HStack(alignment: .top, spacing: DesignTokens.standard) {
            Image(systemName: message.role == .user ? "person.crop.circle" : "cpu")
                .font(.title3)
                .foregroundStyle(message.role == .user ? DesignTokens.secondaryInk : DesignTokens.cobalt)
                .frame(width: 24)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: DesignTokens.compact) {
                Text(localized(message.role == .user ? "chat.role.you" : "chat.role.assistant"))
                    .font(.caption.monospaced().weight(.semibold))
                    .foregroundStyle(DesignTokens.secondaryInk)
                    .textCase(.uppercase)
                if message.content.isEmpty {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("chat.generating")
                } else {
                    Text(message.content)
                        .font(.body)
                        .textSelection(.enabled)
                }
            }
        }
        .padding(.vertical, DesignTokens.standard)
    }

    private func localized(_ key: String) -> String {
        String(localized: String.LocalizationValue(key), bundle: .module)
    }
}

#Preview("Compact Mac") {
    ContentView(model: AppModel(environment: [:]))
        .frame(width: 720, height: 520)
}

#Preview("Wide Mac") {
    ContentView(model: AppModel(environment: [:]))
        .frame(width: 1180, height: 760)
}
