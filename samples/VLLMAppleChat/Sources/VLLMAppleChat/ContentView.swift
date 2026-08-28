import SwiftUI

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
