import SwiftUI

@main
struct VLLMAppleChatApp: App {
    @StateObject private var model = AppModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .frame(minWidth: 680, minHeight: 480)
                .onChange(of: scenePhase) { phase in
                    if phase == .background {
                        model.cancelGeneration()
                    }
                }
        }
        .defaultSize(width: 1040, height: 720)
        .commands {
            CommandGroup(after: .newItem) {
                Button("action.clear") {
                    model.clearTranscript()
                }
                .keyboardShortcut("k", modifiers: [.command, .shift])
            }
        }
    }
}
