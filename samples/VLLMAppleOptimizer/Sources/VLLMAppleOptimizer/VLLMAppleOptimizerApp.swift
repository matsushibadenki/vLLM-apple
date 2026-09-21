import SwiftUI

@main
struct VLLMAppleOptimizerApp: App {
    @StateObject private var model = OptimizerAppModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .frame(minWidth: 760, minHeight: 620)
        }
        .windowResizability(.contentMinSize)
    }
}
