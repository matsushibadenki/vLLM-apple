import SwiftUI

@main
struct SandboxOptimizerApp: App {
    @StateObject private var model = SandboxOptimizerModel()

    var body: some Scene {
        WindowGroup {
            SandboxContentView(model: model)
                .frame(minWidth: 560, minHeight: 520)
        }
    }
}
